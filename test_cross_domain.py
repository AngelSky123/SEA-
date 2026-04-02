import torch
from torch.utils.data import DataLoader
import argparse
import yaml

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from utils_metrics import compute_metrics

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_config_simple():
    with open('config/default.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, Config(v) if isinstance(v, dict) else v)
    return Config(cfg_dict)


def detect_model_dims(ckpt_state, fallback_dim=256):
    in_dim = None
    dim    = fallback_dim
    if 'encoder.gat.proj.weight' in ckpt_state:
        w      = ckpt_state['encoder.gat.proj.weight']
        dim    = w.shape[0]
        in_dim = w.shape[1]
    return in_dim, dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work',   required=True)
    parser.add_argument('--target', nargs='+', default=None)
    args = parser.parse_args()

    cfg         = get_config_simple()
    target_envs = args.target if args.target else cfg.domain.target

    test_data = MMFiDataset(cfg.data.root, target_envs, cfg.data.seq_len)
    loader    = DataLoader(test_data, batch_size=32, shuffle=False,
                           num_workers=8, pin_memory=True)

    ckpt = torch.load(args.work, map_location=device, weights_only=False)

    if 'in_dim' in ckpt:
        in_dim = ckpt['in_dim']
        _, dim_ckpt = detect_model_dims(
            ckpt['model'], fallback_dim=cfg.model.dim)
        print(f"  in_dim = {in_dim}  (from checkpoint metadata)")
    else:
        in_dim, dim_ckpt = detect_model_dims(
            ckpt['model'], fallback_dim=cfg.model.dim)
        if in_dim is None:
            sample_csi, _, _ = test_data[0]
            in_dim = sample_csi.shape[-1]
            print(f"  in_dim = {in_dim}  (probed from dataset)")
        else:
            print(f"  in_dim = {in_dim}  (inferred from checkpoint weights)")

    pose_head_old = ckpt.get('pose_head_old', True)

    print(f"  dim    = {dim_ckpt}")
    print(f"  PoseHead: {'old-style' if pose_head_old else 'new-style'}")

    model = WiFiPoseModel(
        in_dim=in_dim,
        dim=dim_ckpt,
        num_joints=cfg.model.num_joints,
        pose_head_old=pose_head_old,
    ).to(device)

    result = model.load_state_dict(ckpt['model'], strict=False)
    if result.missing_keys:
        print(f"  Missing keys  ({len(result.missing_keys)}): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}")
    if not result.missing_keys and not result.unexpected_keys:
        print("  All keys matched perfectly.")

    print(f"Loaded: {args.work}")
    model.eval()

    all_pred, all_gt = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            # model 返回 8 个值，只需要 pose_s（第一个）
            outputs = model(x, x, alpha=0.0)
            pose = outputs[0]
            all_pred.append(pose.cpu())
            all_gt.append(y[:, 0].cpu())

    metrics = compute_metrics(all_pred, all_gt)
    print("\n===== RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
