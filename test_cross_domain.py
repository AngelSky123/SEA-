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
    """
    从 checkpoint state_dict 反推 in_dim、dim。

    encoder.gat.proj.weight : (dim, in_dim)

    修复（v6）：不再尝试从权重形状推断 pose_head_old。
    v5 新旧 PoseHead 输入维度均为 dim*2，权重形状完全相同，
    无法通过 head.mlp.0.weight.shape 区分。
    pose_head_old 应从 checkpoint metadata 字段读取。
    """
    in_dim = None
    dim    = fallback_dim

    if 'encoder.gat.proj.weight' in ckpt_state:
        w      = ckpt_state['encoder.gat.proj.weight']   # (dim, in_dim)
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

    # ── 反推 in_dim 和 dim ────────────────────────────────────────────
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

    # ── 反推 pose_head_old ────────────────────────────────────────────
    # 修复（v6）：v5 新旧 PoseHead 权重形状相同（均为 dim*2 输入），
    # 无法从权重反推。必须依赖 checkpoint metadata。
    # 若 metadata 中无此字段，说明是 v4 之前的旧 checkpoint，默认 True。
    pose_head_old = ckpt.get('pose_head_old', True)

    print(f"  dim    = {dim_ckpt}")
    print(f"  PoseHead: {'old-style concat(x0,xT)' if pose_head_old else 'new-style max-pool+motion'}")

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
            pose, _, _, _, _, _, _ = model(x, x, alpha=0.0)
            all_pred.append(pose.cpu())
            all_gt.append(y[:, 0].cpu())

    metrics = compute_metrics(all_pred, all_gt)
    print("\n===== RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
