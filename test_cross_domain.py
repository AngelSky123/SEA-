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

    # 修复（v4）：从 checkpoint 读取 in_dim，回退到数据集探测，再回退到 40
    ckpt = torch.load(args.work, map_location=device)
    if 'in_dim' in ckpt:
        in_dim = ckpt['in_dim']
        print(f"  in_dim = {in_dim}  (loaded from checkpoint)")
    else:
        sample_csi, _, _ = test_data[0]
        in_dim = sample_csi.shape[-1]
        print(f"  in_dim = {in_dim}  (probed from dataset, checkpoint has no in_dim)")

    model = WiFiPoseModel(
        in_dim=in_dim,
        dim=cfg.model.dim,
        num_joints=cfg.model.num_joints,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
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