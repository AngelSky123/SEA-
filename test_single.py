import torch
import argparse
import yaml

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from visualization.pose_vis import plot_pose
from utils_metrics import mpjpe, pa_mpjpe, pck

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
    parser.add_argument('--env',     default='E04')
    parser.add_argument('--subject', default='S35')
    parser.add_argument('--action',  default='A01')
    parser.add_argument('--work',    required=True, help="模型权重路径")
    args = parser.parse_args()

    cfg     = get_config_simple()
    dataset = MMFiDataset(cfg.data.root, [args.env], cfg.data.seq_len)
    model   = WiFiPoseModel(dim=cfg.model.dim).to(device)

    ckpt = torch.load(args.work, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"Loaded: {args.work}")
    print(f"Testing {args.env}-{args.subject}-{args.action}")

    sample_idx = 0
    for i, s in enumerate(dataset.samples):
        if args.subject in s["csi_dir"] and args.action in s["csi_dir"]:
            sample_idx = i
            break

    # 数据集现在返回三元组 (csi, pose_centered, root_offset)
    csi, pose, root_offset = dataset[sample_idx]
    x   = csi.unsqueeze(0).to(device)
    gt  = pose[0].to(device)           # 已中心化的第 0 帧 GT，(J, 3)

    with torch.no_grad():
        pred, _, _, _, _, _ = model(x, x, alpha=0.0)

    pred = pred[0]   # (J, 3)，同样是相对坐标

    print(f"MPJPE:    {mpjpe(pred, gt).item():.4f} m")
    print(f"PA-MPJPE: {pa_mpjpe(pred, gt).item():.4f} m")
    print(f"PCK@50mm: {pck(pred, gt, 0.05).item():.4f}")
    print(f"PCK@20mm: {pck(pred, gt, 0.02).item():.4f}")

    plot_pose(gt.cpu().numpy(), pred.cpu().numpy(), save_path="single_pose.png")
    print("Visualization saved: single_pose.png")


if __name__ == "__main__":
    main()