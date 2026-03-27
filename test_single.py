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


def detect_model_dims(ckpt_state, fallback_dim=256):
    """
    从 checkpoint state_dict 反推 in_dim、dim、pose_head_old。
    """
    in_dim        = None
    dim           = fallback_dim
    pose_head_old = False

    if 'encoder.gat.proj.weight' in ckpt_state:
        w      = ckpt_state['encoder.gat.proj.weight']
        dim    = w.shape[0]
        in_dim = w.shape[1]

    if 'head.mlp.0.weight' in ckpt_state:
        pose_in = ckpt_state['head.mlp.0.weight'].shape[1]
        if pose_in == dim * 2:
            pose_head_old = True

    return in_dim, dim, pose_head_old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',     default='E04')
    parser.add_argument('--subject', default='S35')
    parser.add_argument('--action',  default='A01')
    parser.add_argument('--work',    required=True)
    args = parser.parse_args()

    cfg     = get_config_simple()
    dataset = MMFiDataset(cfg.data.root, [args.env], cfg.data.seq_len)

    ckpt = torch.load(args.work, map_location=device)

    if 'in_dim' in ckpt:
        in_dim = ckpt['in_dim']
        _, dim_ckpt, pose_head_old = detect_model_dims(
            ckpt['model'], fallback_dim=cfg.model.dim)
        print(f"  in_dim = {in_dim}  (from checkpoint metadata)")
    else:
        in_dim, dim_ckpt, pose_head_old = detect_model_dims(
            ckpt['model'], fallback_dim=cfg.model.dim)
        if in_dim is None:
            sample_csi, _, _ = dataset[0]
            in_dim = sample_csi.shape[-1]
            print(f"  in_dim = {in_dim}  (probed from dataset)")
        else:
            print(f"  in_dim = {in_dim}  (inferred from checkpoint weights)")

    print(f"  dim    = {dim_ckpt}")
    print(f"  PoseHead: {'old-style concat(x0,xT)' if pose_head_old else 'new-style x0-only'}")

    model = WiFiPoseModel(
        in_dim=in_dim,
        dim=dim_ckpt,
        num_joints=cfg.model.num_joints,
        pose_head_old=pose_head_old,
    ).to(device)

    result = model.load_state_dict(ckpt['model'], strict=False)
    if not result.missing_keys and not result.unexpected_keys:
        print("  All keys matched perfectly.")
    else:
        if result.missing_keys:
            print(f"  Missing keys ({len(result.missing_keys)}): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}")

    model.eval()
    print(f"Loaded: {args.work}")
    print(f"Testing {args.env}-{args.subject}-{args.action}")

    sample_idx = 0
    for i, s in enumerate(dataset.samples):
        if args.subject in s["csi_dir"] and args.action in s["csi_dir"]:
            sample_idx = i
            break

    csi, pose, root_offset = dataset[sample_idx]
    x  = csi.unsqueeze(0).to(device)
    gt = pose[0].to(device)

    with torch.no_grad():
        pred, _, _, _, _, _, _ = model(x, x, alpha=0.0)

    pred = pred[0]
    print(f"MPJPE:    {mpjpe(pred, gt).item():.4f} m")
    print(f"PA-MPJPE: {pa_mpjpe(pred, gt).item():.4f} m")
    print(f"PCK@50mm: {pck(pred, gt, 0.05).item():.4f}")
    print(f"PCK@20mm: {pck(pred, gt, 0.02).item():.4f}")

    plot_pose(gt.cpu().numpy(), pred.cpu().numpy(), save_path="single_pose.png")
    print("Visualization saved: single_pose.png")


if __name__ == "__main__":
    main()