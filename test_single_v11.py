"""v11.4 单样本测试"""
import torch, torch.nn as nn, torch.nn.functional as F
import argparse, yaml
from dataset.mmfi_dataset import MMFiDataset
from models.encoder import Encoder
from visualization.pose_vis import plot_pose
from utils_metrics import mpjpe, pa_mpjpe, pck

device = "cuda" if torch.cuda.is_available() else "cpu"

class PoseHeadV2(nn.Module):
    def __init__(self, dim, n_antennas=3, num_joints=17, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        feat_dim = dim*4 + dim*n_antennas
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, dim*2), nn.LayerNorm(dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim*2), nn.LayerNorm(dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_joints*3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, num_joints*3),
        )
    def forward(self, feat, mid_idx=10):
        B, T, N, D = feat.shape
        x_spa = feat.mean(dim=2)
        x_max = x_spa.max(dim=1).values
        x_mean = x_spa.mean(dim=1)
        mid = min(mid_idx, T-1)
        x_mid = x_spa[:, mid]
        motion = x_spa[:, -1] - x_spa[:, 0]
        ant_feats = feat[:, mid].reshape(B, N*D)
        pose_in = torch.cat([x_max, x_mean, x_mid, motion, ant_feats], dim=-1)
        return self.mlp(pose_in).view(B, self.num_joints, 3), \
               self.vel_head(motion).view(B, self.num_joints, 3)

class ClassificationHead(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim*2, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, n_classes),
        )
    def forward(self, feat):
        x_spa = feat.mean(dim=2)
        return self.head(torch.cat([x_spa.max(dim=1).values, x_spa[:,-1]-x_spa[:,0]], dim=-1))

class FullModel(nn.Module):
    def __init__(self, in_dim, dim, n_classes=27, num_joints=17, n_antennas=3):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHeadV2(dim, n_antennas=n_antennas, num_joints=num_joints)
        self.mid_frame = 10
    def forward(self, x, mode='reg'):
        feat = self.encoder(x)
        if mode == 'reg': return self.reg_head(feat, self.mid_frame)
        else: return self.cls_head(feat)

# 兼容旧版
class PoseHeadOld(nn.Module):
    def __init__(self, dim, num_joints=17, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, dim*2), nn.LayerNorm(dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_joints*3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, num_joints*3),
        )
    def forward(self, feat, mid_idx=0):
        B = feat.shape[0]
        x_spa = feat.mean(dim=2)
        pose_in = torch.cat([x_spa.max(dim=1).values, x_spa[:,-1]-x_spa[:,0]], dim=-1)
        return self.mlp(pose_in).view(B, self.num_joints, 3), \
               self.vel_head(x_spa[:,-1]-x_spa[:,0]).view(B, self.num_joints, 3)

class FullModelOld(nn.Module):
    def __init__(self, in_dim, dim, n_classes=27, num_joints=17):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHeadOld(dim, num_joints)
        self.mid_frame = 0
    def forward(self, x, mode='reg'):
        feat = self.encoder(x)
        if mode == 'reg': return self.reg_head(feat)
        else: return self.cls_head(feat)

def detect_version(state):
    for k in state:
        if 'reg_head.mlp.8' in k: return 'v2'
    return 'old'

def main():
    from utils.logger import setup_logger
    setup_logger("test_single_v11_log.log")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='E04')
    parser.add_argument('--subject', default='S34')
    parser.add_argument('--action', default='A13')
    parser.add_argument('--work', required=True)
    parser.add_argument('--frame_offset', type=int, default=0)
    args = parser.parse_args()

    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)

    dataset = MMFiDataset(cfg['data']['root'], [args.env], cfg['data']['seq_len'])
    ckpt = torch.load(args.work, map_location=device, weights_only=False)
    state = ckpt['model']

    in_dim = ckpt.get('in_dim', None)
    dim = 256
    if 'encoder.gat.proj.weight' in state:
        dim = state['encoder.gat.proj.weight'].shape[0]
        if in_dim is None: in_dim = state['encoder.gat.proj.weight'].shape[1]
    if in_dim is None:
        s, _, _ = dataset[0]; in_dim = s.shape[-1]

    mid_frame = ckpt.get('mid_frame', 0)
    version = detect_version(state)
    sample_csi, _, _ = dataset[0]
    N = sample_csi.shape[1]

    print(f"  in_dim={in_dim} dim={dim} version={version} mid_frame={mid_frame}")

    if version == 'v2':
        model = FullModel(in_dim=in_dim, dim=dim, n_antennas=N).to(device)
        model.mid_frame = mid_frame
    else:
        model = FullModelOld(in_dim=in_dim, dim=dim).to(device)

    result = model.load_state_dict(state, strict=False)
    if not result.missing_keys and not result.unexpected_keys:
        print("  All keys matched.")
    else:
        if result.missing_keys: print(f"  Missing: {len(result.missing_keys)}")
        if result.unexpected_keys: print(f"  Unexpected: {len(result.unexpected_keys)}")

    model.eval()
    print(f"  Testing {args.env}-{args.subject}-{args.action} (offset={args.frame_offset})")

    matches = []
    for i, s in enumerate(dataset.samples):
        if args.subject in s["csi_dir"] and args.action in s["csi_dir"]:
            matches.append(i)
    if not matches:
        print(f"  [Error] 未找到匹配样本"); return

    offset = min(args.frame_offset, len(matches)-1)
    sample_idx = matches[offset]
    print(f"  匹配: {len(matches)} 样本, 使用第 {offset} 个")

    csi, pose, _ = dataset[sample_idx]
    x = csi.unsqueeze(0).to(device)
    gt_idx = min(mid_frame, pose.shape[0]-1)
    gt = pose[gt_idx].to(device)

    with torch.no_grad():
        pred_pose, _ = model(x, mode='reg')
        pred = pred_pose[0]

    m = mpjpe(pred, gt).item()
    pa = pa_mpjpe(pred, gt).item()
    print(f"\n  MPJPE:    {m*1000:.1f} mm")
    print(f"  PA-MPJPE: {pa*1000:.1f} mm")
    print(f"  PCK@50mm: {pck(pred, gt, 0.05).item():.4f}")
    print(f"  PCK@20mm: {pck(pred, gt, 0.02).item():.4f}")

    save_path = f"single_pose_{args.env}_{args.subject}_{args.action}_f{offset}.png"
    plot_pose(gt.cpu().numpy(), pred.cpu().numpy(), save_path=save_path)
    print(f"  Saved: {save_path}")

if __name__ == "__main__":
    main()