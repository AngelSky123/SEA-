"""v11.4 测试脚本（支持中间帧预测）"""
import torch, torch.nn as nn, torch.nn.functional as F
import argparse, yaml
from torch.utils.data import DataLoader
from dataset.mmfi_dataset import MMFiDataset
from models.encoder import Encoder
from utils_metrics import compute_metrics

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
        x_max = x_spa.max(dim=1).values
        motion = x_spa[:, -1] - x_spa[:, 0]
        return self.head(torch.cat([x_max, motion], dim=-1))

class FullModel(nn.Module):
    def __init__(self, in_dim, dim, n_classes=27, num_joints=17, n_antennas=3):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHeadV2(dim, n_antennas=n_antennas, num_joints=num_joints)
        self.mid_frame = 10
    def forward(self, x, mode='reg'):
        feat = self.encoder(x)
        if mode == 'cls': return self.cls_head(feat)
        elif mode == 'reg': return self.reg_head(feat, self.mid_frame)
        else:
            logits = self.cls_head(feat)
            pose, vel = self.reg_head(feat, self.mid_frame)
            return logits, pose, vel

# ── 兼容旧版 PoseHead（v11.1-v11.3）──────────────────────────────────
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
        x_max = x_spa.max(dim=1).values
        motion = x_spa[:, -1] - x_spa[:, 0]
        pose_in = torch.cat([x_max, motion], dim=-1)
        return self.mlp(pose_in).view(B, self.num_joints, 3), \
               self.vel_head(motion).view(B, self.num_joints, 3)

class FullModelOld(nn.Module):
    def __init__(self, in_dim, dim, n_classes=27, num_joints=17):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHeadOld(dim, num_joints)
    def forward(self, x, mode='reg'):
        feat = self.encoder(x)
        if mode == 'cls': return self.cls_head(feat)
        elif mode == 'reg': return self.reg_head(feat)
        else:
            logits = self.cls_head(feat)
            pose, vel = self.reg_head(feat)
            return logits, pose, vel

def detect_version(state):
    """检测 checkpoint 是 v11.4 (PoseHeadV2) 还是旧版"""
    for k in state:
        if 'reg_head.mlp.8' in k:   # 4 层 MLP = v11.4
            return 'v2'
    return 'old'

def main():
    from utils.logger import setup_logger
    setup_logger("test_v11_log.log")
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', required=True)
    parser.add_argument('--target', nargs='+', default=None)
    args = parser.parse_args()

    with open('config/default.yaml') as f:
        cfg = yaml.safe_load(f)
    target_envs = args.target or cfg['domain']['target']

    test_data = MMFiDataset(cfg['data']['root'], target_envs, cfg['data']['seq_len'])
    loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    ckpt = torch.load(args.work, map_location=device, weights_only=False)
    state = ckpt['model']

    in_dim = ckpt.get('in_dim', None)
    dim = 256
    if 'encoder.gat.proj.weight' in state:
        dim = state['encoder.gat.proj.weight'].shape[0]
        if in_dim is None: in_dim = state['encoder.gat.proj.weight'].shape[1]
    if in_dim is None:
        s, _, _ = test_data[0]; in_dim = s.shape[-1]

    mid_frame = ckpt.get('mid_frame', 0)
    version = detect_version(state)
    print(f"  in_dim={in_dim} dim={dim} version={version} mid_frame={mid_frame}")

    sample_csi, _, _ = test_data[0]
    N = sample_csi.shape[1]

    if version == 'v2':
        model = FullModel(in_dim=in_dim, dim=dim, n_antennas=N).to(device)
        model.mid_frame = mid_frame
    else:
        model = FullModelOld(in_dim=in_dim, dim=dim).to(device)

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys: print(f"  Missing: {len(result.missing_keys)}")
    if result.unexpected_keys: print(f"  Unexpected: {len(result.unexpected_keys)}")
    if not result.missing_keys and not result.unexpected_keys: print("  All keys matched.")

    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            pose, _ = model(x, mode='reg')
            all_pred.append(pose.cpu())
            gt_idx = min(mid_frame, y.shape[1]-1)
            all_gt.append(y[:, gt_idx])

    metrics = compute_metrics(all_pred, all_gt)
    print(f"\n===== RESULTS (target={target_envs}) =====")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}  ({v*1000:.1f}mm)" if 'MPJPE' in k else f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()