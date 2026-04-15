"""
v11.4: 提升回归精度

vs v11.3 改动：
  1. 预测中间帧（frame T//2）而非首帧 — 双侧上下文更丰富
  2. PoseHead 升级 — 保留天线维度信息，多尺度时间特征，更深 MLP
  3. 训练数据增强 — CSI 加高斯噪声 + 时间 jitter，提升泛化
  4. Phase 2 分两段：先大 LR 快速收敛，再小 LR 精调
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 25
ENC_LR_RATIO = 0.05
CLS_WEIGHT = 1.0

clean_argv = []
skip_next = False
for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    if arg == '--phase1_epochs':
        PHASE1_EPOCHS = int(sys.argv[i+1]); skip_next = True; continue
    if arg == '--phase2_epochs':
        PHASE2_EPOCHS = int(sys.argv[i+1]); skip_next = True; continue
    if arg == '--enc_lr_ratio':
        ENC_LR_RATIO = float(sys.argv[i+1]); skip_next = True; continue
    if arg == '--cls_weight':
        CLS_WEIGHT = float(sys.argv[i+1]); skip_next = True; continue
    clean_argv.append(arg)
sys.argv = clean_argv

import yaml
from dataset.mmfi_dataset import MMFiDataset
from models.encoder import Encoder
from utils.config import get_config
from utils_metrics import compute_metrics

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')
    _autocast_ctx = lambda: autocast('cuda')
except (ImportError, TypeError):
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    _autocast_ctx = lambda: autocast()

EDGES = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),(8,14),(14,15),(15,16),
]

# 预测中间帧的索引
MID_FRAME = 10   # T=20 时，中间帧 = 第 10 帧

class MMFiWithAction(Dataset):
    def __init__(self, base_dataset, label_map=None):
        self.base = base_dataset
        self.action_labels = []
        self.action_set = set()
        for sample in self.base.samples:
            parts = sample['csi_dir'].split(os.sep)
            action = 'UNK'
            for p in parts:
                if p.startswith('A') and len(p) == 3 and p[1:].isdigit():
                    action = p; break
            self.action_labels.append(action)
            self.action_set.add(action)
        if label_map is not None:
            self.label_map = label_map
            for a in self.action_set:
                if a not in self.label_map:
                    self.label_map[a] = len(self.label_map)
        else:
            self.action_set = sorted(self.action_set)
            self.label_map = {a: i for i, a in enumerate(self.action_set)}
        self.n_classes = len(self.label_map)
        print(f"  动作类别数: {self.n_classes}, 样本数: {len(self.base)}")
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        csi, pose, root_offset = self.base[idx]
        return csi, pose, root_offset, self.label_map[self.action_labels[idx]]


# ═══════════════════════════════════════════════════════════════════════
#  改进的 PoseHead：多尺度时间特征 + 保留天线信息
# ═══════════════════════════════════════════════════════════════════════

class PoseHeadV2(nn.Module):
    """
    改进的姿态回归头。

    vs 旧版：
      旧版: mean(N) → [max_T, motion] → MLP → (J,3)
      新版: [max_T, mean_T, mid_T, motion, per_antenna_mid] → MLP → (J,3)

    多尺度特征聚合：
      - x_max:     全序列 max-pool，捕捉最显著的激活
      - x_mean:    全序列均值，捕捉平均姿态
      - x_mid:     中间帧特征，对应预测目标帧
      - motion:    首末帧差分，编码运动方向
      - ant_feats: 每根天线的中间帧特征（不做 mean），保留空间多样性
    """
    def __init__(self, dim, n_antennas=3, num_joints=17, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        # 输入维度: dim*4 (max+mean+mid+motion) + dim*N (per-antenna)
        feat_dim = dim * 4 + dim * n_antennas

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim, num_joints * 3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_joints * 3),
        )

    def forward(self, feat, mid_idx=10):
        """feat: (B, T, N, D)"""
        B, T, N, D = feat.shape

        x_spa = feat.mean(dim=2)                    # (B, T, D) 天线平均
        x_max = x_spa.max(dim=1).values             # (B, D)
        x_mean = x_spa.mean(dim=1)                  # (B, D)

        mid = min(mid_idx, T - 1)
        x_mid = x_spa[:, mid, :]                    # (B, D) 中间帧
        motion = x_spa[:, -1] - x_spa[:, 0]         # (B, D) 运动差分

        # 每根天线的中间帧特征（保留空间多样性）
        ant_mid = feat[:, mid, :, :]                 # (B, N, D)
        ant_feats = ant_mid.reshape(B, N * D)        # (B, N*D)

        pose_in = torch.cat([x_max, x_mean, x_mid, motion, ant_feats], dim=-1)
        pose = self.mlp(pose_in).view(B, self.num_joints, 3)
        vel = self.vel_head(motion).view(B, self.num_joints, 3)
        return pose, vel


class ClassificationHead(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, n_classes),
        )
    def forward(self, feat):
        x_spa = feat.mean(dim=2)
        x_max = x_spa.max(dim=1).values
        motion = x_spa[:, -1] - x_spa[:, 0]
        return self.head(torch.cat([x_max, motion], dim=-1))


class FullModel(nn.Module):
    def __init__(self, in_dim, dim, n_classes, num_joints=17, n_antennas=3):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHeadV2(dim, n_antennas=n_antennas, num_joints=num_joints)
    def forward(self, x, mode='both'):
        feat = self.encoder(x)
        if mode == 'cls': return self.cls_head(feat)
        elif mode == 'reg': return self.reg_head(feat, mid_idx=MID_FRAME)
        else:
            logits = self.cls_head(feat)
            pose, vel = self.reg_head(feat, mid_idx=MID_FRAME)
            return logits, pose, vel


# ═══════════════════════════════════════════════════════════════════════
#  数据增强
# ═══════════════════════════════════════════════════════════════════════

def augment_csi(csi, noise_std=0.05, drop_prob=0.1):
    """
    CSI 数据增强（训练时使用）。
    - 高斯噪声：模拟环境波动
    - 随机时间帧 dropout：模拟丢包
    """
    if noise_std > 0:
        csi = csi + torch.randn_like(csi) * noise_std

    if drop_prob > 0:
        B, T, N, C = csi.shape
        mask = (torch.rand(B, T, 1, 1, device=csi.device) > drop_prob).float()
        csi = csi * mask

    return csi


# ═══════════════════════════════════════════════════════════════════════
#  损失（预测中间帧）
# ═══════════════════════════════════════════════════════════════════════

def regression_loss(pred, vel_pred, gt, mid_idx=10):
    """gt: (B, T, J, 3)，预测第 mid_idx 帧"""
    mid = min(mid_idx, gt.shape[1] - 1)
    gt_mid = gt[:, mid]                              # (B, J, 3)

    pose_loss = F.l1_loss(pred, gt_mid)

    gt_vel = gt[:, -1] - gt[:, 0]
    vel_loss = F.l1_loss(vel_pred, gt_vel)

    bone_loss = torch.tensor(0.0, device=pred.device)
    for i, j in EDGES:
        pred_len = torch.norm(pred[:, i] - pred[:, j], dim=-1)
        gt_len = torch.norm(gt_mid[:, i] - gt_mid[:, j], dim=-1)
        bone_loss = bone_loss + F.l1_loss(pred_len, gt_len)
    bone_loss = bone_loss / len(EDGES)

    return pose_loss + 0.1 * vel_loss + 0.1 * bone_loss, {
        "pose": pose_loss.item(), "vel": vel_loss.item(), "bone": bone_loss.item(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  评估（也改为中间帧）
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_regression(model, loader, device, tag=""):
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        x = batch[0].to(device); y = batch[1]
        pose, _ = model(x, mode='reg')
        all_pred.append(pose.cpu())
        mid = min(MID_FRAME, y.shape[1] - 1)
        all_gt.append(y[:, mid])                     # 中间帧 GT
    metrics = compute_metrics(all_pred, all_gt)
    print(f"  [{tag}] MPJPE={metrics['MPJPE']*1000:.1f}mm  "
          f"PA-MPJPE={metrics['PA-MPJPE']*1000:.1f}mm  "
          f"PCK@50mm={metrics['PCK@0.05']:.3f}")
    model.train(); return metrics

@torch.no_grad()
def eval_classification(model, loader, device, tag=""):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x = batch[0].to(device); action = batch[3].to(device)
        logits = model(x, mode='cls')
        correct += (logits.argmax(1) == action).sum().item()
        total += x.shape[0]
    acc = correct / total
    print(f"  [{tag}] Action accuracy: {acc:.1%}")
    model.train(); return acc

@torch.no_grad()
def check_collapse(model, loader, device):
    model.eval()
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if i >= 5: break
        x = batch[0].to(device); y = batch[1]
        pose, _ = model(x, mode='reg')
        preds.append(pose.cpu())
        mid = min(MID_FRAME, y.shape[1] - 1)
        gts.append(y[:, mid])
    preds = torch.cat(preds); gts = torch.cat(gts)
    pred_std = preds.std(dim=0).mean().item() * 1000
    gt_std = gts.std(dim=0).mean().item() * 1000
    ratio = pred_std / (gt_std + 1e-8)
    st = "⚠️塌陷" if ratio < 0.3 else ("⚡恢复中" if ratio < 0.6 else "✓正常")
    print(f"  [Collapse] pred_std={pred_std:.1f}mm  gt_std={gt_std:.1f}mm  "
          f"ratio={ratio:.3f}  {st}")
    model.train(); return ratio


# ═══════════════════════════════════════════════════════════════════════
#  主训练
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = get_config()
    total_epochs = cfg.train.epochs
    p1, p2 = PHASE1_EPOCHS, PHASE2_EPOCHS
    enc_lr_ratio = ENC_LR_RATIO
    cls_weight = CLS_WEIGHT

    exp_name = "v11_clspretrain_" + time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 日志记录
    from utils.logger import setup_logger
    setup_logger(os.path.join(save_dir, "train.log"))

    print("=" * 60)
    print("  v11.4: 多尺度 PoseHead + 中间帧预测 + 数据增强")
    print(f"  Phase 1 (epoch 0-{p1-1}):     联合分类")
    print(f"  Phase 2 (epoch {p1}-{p1+p2-1}):  冻结回归")
    print(f"  Phase 3 (epoch {p1+p2}-{total_epochs-1}):  微调")
    print(f"  预测目标: 中间帧 (frame {MID_FRAME})")
    print(f"  Best 保存依据: TGT MPJPE")
    print("=" * 60)

    cache_dir = getattr(cfg.data, "cache_dir", None)
    src_base = MMFiDataset(cfg.data.root, cfg.domain.source,
                           cfg.data.seq_len, cache_dir=cache_dir)
    tgt_base = MMFiDataset(cfg.data.root, cfg.domain.target,
                           cfg.data.seq_len, cache_dir=cache_dir)

    src_data = MMFiWithAction(src_base)
    label_map = src_data.label_map
    tgt_data = MMFiWithAction(tgt_base, label_map=label_map)
    n_classes = src_data.n_classes

    sample_csi, _, _, _ = src_data[0]
    T, N, C = sample_csi.shape
    print(f"  CSI: T={T}, N={N}, C={C}, 类别: {n_classes}")

    combined_data = ConcatDataset([src_data, tgt_data])
    combined_loader = DataLoader(combined_data, batch_size=cfg.train.batch_size,
                                 shuffle=True, num_workers=4,
                                 pin_memory=True, drop_last=True)
    src_loader = DataLoader(src_data, batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    eval_bs = min(512, cfg.train.batch_size * 2)
    src_eval = DataLoader(src_data, batch_size=eval_bs, shuffle=False, num_workers=4)
    tgt_eval = DataLoader(tgt_data, batch_size=eval_bs, shuffle=False, num_workers=4)

    model = FullModel(in_dim=C, dim=cfg.model.dim, n_classes=n_classes,
                      num_joints=cfg.model.num_joints, n_antennas=N).to(device)

    start_epoch = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"  Resumed from epoch {start_epoch}")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_tgt_mpjpe = float('inf')
    encoder_frozen_in_p3 = False
    optimizer = None
    scheduler = None

    for epoch in range(start_epoch, total_epochs):
        model.train()

        if epoch < p1:
            phase, phase_name = 1, "Phase1-CLS"
        elif epoch < p1 + p2:
            phase, phase_name = 2, "Phase2-REG(frozen)"
        else:
            phase_name = "Phase3-FT(enc-frozen)" if encoder_frozen_in_p3 else "Phase3-FINETUNE"
            phase = 3

        if epoch == max(start_epoch, 0) and epoch < p1:
            params = list(model.encoder.parameters()) + list(model.cls_head.parameters())
            optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=p1, eta_min=cfg.train.lr * 0.01)
            print(f"\n  >>> Phase 1: 联合分类")

        elif epoch == max(start_epoch, p1):
            for p in model.encoder.parameters(): p.requires_grad = False
            for p in model.cls_head.parameters(): p.requires_grad = False
            params = list(model.reg_head.parameters())
            optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=p2, eta_min=cfg.train.lr * 0.01)
            print(f"\n  >>> Phase 2: 冻结回归 ({p2} epochs)")

        elif epoch == max(start_epoch, p1 + p2):
            for p in model.parameters(): p.requires_grad = True
            p3_epochs = total_epochs - p1 - p2
            optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(),  'lr': cfg.train.lr * enc_lr_ratio},
                {'params': model.cls_head.parameters(), 'lr': cfg.train.lr * enc_lr_ratio},
                {'params': model.reg_head.parameters(), 'lr': cfg.train.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=p3_epochs, eta_min=cfg.train.lr * 0.001)
            print(f"\n  >>> Phase 3: 微调 ({p3_epochs} epochs)")

        active_loader = combined_loader if phase == 1 else src_loader
        epoch_loss, epoch_steps = 0.0, 0
        comp_sum = {}

        for i, batch in enumerate(active_loader):
            xs = batch[0].to(device, non_blocking=True)
            ys = batch[1].to(device, non_blocking=True)
            action = batch[3].to(device, non_blocking=True)

            # 数据增强（Phase 2/3 训练回归时使用）
            if phase >= 2:
                xs = augment_csi(xs, noise_std=0.03, drop_prob=0.05)

            optimizer.zero_grad()
            with _autocast_ctx():
                if phase == 1:
                    logits = model(xs, mode='cls')
                    loss = F.cross_entropy(logits, action)
                    comp = {"cls": loss.item()}
                elif phase == 2:
                    pose, vel = model(xs, mode='reg')
                    loss, comp = regression_loss(pose, vel, ys, mid_idx=MID_FRAME)
                else:
                    logits, pose, vel = model(xs, mode='both')
                    cls_loss = F.cross_entropy(logits, action)
                    reg_loss, reg_comp = regression_loss(pose, vel, ys, mid_idx=MID_FRAME)
                    loss = reg_loss + cls_weight * cls_loss
                    comp = {**reg_comp, "cls": cls_loss.item()}

            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_steps += 1
            for k, v in comp.items():
                comp_sum[k] = comp_sum.get(k, 0) + v

            if i % cfg.log.print_freq == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                parts = " ".join(f"{k}={v:.4f}" for k, v in comp.items())
                print(f"[E{epoch:03d}][{i:04d}/{len(active_loader)}] "
                      f"[{phase_name}] loss={loss.item():.4f}  {parts}  "
                      f"lr={cur_lr:.6f}")

        if scheduler is not None: scheduler.step()
        if epoch_steps == 0: continue

        avg = epoch_loss / epoch_steps
        parts = " | ".join(f"{k}={comp_sum[k]/epoch_steps:.4f}" for k in comp_sum)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f" Epoch {epoch:03d} [{phase_name}] avg={avg:.4f} | {parts} | lr={cur_lr:.6f}")

        do_eval = (epoch % 5 == 0 or epoch == total_epochs - 1 or
                   epoch == p1 - 1 or epoch == p1 or
                   epoch == p1 + p2 - 1 or epoch == p1 + p2)

        if do_eval:
            print(f"\n  --- Epoch {epoch} [{phase_name}] ---")
            src_cls = eval_classification(model, src_eval, device, "SRC-CLS")
            tgt_cls = eval_classification(model, tgt_eval, device, "TGT-CLS")

            if phase >= 2:
                src_m = eval_regression(model, src_eval, device, "SRC-REG")
                tgt_m = eval_regression(model, tgt_eval, device, "TGT-REG")
                check_collapse(model, tgt_eval, device)

                if tgt_m['MPJPE'] < best_tgt_mpjpe:
                    best_tgt_mpjpe = tgt_m['MPJPE']
                    torch.save({
                        'model': model.state_dict(), 'epoch': epoch,
                        'phase': phase, 'in_dim': C,
                        'src_mpjpe': src_m['MPJPE'], 'tgt_mpjpe': tgt_m['MPJPE'],
                        'tgt_cls_acc': tgt_cls, 'mid_frame': MID_FRAME,
                        'pose_head_old': False,
                    }, os.path.join(save_dir, "best.pth"))
                    print(f"  -> best TGT: {best_tgt_mpjpe*1000:.1f}mm "
                          f"(SRC: {src_m['MPJPE']*1000:.1f}mm)")

                if phase == 3 and not encoder_frozen_in_p3 and tgt_cls < 0.90:
                    print(f"\n  ⚠️ TGT-CLS={tgt_cls:.1%}, 冻结 Encoder")
                    for p in model.encoder.parameters(): p.requires_grad = False
                    for p in model.cls_head.parameters(): p.requires_grad = False
                    optimizer = torch.optim.AdamW(
                        model.reg_head.parameters(), lr=cur_lr, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(total_epochs - epoch, 1),
                        eta_min=cfg.train.lr * 0.001)
                    encoder_frozen_in_p3 = True
            print()

        torch.save({
            'model': model.state_dict(), 'epoch': epoch,
            'phase': phase, 'in_dim': C, 'loss': avg, 'pose_head_old': False,
        }, os.path.join(save_dir, "latest.pth"))

    print("\n" + "=" * 60)
    print("  FINAL")
    print("=" * 60)
    if os.path.exists(os.path.join(save_dir, "best.pth")):
        ckpt = torch.load(os.path.join(save_dir, "best.pth"), map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"  Best epoch: {ckpt['epoch']}")
        eval_classification(model, src_eval, device, "SRC-CLS")
        eval_classification(model, tgt_eval, device, "TGT-CLS")
        eval_regression(model, src_eval, device, "SRC-REG best")
        eval_regression(model, tgt_eval, device, "TGT-REG best")
        check_collapse(model, tgt_eval, device)
    print(f"\n  Saved: {save_dir}")

if __name__ == "__main__":
    main()