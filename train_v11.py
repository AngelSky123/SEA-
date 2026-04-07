"""
v11: 分类预训练 + 回归微调

核心发现：
  - 动作分类用简单 MLP 就能达到 96%（CSI 信号丰富）
  - 但回归训练全部塌陷到均值（MSE 的均值捷径）
  - 分类没有均值捷径（不存在"平均类别"）

解决方案：
  Phase 1: 用动作分类（27 类）预训练 Encoder
           → 迫使 Encoder 学会区分不同 CSI 输入
  Phase 2: 冻结 Encoder，只训练 PoseHead 做回归
           → PoseHead 必须利用 Encoder 的多样特征
  Phase 3: 解冻全部，小学习率端到端微调
           → 让 Encoder 从"区分动作"微调到"精确定位关节"

用法：
    python train_v11.py --epochs 50
    python train_v11.py --epochs 80 --phase1_epochs 20 --phase2_epochs 20
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# ── 自定义参数处理 ────────────────────────────────────────────────────
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15

clean_argv = []
skip_next = False
for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    if arg == '--phase1_epochs':
        PHASE1_EPOCHS = int(sys.argv[i+1])
        skip_next = True
        continue
    if arg == '--phase2_epochs':
        PHASE2_EPOCHS = int(sys.argv[i+1])
        skip_next = True
        continue
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


# ═══════════════════════════════════════════════════════════════════════
#  数据集：带动作标签
# ═══════════════════════════════════════════════════════════════════════

class MMFiWithAction(Dataset):
    """在 MMFiDataset 基础上提取动作标签。"""
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.action_labels = []
        self.action_set = set()

        for sample in self.base.samples:
            parts = sample['csi_dir'].split(os.sep)
            action = 'UNK'
            for p in parts:
                if p.startswith('A') and len(p) == 3 and p[1:].isdigit():
                    action = p
                    break
            self.action_labels.append(action)
            self.action_set.add(action)

        self.action_set = sorted(self.action_set)
        self.label_map = {a: i for i, a in enumerate(self.action_set)}
        self.n_classes = len(self.action_set)
        print(f"  动作类别数: {self.n_classes}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        csi, pose, root_offset = self.base[idx]
        action_id = self.label_map[self.action_labels[idx]]
        return csi, pose, root_offset, action_id


# ═══════════════════════════════════════════════════════════════════════
#  模型
# ═══════════════════════════════════════════════════════════════════════

class PoseHead(nn.Module):
    """回归头：用 Encoder 特征预测姿态。"""
    def __init__(self, dim, num_joints=17, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.mlp = nn.Sequential(
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

    def forward(self, feat):
        """feat: (B, T, N, D)"""
        B = feat.shape[0]
        x_spa = feat.mean(dim=2)                          # (B, T, D)
        x_max = x_spa.max(dim=1).values                   # (B, D)
        motion = x_spa[:, -1] - x_spa[:, 0]               # (B, D)
        pose_in = torch.cat([x_max, motion], dim=-1)       # (B, 2D)
        pose = self.mlp(pose_in).view(B, self.num_joints, 3)
        vel  = self.vel_head(motion).view(B, self.num_joints, 3)
        return pose, vel


class ClassificationHead(nn.Module):
    """分类头：用 Encoder 特征预测动作类别。"""
    def __init__(self, dim, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, n_classes),
        )

    def forward(self, feat):
        """feat: (B, T, N, D)"""
        x_spa = feat.mean(dim=2)                   # (B, T, D)
        x_max = x_spa.max(dim=1).values             # (B, D)
        motion = x_spa[:, -1] - x_spa[:, 0]         # (B, D)
        x = torch.cat([x_max, motion], dim=-1)       # (B, 2D)
        return self.head(x)


class FullModel(nn.Module):
    """
    完整模型：Encoder + 分类头 + 回归头。
    分类头和回归头看到相同的 Encoder 特征（相同的 pooling）。
    """
    def __init__(self, in_dim, dim, n_classes, num_joints=17):
        super().__init__()
        self.encoder  = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHead(dim, num_joints)

    def forward(self, x, mode='both'):
        """
        mode:
          'cls'  — 只返回分类 logits
          'reg'  — 只返回 (pose, vel)
          'both' — 返回 (logits, pose, vel)
        """
        feat = self.encoder(x)

        if mode == 'cls':
            return self.cls_head(feat)
        elif mode == 'reg':
            return self.reg_head(feat)
        else:
            logits = self.cls_head(feat)
            pose, vel = self.reg_head(feat)
            return logits, pose, vel


# ═══════════════════════════════════════════════════════════════════════
#  损失
# ═══════════════════════════════════════════════════════════════════════

def regression_loss(pred, vel_pred, gt):
    pose_loss = F.l1_loss(pred, gt[:, 0])

    gt_vel = gt[:, -1] - gt[:, 0]
    vel_loss = F.l1_loss(vel_pred, gt_vel)

    bone_loss = torch.tensor(0.0, device=pred.device)
    for i, j in EDGES:
        pred_len = torch.norm(pred[:, i] - pred[:, j],    dim=-1)
        gt_len   = torch.norm(gt[:, 0, i] - gt[:, 0, j], dim=-1)
        bone_loss = bone_loss + F.l1_loss(pred_len, gt_len)
    bone_loss = bone_loss / len(EDGES)

    return pose_loss + 0.1 * vel_loss + 0.1 * bone_loss, {
        "pose": pose_loss.item(),
        "vel":  vel_loss.item(),
        "bone": bone_loss.item(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  评估 & 诊断
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_regression(model, loader, device, tag=""):
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1]
        pose, _ = model(x, mode='reg')
        all_pred.append(pose.cpu())
        all_gt.append(y[:, 0])
    metrics = compute_metrics(all_pred, all_gt)
    print(f"  [{tag}] MPJPE={metrics['MPJPE']*1000:.1f}mm  "
          f"PA-MPJPE={metrics['PA-MPJPE']*1000:.1f}mm  "
          f"PCK@50mm={metrics['PCK@0.05']:.3f}")
    model.train()
    return metrics


@torch.no_grad()
def eval_classification(model, loader, device, tag=""):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x = batch[0].to(device)
        action = batch[3].to(device)
        logits = model(x, mode='cls')
        correct += (logits.argmax(1) == action).sum().item()
        total += x.shape[0]
    acc = correct / total
    print(f"  [{tag}] Action accuracy: {acc:.1%}")
    model.train()
    return acc


@torch.no_grad()
def check_collapse(model, loader, device):
    model.eval()
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        x = batch[0].to(device)
        y = batch[1]
        pose, _ = model(x, mode='reg')
        preds.append(pose.cpu())
        gts.append(y[:, 0])

    preds = torch.cat(preds)
    gts = torch.cat(gts)
    pred_std = preds.std(dim=0).mean().item() * 1000
    gt_std = gts.std(dim=0).mean().item() * 1000
    ratio = pred_std / (gt_std + 1e-8)

    st = "⚠️塌陷" if ratio < 0.3 else ("⚡恢复中" if ratio < 0.6 else "✓正常")
    print(f"  [Collapse] pred_std={pred_std:.1f}mm  gt_std={gt_std:.1f}mm  "
          f"ratio={ratio:.3f}  {st}")
    model.train()
    return ratio


# ═══════════════════════════════════════════════════════════════════════
#  主训练
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = get_config()
    total_epochs = cfg.train.epochs

    exp_name = "v11_clspretrain_" + time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    p1 = PHASE1_EPOCHS
    p2 = PHASE2_EPOCHS
    p3 = max(0, total_epochs - p1 - p2)

    print("=" * 60)
    print("  v11: 分类预训练 → 冻结回归 → 端到端微调")
    print(f"  Phase 1 (epoch 0-{p1-1}):  分类预训练 Encoder")
    print(f"  Phase 2 (epoch {p1}-{p1+p2-1}): 冻结 Encoder，训练 PoseHead")
    print(f"  Phase 3 (epoch {p1+p2}-{total_epochs-1}): 端到端微调")
    print("=" * 60)

    cache_dir = getattr(cfg.data, "cache_dir", None)
    src_base = MMFiDataset(cfg.data.root, cfg.domain.source,
                           cfg.data.seq_len, cache_dir=cache_dir)
    tgt_base = MMFiDataset(cfg.data.root, cfg.domain.target,
                           cfg.data.seq_len, cache_dir=cache_dir)

    src_data = MMFiWithAction(src_base)
    tgt_data = MMFiWithAction(tgt_base)
    n_classes = src_data.n_classes

    sample_csi, _, _, _ = src_data[0]
    T, N, C = sample_csi.shape
    print(f"  CSI: T={T}, N={N}, C={C}")
    print(f"  动作类别: {n_classes}")

    src_loader = DataLoader(src_data, batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    src_eval = DataLoader(src_data, batch_size=64, shuffle=False, num_workers=4)
    tgt_eval = DataLoader(tgt_data, batch_size=64, shuffle=False, num_workers=4)

    model = FullModel(
        in_dim=C,
        dim=cfg.model.dim,
        n_classes=n_classes,
        num_joints=cfg.model.num_joints,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    best_mpjpe = float('inf')

    for epoch in range(total_epochs):
        model.train()

        # ── 确定当前阶段 ─────────────────────────────────────────
        if epoch < p1:
            phase = 1
            phase_name = "Phase1-CLS"
        elif epoch < p1 + p2:
            phase = 2
            phase_name = "Phase2-REG(frozen)"
        else:
            phase = 3
            phase_name = "Phase3-FINETUNE"

        # ── 设置优化器（每个阶段重新创建）─────────────────────────
        if epoch == 0:
            # Phase 1: 只训练 encoder + cls_head
            params = list(model.encoder.parameters()) + \
                     list(model.cls_head.parameters())
            optimizer = torch.optim.AdamW(params, lr=cfg.train.lr,
                                          weight_decay=1e-4)
            print(f"\n  >>> Phase 1: 训练 Encoder + 分类头")

        elif epoch == p1:
            # Phase 2: 冻结 encoder，只训练 reg_head
            for p in model.encoder.parameters():
                p.requires_grad = False
            for p in model.cls_head.parameters():
                p.requires_grad = False

            params = list(model.reg_head.parameters())
            optimizer = torch.optim.AdamW(params, lr=cfg.train.lr,
                                          weight_decay=1e-4)
            n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            n_train  = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"\n  >>> Phase 2: 冻结 Encoder ({n_frozen} params frozen), "
                  f"训练 PoseHead ({n_train} params)")

        elif epoch == p1 + p2:
            # Phase 3: 解冻全部，小学习率
            for p in model.parameters():
                p.requires_grad = True

            # Encoder 用小 LR，PoseHead 用正常 LR
            optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': cfg.train.lr * 0.1},
                {'params': model.cls_head.parameters(), 'lr': cfg.train.lr * 0.1},
                {'params': model.reg_head.parameters(), 'lr': cfg.train.lr},
            ], weight_decay=1e-4)
            print(f"\n  >>> Phase 3: 端到端微调 (Encoder LR=×0.1)")

        # ── 训练一个 epoch ────────────────────────────────────────
        epoch_loss = 0.0
        epoch_steps = 0
        comp_sum = {}

        for i, batch in enumerate(src_loader):
            xs = batch[0].to(device, non_blocking=True)
            ys = batch[1].to(device, non_blocking=True)
            action = batch[3].to(device, non_blocking=True)

            optimizer.zero_grad()

            with _autocast_ctx():
                if phase == 1:
                    # 只做分类
                    logits = model(xs, mode='cls')
                    loss = F.cross_entropy(logits, action)
                    comp = {"cls": loss.item()}

                elif phase == 2:
                    # 只做回归（encoder 冻结）
                    pose, vel = model(xs, mode='reg')
                    loss, comp = regression_loss(pose, vel, ys)

                else:
                    # Phase 3: 分类 + 回归联合
                    logits, pose, vel = model(xs, mode='both')
                    cls_loss = F.cross_entropy(logits, action)
                    reg_loss, reg_comp = regression_loss(pose, vel, ys)
                    # 分类作为辅助损失，保持 Encoder 特征不退化
                    loss = reg_loss + 0.5 * cls_loss
                    comp = {**reg_comp, "cls": cls_loss.item()}

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_steps += 1
            for k, v in comp.items():
                comp_sum[k] = comp_sum.get(k, 0) + v

            if i % cfg.log.print_freq == 0:
                parts = " ".join(f"{k}={v:.4f}" for k, v in comp.items())
                print(f"[E{epoch:03d}][{i:04d}/{len(src_loader)}] "
                      f"[{phase_name}] loss={loss.item():.4f}  {parts}")

        if epoch_steps == 0:
            continue

        avg = epoch_loss / epoch_steps
        parts = " | ".join(f"{k}={comp_sum[k]/epoch_steps:.4f}"
                           for k in comp_sum)
        print(f" Epoch {epoch:03d} [{phase_name}] avg={avg:.4f} | {parts}")

        # ── 评估 ─────────────────────────────────────────────────
        if epoch % 3 == 0 or epoch == total_epochs - 1 or \
           epoch == p1 - 1 or epoch == p1 or epoch == p1 + p2 - 1 or epoch == p1 + p2:
            print(f"\n  --- Epoch {epoch} [{phase_name}] ---")

            if phase >= 1:
                eval_classification(model, src_eval, device, "SRC-CLS")
                if phase == 1:
                    eval_classification(model, tgt_eval, device, "TGT-CLS")

            if phase >= 2:
                src_m = eval_regression(model, src_eval, device, "SRC-REG")
                tgt_m = eval_regression(model, tgt_eval, device, "TGT-REG")
                check_collapse(model, tgt_eval, device)

                if src_m['MPJPE'] < best_mpjpe:
                    best_mpjpe = src_m['MPJPE']
                    torch.save({
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'phase': phase,
                        'in_dim': C,
                        'src_mpjpe': src_m['MPJPE'],
                        'tgt_mpjpe': tgt_m['MPJPE'],
                        'pose_head_old': False,
                    }, os.path.join(save_dir, "best.pth"))
                    print(f"  → best SRC: {best_mpjpe*1000:.1f}mm")
            print()

        # 保存 latest
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'phase': phase,
            'in_dim': C,
            'loss': avg,
            'pose_head_old': False,
        }, os.path.join(save_dir, "latest.pth"))

    # ── 最终 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL")
    print("=" * 60)
    if os.path.exists(os.path.join(save_dir, "best.pth")):
        ckpt = torch.load(os.path.join(save_dir, "best.pth"),
                          map_location=device)
        model.load_state_dict(ckpt['model'])
        eval_classification(model, src_eval, device, "SRC-CLS")
        eval_regression(model, src_eval, device, "SRC-REG best")
        eval_regression(model, tgt_eval, device, "TGT-REG best")
        check_collapse(model, tgt_eval, device)

    print(f"\n  Saved: {save_dir}")


if __name__ == "__main__":
    main()