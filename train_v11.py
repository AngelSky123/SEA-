"""
v11.1: 分类预训练 + 回归微调（修复目标域分类崩塌）

v11 的问题：
  Phase 1 只用源域数据训练分类 → Encoder 过拟合到源域特征
  SRC accuracy=98%, TGT accuracy=6.6%（接近随机）

修复：
  Phase 1 同时用源域+目标域做分类（目标域也有动作标签 A01-A27）
  → Encoder 被迫学习跨域的运动特征
  → 目标域分类准确率也应 >80%
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

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
                    action = p
                    break
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

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        csi, pose, root_offset = self.base[idx]
        action_id = self.label_map[self.action_labels[idx]]
        return csi, pose, root_offset, action_id

class PoseHead(nn.Module):
    def __init__(self, dim, num_joints=17, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_joints * 3),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_joints * 3),
        )

    def forward(self, feat):
        B = feat.shape[0]
        x_spa = feat.mean(dim=2)
        x_max = x_spa.max(dim=1).values
        motion = x_spa[:, -1] - x_spa[:, 0]
        pose_in = torch.cat([x_max, motion], dim=-1)
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
        x = torch.cat([x_max, motion], dim=-1)
        return self.head(x)

class FullModel(nn.Module):
    def __init__(self, in_dim, dim, n_classes, num_joints=17):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHead(dim, num_joints)

    def forward(self, x, mode='both'):
        feat = self.encoder(x)
        if mode == 'cls':
            return self.cls_head(feat)
        elif mode == 'reg':
            return self.reg_head(feat)
        else:
            logits = self.cls_head(feat)
            pose, vel = self.reg_head(feat)
            return logits, pose, vel

def regression_loss(pred, vel_pred, gt):
    pose_loss = F.l1_loss(pred, gt[:, 0])
    gt_vel = gt[:, -1] - gt[:, 0]
    vel_loss = F.l1_loss(vel_pred, gt_vel)
    bone_loss = torch.tensor(0.0, device=pred.device)
    for i, j in EDGES:
        pred_len = torch.norm(pred[:, i] - pred[:, j], dim=-1)
        gt_len = torch.norm(gt[:, 0, i] - gt[:, 0, j], dim=-1)
        bone_loss = bone_loss + F.l1_loss(pred_len, gt_len)
    bone_loss = bone_loss / len(EDGES)
    return pose_loss + 0.1 * vel_loss + 0.1 * bone_loss, {
        "pose": pose_loss.item(), "vel": vel_loss.item(), "bone": bone_loss.item(),
    }

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
        if i >= 5: break
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

def main():
    cfg = get_config()
    total_epochs = cfg.train.epochs

    exp_name = "v11_clspretrain_" + time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    p1 = PHASE1_EPOCHS
    p2 = PHASE2_EPOCHS

    print("=" * 60)
    print("  v11.1: 跨域分类预训练 → 冻结回归 → 端到端微调")
    print(f"  Phase 1 (epoch 0-{p1-1}):  源域+目标域联合分类")
    print(f"  Phase 2 (epoch {p1}-{p1+p2-1}): 冻结 Encoder, 训练 PoseHead")
    print(f"  Phase 3 (epoch {p1+p2}-{total_epochs-1}): 端到端微调")
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
    print(f"  CSI: T={T}, N={N}, C={C}")
    print(f"  动作类别: {n_classes}")
    print(f"  源域: {len(src_data)}, 目标域: {len(tgt_data)}")

    combined_data = ConcatDataset([src_data, tgt_data])
    combined_loader = DataLoader(combined_data, batch_size=cfg.train.batch_size,
                                 shuffle=True, num_workers=4,
                                 pin_memory=True, drop_last=True)
    print(f"  Phase 1 联合数据集: {len(combined_data)} 样本")

    src_loader = DataLoader(src_data, batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)

    eval_bs = min(512, cfg.train.batch_size * 2)
    src_eval = DataLoader(src_data, batch_size=eval_bs, shuffle=False, num_workers=4)
    tgt_eval = DataLoader(tgt_data, batch_size=eval_bs, shuffle=False, num_workers=4)

    model = FullModel(in_dim=C, dim=cfg.model.dim, n_classes=n_classes,
                      num_joints=cfg.model.num_joints).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    best_mpjpe = float('inf')

    for epoch in range(total_epochs):
        model.train()

        if epoch < p1:
            phase, phase_name = 1, "Phase1-CLS"
        elif epoch < p1 + p2:
            phase, phase_name = 2, "Phase2-REG(frozen)"
        else:
            phase, phase_name = 3, "Phase3-FINETUNE"

        if epoch == 0:
            params = list(model.encoder.parameters()) + list(model.cls_head.parameters())
            optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=1e-4)
            print(f"\n  >>> Phase 1: 源域+目标域联合分类")

        elif epoch == p1:
            for p in model.encoder.parameters(): p.requires_grad = False
            for p in model.cls_head.parameters(): p.requires_grad = False
            params = list(model.reg_head.parameters())
            optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=1e-4)
            print(f"\n  >>> Phase 2: 冻结 Encoder, 训练 PoseHead")

        elif epoch == p1 + p2:
            for p in model.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': cfg.train.lr * 0.1},
                {'params': model.cls_head.parameters(), 'lr': cfg.train.lr * 0.1},
                {'params': model.reg_head.parameters(), 'lr': cfg.train.lr},
            ], weight_decay=1e-4)
            print(f"\n  >>> Phase 3: 端到端微调 (Encoder LR=x0.1)")

        active_loader = combined_loader if phase == 1 else src_loader

        epoch_loss, epoch_steps = 0.0, 0
        comp_sum = {}

        for i, batch in enumerate(active_loader):
            xs = batch[0].to(device, non_blocking=True)
            ys = batch[1].to(device, non_blocking=True)
            action = batch[3].to(device, non_blocking=True)

            optimizer.zero_grad()
            with _autocast_ctx():
                if phase == 1:
                    logits = model(xs, mode='cls')
                    loss = F.cross_entropy(logits, action)
                    comp = {"cls": loss.item()}
                elif phase == 2:
                    pose, vel = model(xs, mode='reg')
                    loss, comp = regression_loss(pose, vel, ys)
                else:
                    logits, pose, vel = model(xs, mode='both')
                    cls_loss = F.cross_entropy(logits, action)
                    reg_loss, reg_comp = regression_loss(pose, vel, ys)
                    loss = reg_loss + 0.5 * cls_loss
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
                parts = " ".join(f"{k}={v:.4f}" for k, v in comp.items())
                print(f"[E{epoch:03d}][{i:04d}/{len(active_loader)}] "
                      f"[{phase_name}] loss={loss.item():.4f}  {parts}")

        if epoch_steps == 0: continue

        avg = epoch_loss / epoch_steps
        parts = " | ".join(f"{k}={comp_sum[k]/epoch_steps:.4f}" for k in comp_sum)
        print(f" Epoch {epoch:03d} [{phase_name}] avg={avg:.4f} | {parts}")

        if epoch % 3 == 0 or epoch == total_epochs - 1 or \
           epoch == p1 - 1 or epoch == p1 or epoch == p1 + p2 - 1 or epoch == p1 + p2:
            print(f"\n  --- Epoch {epoch} [{phase_name}] ---")
            eval_classification(model, src_eval, device, "SRC-CLS")
            eval_classification(model, tgt_eval, device, "TGT-CLS")

            if phase >= 2:
                src_m = eval_regression(model, src_eval, device, "SRC-REG")
                tgt_m = eval_regression(model, tgt_eval, device, "TGT-REG")
                check_collapse(model, tgt_eval, device)
                if src_m['MPJPE'] < best_mpjpe:
                    best_mpjpe = src_m['MPJPE']
                    torch.save({
                        'model': model.state_dict(), 'epoch': epoch,
                        'phase': phase, 'in_dim': C,
                        'src_mpjpe': src_m['MPJPE'], 'tgt_mpjpe': tgt_m['MPJPE'],
                        'pose_head_old': False,
                    }, os.path.join(save_dir, "best.pth"))
                    print(f"  -> best SRC: {best_mpjpe*1000:.1f}mm")
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
        eval_classification(model, src_eval, device, "SRC-CLS")
        eval_classification(model, tgt_eval, device, "TGT-CLS")
        eval_regression(model, src_eval, device, "SRC-REG best")
        eval_regression(model, tgt_eval, device, "TGT-REG best")
        check_collapse(model, tgt_eval, device)
    print(f"\n  Saved: {save_dir}")

if __name__ == "__main__":
    main()