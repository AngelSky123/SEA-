"""
严格 UDA - SHOT 方法 (Source Hypothesis Transfer)

Why SHOT works when GRL/pseudo-labels fail:
  - GRL 试图让两个域的特征分布对齐 → 域差距太大，失败
  - 伪标签需要初始准确率 > 30% → 只有 6.5%，失败
  - SHOT 不做域对齐，也不依赖初始准确率
    它在目标域上最小化分类熵（鼓励高置信预测）
    + 最大化类别多样性（防止所有样本塌陷到同一类）
    两者结合 = information maximization (IM)

流程：
  Stage 1: 源域正常训练（分类 + 回归）
  Stage 2: SHOT 适应（只更新 Encoder，冻结分类头和回归头）
    - 目标域 CSI 输入 Encoder → cls_head 输出 logits
    - L_ent: 最小化单样本预测熵（鼓励高置信度）
    - L_div: 最大化 batch 级别类别分布的熵（防止单类塌陷）
    - 只更新 Encoder 参数（分类/回归头冻结 = "hypothesis transfer"）
  Stage 3: 冻结 Encoder，训练 PoseHead（用源域）
  Stage 4: 端到端微调（用源域）

全程不使用目标域任何标签（严格 UDA）。

用法:
    python train_uda_shot.py --epochs 80 --batch_size 256
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

STAGE1_EPOCHS = 15    # 源域分类
SHOT_EPOCHS = 10      # SHOT 适应
STAGE3_EPOCHS = 20    # 冻结回归
ENC_LR_RATIO = 0.05

clean_argv = []
skip_next = False
for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    if arg == '--stage1_epochs':
        STAGE1_EPOCHS = int(sys.argv[i+1]); skip_next = True; continue
    if arg == '--shot_epochs':
        SHOT_EPOCHS = int(sys.argv[i+1]); skip_next = True; continue
    if arg == '--stage3_epochs':
        STAGE3_EPOCHS = int(sys.argv[i+1]); skip_next = True; continue
    if arg == '--enc_lr_ratio':
        ENC_LR_RATIO = float(sys.argv[i+1]); skip_next = True; continue
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
                    action = p; break
            self.action_labels.append(action)
            self.action_set.add(action)
        if label_map is not None:
            self.label_map = label_map
        else:
            self.action_set = sorted(self.action_set)
            self.label_map = {a: i for i, a in enumerate(self.action_set)}
        self.n_classes = len(self.label_map)
        print(f"  动作类别数: {self.n_classes}, 样本数: {len(self.base)}")
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        csi, pose, root_offset = self.base[idx]
        return csi, pose, root_offset, self.label_map[self.action_labels[idx]]

class TgtEvalDataset(Dataset):
    def __init__(self, base_dataset, label_map):
        self.base = base_dataset
        self.labels = []
        for sample in self.base.samples:
            parts = sample['csi_dir'].split(os.sep)
            action = 'UNK'
            for p in parts:
                if p.startswith('A') and len(p) == 3 and p[1:].isdigit():
                    action = p; break
            self.labels.append(label_map.get(action, 0))
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        csi, pose, root_offset = self.base[idx]
        return csi, pose, root_offset, self.labels[idx]

# ═══════════════════════════════════════════════════════════════════════
#  模型
# ═══════════════════════════════════════════════════════════════════════

class PoseHead(nn.Module):
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
    def forward(self, feat):
        B = feat.shape[0]
        x_spa = feat.mean(dim=2)
        x_max = x_spa.max(dim=1).values
        motion = x_spa[:, -1] - x_spa[:, 0]
        pose_in = torch.cat([x_max, motion], dim=-1)
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
    def __init__(self, in_dim, dim, n_classes, num_joints=17):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, dim=dim)
        self.cls_head = ClassificationHead(dim, n_classes)
        self.reg_head = PoseHead(dim, num_joints)
    def forward_cls(self, x):
        return self.cls_head(self.encoder(x))
    def forward_reg(self, x):
        return self.reg_head(self.encoder(x))
    def forward(self, x, mode='both'):
        feat = self.encoder(x)
        if mode == 'cls': return self.cls_head(feat)
        elif mode == 'reg': return self.reg_head(feat)
        else:
            return self.cls_head(feat), *self.reg_head(feat)

# ═══════════════════════════════════════════════════════════════════════
#  SHOT 损失函数
# ═══════════════════════════════════════════════════════════════════════

def shot_loss(logits):
    """
    Information Maximization loss (SHOT 核心)

    L = L_ent - L_div

    L_ent: 单样本预测熵（最小化 → 鼓励高置信度）
      每个样本的 softmax 输出应接近 one-hot

    L_div: batch 级别类别分布熵（最大化 → 鼓励类别多样性）
      batch 内所有样本的平均 softmax 应接近均匀分布
      防止所有样本塌陷到同一个类别

    两者方向相反：
      L_ent 最小化 → 每个样本高置信
      L_div 最大化 → 不同样本分到不同类
    """
    probs = F.softmax(logits, dim=1)                          # (B, C)

    # 单样本熵（越小 = 越确信）
    ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1)       # (B,)
    L_ent = ent.mean()

    # batch 级别类别分布熵（越大 = 越多样）
    mean_probs = probs.mean(dim=0)                             # (C,)
    L_div = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()

    # IM loss = minimize entropy - maximize diversity
    return L_ent - L_div, {
        "ent": L_ent.item(),
        "div": L_div.item(),
    }

# ═══════════════════════════════════════════════════════════════════════
#  标准损失 & 评估
# ═══════════════════════════════════════════════════════════════════════

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
        x = batch[0].to(device); y = batch[1]
        pose, _ = model.forward_reg(x)
        all_pred.append(pose.cpu()); all_gt.append(y[:, 0])
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
        logits = model.forward_cls(x)
        correct += (logits.argmax(1) == action).sum().item()
        total += x.shape[0]
    acc = correct / total
    print(f"  [{tag}] Acc: {acc:.1%}")
    model.train(); return acc

@torch.no_grad()
def check_collapse(model, loader, device):
    model.eval()
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if i >= 5: break
        x = batch[0].to(device); y = batch[1]
        pose, _ = model.forward_reg(x)
        preds.append(pose.cpu()); gts.append(y[:, 0])
    preds = torch.cat(preds); gts = torch.cat(gts)
    pred_std = preds.std(dim=0).mean().item() * 1000
    gt_std = gts.std(dim=0).mean().item() * 1000
    ratio = pred_std / (gt_std + 1e-8)
    st = "⚠️" if ratio < 0.3 else ("⚡" if ratio < 0.6 else "✓")
    print(f"  [Collapse] pred_std={pred_std:.1f}mm  gt_std={gt_std:.1f}mm  "
          f"ratio={ratio:.3f}  {st}")
    model.train(); return ratio

# ═══════════════════════════════════════════════════════════════════════
#  主训练
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = get_config()
    total_epochs = cfg.train.epochs
    s1 = STAGE1_EPOCHS
    s_shot = SHOT_EPOCHS
    s3 = STAGE3_EPOCHS
    s4 = max(0, total_epochs - s1 - s_shot - s3)
    enc_lr_ratio = ENC_LR_RATIO

    exp_name = "uda_shot_" + time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 日志记录
    from utils.logger import setup_logger
    setup_logger(os.path.join(save_dir, "train.log"))

    print("=" * 60)
    print("  严格 UDA - SHOT (Information Maximization)")
    print(f"  Stage 1 (0-{s1-1}):          源域分类")
    print(f"  Stage 2 ({s1}-{s1+s_shot-1}):  SHOT 目标域适应")
    print(f"  Stage 3 ({s1+s_shot}-{s1+s_shot+s3-1}): 冻结回归")
    print(f"  Stage 4 ({s1+s_shot+s3}-{total_epochs-1}): 微调")
    print(f"  目标域标签: 完全不使用")
    print("=" * 60)

    cache_dir = getattr(cfg.data, "cache_dir", None)
    src_base = MMFiDataset(cfg.data.root, cfg.domain.source,
                           cfg.data.seq_len, cache_dir=cache_dir)
    tgt_base = MMFiDataset(cfg.data.root, cfg.domain.target,
                           cfg.data.seq_len, cache_dir=cache_dir)

    src_data = MMFiWithAction(src_base)
    label_map = src_data.label_map
    n_classes = src_data.n_classes
    tgt_eval_data = TgtEvalDataset(tgt_base, label_map)

    sample_csi, _, _, _ = src_data[0]
    T, N, C = sample_csi.shape
    print(f"  CSI: T={T}, N={N}, C={C}")

    src_loader = DataLoader(src_data, batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    tgt_loader = DataLoader(tgt_base, batch_size=cfg.train.batch_size,
                            shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)
    eval_bs = min(512, cfg.train.batch_size * 2)
    src_eval = DataLoader(src_data, batch_size=eval_bs, shuffle=False, num_workers=4)
    tgt_eval = DataLoader(tgt_eval_data, batch_size=eval_bs, shuffle=False, num_workers=4)

    model = FullModel(in_dim=C, dim=cfg.model.dim, n_classes=n_classes,
                      num_joints=cfg.model.num_joints).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_tgt_mpjpe = float('inf')
    global_epoch = 0

    # ══════════════════════════════════════════════════════════════
    #  Stage 1: 源域分类
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*40}")
    print(f"  Stage 1: 源域分类 ({s1} epochs)")
    print(f"{'='*40}")

    params = list(model.encoder.parameters()) + list(model.cls_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=s1, eta_min=cfg.train.lr * 0.01)

    for ep in range(s1):
        model.train()
        ep_loss, ep_steps = 0, 0
        for i, batch in enumerate(src_loader):
            xs = batch[0].to(device, non_blocking=True)
            action = batch[3].to(device, non_blocking=True)
            optimizer.zero_grad()
            with _autocast_ctx():
                logits = model.forward_cls(xs)
                loss = F.cross_entropy(logits, action)
            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            ep_loss += loss.item(); ep_steps += 1
            if i % cfg.log.print_freq == 0:
                print(f"[E{global_epoch:03d}][{i:04d}/{len(src_loader)}] "
                      f"[Stage1-CLS] cls={loss.item():.4f}")
        scheduler.step()
        if ep_steps > 0:
            print(f" Epoch {global_epoch} [Stage1] avg_cls={ep_loss/ep_steps:.4f}")

        if ep % 5 == 0 or ep == s1 - 1:
            print(f"\n  --- Stage 1 epoch {ep} ---")
            eval_classification(model, src_eval, device, "SRC-CLS")
            eval_classification(model, tgt_eval, device, "TGT-CLS(监控)")
            print()
        global_epoch += 1

    # ══════════════════════════════════════════════════════════════
    #  Stage 2: SHOT 目标域适应
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*40}")
    print(f"  Stage 2: SHOT 适应 ({s_shot} epochs)")
    print(f"  只更新 Encoder，冻结 cls_head + reg_head")
    print(f"{'='*40}")

    # 冻结分类头和回归头（hypothesis transfer 的核心）
    for p in model.cls_head.parameters(): p.requires_grad = False
    for p in model.reg_head.parameters(): p.requires_grad = False
    # 只训练 Encoder
    optimizer = torch.optim.AdamW(
        model.encoder.parameters(), lr=cfg.train.lr * 0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=s_shot, eta_min=cfg.train.lr * 0.001)

    for ep in range(s_shot):
        model.train()
        ep_loss, ep_steps = 0, 0
        ent_sum, div_sum = 0, 0

        for i, batch in enumerate(tgt_loader):
            xt = batch[0].to(device, non_blocking=True)

            optimizer.zero_grad()
            with _autocast_ctx():
                logits = model.forward_cls(xt)
                loss, comp = shot_loss(logits)

            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            ep_loss += loss.item(); ep_steps += 1
            ent_sum += comp['ent']; div_sum += comp['div']

            if i % cfg.log.print_freq == 0:
                print(f"[E{global_epoch:03d}][{i:04d}/{len(tgt_loader)}] "
                      f"[SHOT] loss={loss.item():.4f}  "
                      f"ent={comp['ent']:.4f}  div={comp['div']:.4f}")

        scheduler.step()
        if ep_steps > 0:
            print(f" Epoch {global_epoch} [SHOT] avg={ep_loss/ep_steps:.4f}  "
                  f"ent={ent_sum/ep_steps:.4f}  div={div_sum/ep_steps:.4f}")

        if ep % 3 == 0 or ep == s_shot - 1:
            print(f"\n  --- SHOT epoch {ep} ---")
            eval_classification(model, src_eval, device, "SRC-CLS")
            tgt_cls = eval_classification(model, tgt_eval, device, "TGT-CLS(监控)")
            print()

        global_epoch += 1

    # ══════════════════════════════════════════════════════════════
    #  Stage 3: 冻结回归
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*40}")
    print(f"  Stage 3: 冻结回归 ({s3} epochs)")
    print(f"{'='*40}")

    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.cls_head.parameters(): p.requires_grad = False
    for p in model.reg_head.parameters(): p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.reg_head.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=s3, eta_min=cfg.train.lr * 0.01)

    for ep in range(s3):
        model.train()
        ep_loss, ep_steps = 0, 0
        for i, batch in enumerate(src_loader):
            xs = batch[0].to(device, non_blocking=True)
            ys = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            with _autocast_ctx():
                pose, vel = model.forward_reg(xs)
                loss, comp = regression_loss(pose, vel, ys)
            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            ep_loss += loss.item(); ep_steps += 1
            if i % cfg.log.print_freq == 0:
                parts = " ".join(f"{k}={v:.4f}" for k, v in comp.items())
                print(f"[E{global_epoch:03d}][{i:04d}/{len(src_loader)}] "
                      f"[Stage3-REG] {parts}")
        scheduler.step()
        if ep_steps > 0:
            print(f" Epoch {global_epoch} [Stage3] avg={ep_loss/ep_steps:.4f}")

        if ep % 5 == 0 or ep == s3 - 1:
            print(f"\n  --- Stage 3 epoch {ep} ---")
            src_m = eval_regression(model, src_eval, device, "SRC-REG")
            tgt_m = eval_regression(model, tgt_eval, device, "TGT-REG")
            check_collapse(model, tgt_eval, device)
            if tgt_m['MPJPE'] < best_tgt_mpjpe:
                best_tgt_mpjpe = tgt_m['MPJPE']
                torch.save({
                    'model': model.state_dict(), 'epoch': global_epoch,
                    'in_dim': C, 'src_mpjpe': src_m['MPJPE'],
                    'tgt_mpjpe': tgt_m['MPJPE'], 'pose_head_old': False,
                }, os.path.join(save_dir, "best.pth"))
                print(f"  -> best TGT: {best_tgt_mpjpe*1000:.1f}mm")
            print()
        global_epoch += 1

    # ══════════════════════════════════════════════════════════════
    #  Stage 4: 端到端微调
    # ══════════════════════════════════════════════════════════════
    if s4 > 0:
        print(f"\n{'='*40}")
        print(f"  Stage 4: 微调 ({s4} epochs)")
        print(f"{'='*40}")

        for p in model.parameters(): p.requires_grad = True
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(),  'lr': cfg.train.lr * enc_lr_ratio},
            {'params': model.cls_head.parameters(), 'lr': cfg.train.lr * enc_lr_ratio},
            {'params': model.reg_head.parameters(), 'lr': cfg.train.lr * 0.5},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=s4, eta_min=cfg.train.lr * 0.001)

        for ep in range(s4):
            model.train()
            ep_loss, ep_steps = 0, 0
            for i, batch in enumerate(src_loader):
                xs = batch[0].to(device, non_blocking=True)
                ys = batch[1].to(device, non_blocking=True)
                action = batch[3].to(device, non_blocking=True)
                optimizer.zero_grad()
                with _autocast_ctx():
                    logits = model.forward_cls(xs)
                    pose, vel = model.forward_reg(xs)
                    reg_loss, reg_comp = regression_loss(pose, vel, ys)
                    cls_loss = F.cross_entropy(logits, action)
                    loss = reg_loss + 0.5 * cls_loss
                if torch.isnan(loss): continue
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                ep_loss += loss.item(); ep_steps += 1
                if i % cfg.log.print_freq == 0:
                    print(f"[E{global_epoch:03d}][{i:04d}/{len(src_loader)}] "
                          f"[Stage4-FT] loss={loss.item():.4f}")
            scheduler.step()

            if ep % 5 == 0 or ep == s4 - 1:
                eval_classification(model, tgt_eval, device, "TGT-CLS(监控)")
                src_m = eval_regression(model, src_eval, device, "SRC-REG")
                tgt_m = eval_regression(model, tgt_eval, device, "TGT-REG")
                check_collapse(model, tgt_eval, device)
                if tgt_m['MPJPE'] < best_tgt_mpjpe:
                    best_tgt_mpjpe = tgt_m['MPJPE']
                    torch.save({
                        'model': model.state_dict(), 'epoch': global_epoch,
                        'in_dim': C, 'src_mpjpe': src_m['MPJPE'],
                        'tgt_mpjpe': tgt_m['MPJPE'], 'pose_head_old': False,
                    }, os.path.join(save_dir, "best.pth"))
                    print(f"  -> best TGT: {best_tgt_mpjpe*1000:.1f}mm")
            global_epoch += 1

    # ── 最终 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL")
    print("=" * 60)
    if os.path.exists(os.path.join(save_dir, "best.pth")):
        ckpt = torch.load(os.path.join(save_dir, "best.pth"), map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"  Best epoch: {ckpt.get('epoch', '?')}")
        eval_classification(model, src_eval, device, "SRC-CLS")
        eval_classification(model, tgt_eval, device, "TGT-CLS(监控)")
        eval_regression(model, src_eval, device, "SRC-REG")
        eval_regression(model, tgt_eval, device, "TGT-REG")
        check_collapse(model, tgt_eval, device)
    torch.save({'model': model.state_dict(), 'in_dim': C, 'pose_head_old': False},
               os.path.join(save_dir, "latest.pth"))
    print(f"\n  Saved: {save_dir}")

if __name__ == "__main__":
    main()