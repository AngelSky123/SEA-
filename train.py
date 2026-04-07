"""
v8 训练脚本：两阶段课程学习

Phase 1 (epoch 0 ~ warmup_epochs): 纯源域监督训练
  - da_weight = 0，所有 DA 损失关闭
  - 让 Encoder + PoseHead 先学会 CSI → Pose 映射
  - 此阶段结束时应在源域达到 <80mm MPJPE

Phase 2 (warmup_epochs ~ end): 渐进引入域自适应
  - da_weight 从 0 线性增至 1.0
  - GRL alpha 同步线性增长
  - IO 一致性损失确保目标域预测与 CSI 输入结构一致

关键改进：
  - 不再从 epoch 0 就激活 disentangle/align/domain
  - 每 5 epoch 同时评估源域和目标域，监控泛化差距
  - diagnose_batch 同时检查源域和目标域特征
"""

import torch
from torch.utils.data import DataLoader
import os
import time
import yaml

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from losses.losses import compute_loss
from utils.config import get_config
from utils.checkpoint import save_checkpoint, load_checkpoint
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


# ── 课程学习调度 ──────────────────────────────────────────────────────

WARMUP_EPOCHS = 25   # Phase 1 持续的 epoch 数


def get_da_weight(epoch, total_epochs, warmup=WARMUP_EPOCHS):
    """Phase 1: 0, Phase 2: 线性从 0 → 1"""
    if epoch < warmup:
        return 0.0
    return min(1.0, (epoch - warmup) / max(total_epochs - warmup - 1, 1))


def get_grl_alpha(epoch, total_epochs, max_alpha=0.05, warmup=WARMUP_EPOCHS):
    """GRL 强度，与 da_weight 同步。"""
    if epoch < warmup:
        return 0.0
    p = (epoch - warmup) / max(total_epochs - warmup - 1, 1)
    return max_alpha * p


def get_inf_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch


# ── 评估 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, tag=""):
    model.eval()
    all_pred, all_gt = [], []
    for x, y, _ in loader:
        x = x.to(device)
        out = model(x, x, alpha=0.0)
        all_pred.append(out[0].cpu())
        all_gt.append(y[:, 0].cpu())
    metrics = compute_metrics(all_pred, all_gt)
    print(f"  [{tag}] MPJPE={metrics['MPJPE']*1000:.1f}mm  "
          f"PA-MPJPE={metrics['PA-MPJPE']*1000:.1f}mm  "
          f"PCK@50mm={metrics['PCK@0.05']:.3f}")
    return metrics


@torch.no_grad()
def diagnose_batch(model, xs, xt, device, epoch, step):
    """检查源域和目标域的特征塌陷状态。"""
    model.eval()

    for tag, data in [("SRC", xs), ("TGT", xt)]:
        enc = model.encoder(data)
        feat = enc.mean(dim=(1, 2))
        feat_std = feat.std(dim=0).mean().item()
        feat_norm = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
        cos = feat_norm @ feat_norm.T
        B = cos.shape[0]
        mask = torch.triu(torch.ones(B, B, device=device), diagonal=1).bool()
        cs = cos[mask].mean().item()
        st = "⚠️" if cs > 0.95 else ("⚡" if cs > 0.8 else "✓")
        print(f"  [Diag E{epoch:03d} {tag}] std={feat_std:.3f} cos={cs:.4f} {st}")

    model.train()


# ── 主训练循环 ────────────────────────────────────────────────────────

def main():
    cfg = get_config()

    exp_name = "v8_" + time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg.__dict__, f)

    print("=" * 60)
    print("  v8 CURRICULUM TRAINING")
    print(f"  Phase 1 (epoch 0-{WARMUP_EPOCHS-1}): source-only supervision")
    print(f"  Phase 2 (epoch {WARMUP_EPOCHS}+): gradual DA + IO consistency")
    print("=" * 60)

    cache_dir = getattr(cfg.data, "cache_dir", None)
    source_data = MMFiDataset(cfg.data.root, cfg.domain.source,
                              cfg.data.seq_len, cache_dir=cache_dir)
    target_data = MMFiDataset(cfg.data.root, cfg.domain.target,
                              cfg.data.seq_len, cache_dir=cache_dir)

    sample_csi, _, _ = source_data[0]
    in_dim = sample_csi.shape[-1]
    print(f"  in_dim={in_dim}")

    source_loader = DataLoader(source_data, batch_size=cfg.train.batch_size,
                               shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, timeout=60)
    target_loader = DataLoader(target_data, batch_size=cfg.train.batch_size,
                               shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, timeout=60)
    # 评估用
    src_eval = DataLoader(source_data, batch_size=64, shuffle=False, num_workers=4)
    tgt_eval = DataLoader(target_data, batch_size=64, shuffle=False, num_workers=4)

    model = WiFiPoseModel(
        in_dim=in_dim,
        dim=cfg.model.dim,
        num_joints=cfg.model.num_joints,
        pose_head_old=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.01)

    start_epoch = 0
    if cfg.resume:
        model, optimizer, start_epoch = load_checkpoint(
            cfg.resume, model, optimizer)

    best_loss = float('inf')
    loss_keys = ["pose", "vel", "bone", "align", "domain", "orth",
                 "var", "cov", "io"]

    target_iter = iter(get_inf_iterator(target_loader))

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        da_w  = get_da_weight(epoch, cfg.train.epochs)
        alpha = get_grl_alpha(epoch, cfg.train.epochs)

        phase = "Phase1-SrcOnly" if da_w == 0 else f"Phase2-DA(w={da_w:.2f})"
        if epoch % 5 == 0:
            print(f"\n  === {phase} ===")

        epoch_loss_sum = 0.0
        epoch_steps    = 0
        comp_sum       = {k: 0.0 for k in loss_keys}

        for i, batch_s in enumerate(source_loader):
            xs, ys, _ = batch_s
            xt, _, _  = next(target_iter)

            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)

            optimizer.zero_grad()

            with _autocast_ctx():
                (pose, vel_pred, fs, ft, ds, dt, orth_loss,
                 pose_t, enc_feat_s, enc_feat_t) = model(xs, xt, alpha=alpha)

                loss, components = compute_loss(
                    pose, vel_pred, ys, fs, ft, ds, dt, orth_loss,
                    pred_t=pose_t,
                    enc_feat_s=enc_feat_s,
                    enc_feat_t=enc_feat_t,
                    csi_s=xs, csi_t=xt,
                    da_weight=da_w)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [E{epoch}] NaN at iter {i}, skip")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += loss.item()
            epoch_steps    += 1
            for k in loss_keys:
                v = components.get(k, 0.0)
                comp_sum[k] += v.item() if torch.is_tensor(v) else v

            if i % cfg.log.print_freq == 0:
                parts = " ".join(f"{k}={components.get(k,0):.4f}"
                                 for k in loss_keys)
                print(f"[E{epoch:03d}][{i:04d}/{len(source_loader)}] "
                      f"loss={loss.item():.4f}  {parts}  "
                      f"da_w={da_w:.2f} alpha={alpha:.3f}")

            if i % 1000 == 0:
                diagnose_batch(model, xs, xt, device, epoch, i)

        scheduler.step()
        if epoch_steps == 0:
            continue

        epoch_avg = epoch_loss_sum / epoch_steps
        parts = " | ".join(f"{k}={comp_sum[k]/epoch_steps:.4f}"
                           for k in loss_keys)
        print(f" Epoch {epoch:03d} [{phase}] avg={epoch_avg:.4f} | {parts}")

        # ── 定期评估 ─────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == cfg.train.epochs - 1:
            print(f"\n  --- Epoch {epoch} evaluation ---")
            evaluate(model, src_eval, device, "SOURCE")
            tgt_m = evaluate(model, tgt_eval, device, "TARGET")
            print()

        # ── 保存 ─────────────────────────────────────────────────
        state = {
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'epoch':         epoch,
            'loss':          epoch_avg,
            'in_dim':        in_dim,
            'pose_head_old': False,
        }
        save_checkpoint(state, save_dir, "latest.pth")
        if epoch_avg < best_loss:
            best_loss = epoch_avg
            save_checkpoint(state, save_dir, "best.pth")
            print(f"  → new best: {best_loss:.4f}")

    print("\n Training Finished")


if __name__ == "__main__":
    main()