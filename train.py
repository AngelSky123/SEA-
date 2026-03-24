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

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# 引入混合精度训练 (AMP)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def get_grl_alpha(epoch, total_epochs, max_alpha=1.0):
    p = epoch / max(total_epochs - 1, 1)
    return max_alpha * p

def get_inf_iterator(dataloader):
    """
    安全的无限数据流生成器。
    替换原有的 itertools.cycle，彻底避免目标域数据的内存泄漏和多进程死锁。
    """
    while True:
        for batch in dataloader:
            yield batch

def main():
    cfg = get_config()

    exp_name = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg.__dict__, f)

    print("===== CONFIG =====")
    print(cfg.__dict__)

    cache_dir   = getattr(cfg.data, "cache_dir", None)
    source_data = MMFiDataset(cfg.data.root, cfg.domain.source,
                              cfg.data.seq_len, cache_dir=cache_dir)
    target_data = MMFiDataset(cfg.data.root, cfg.domain.target,
                              cfg.data.seq_len, cache_dir=cache_dir)

    # 调低 num_workers 到 4 防止 IO 瓶颈，加入 timeout=60 防止 DataLoader 死锁
    source_loader = DataLoader(source_data, batch_size=cfg.train.batch_size,
                               shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, timeout=60)
    target_loader = DataLoader(target_data, batch_size=cfg.train.batch_size,
                               shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True, timeout=60)

    model = WiFiPoseModel(dim=cfg.model.dim,
                          num_joints=cfg.model.num_joints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.01)

    start_epoch = 0
    if cfg.resume:
        model, optimizer, start_epoch = load_checkpoint(
            cfg.resume, model, optimizer)

    best_loss = float('inf')
    loss_keys = ["pose", "vel", "bone", "align", "domain", "orth"]

    # 在 epoch 循环外初始化安全的无限迭代器
    target_iter = iter(get_inf_iterator(target_loader))

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        alpha = get_grl_alpha(epoch, cfg.train.epochs)

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
            
            # 使用混合精度加速前向传播
            with autocast():
                pose, vel_pred, fs, ft, ds, dt, orth_loss = model(xs, xt, alpha=alpha)
                loss, components = compute_loss(
                    pose, vel_pred, ys, fs, ft, ds, dt, orth_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Epoch {epoch}] NaN/Inf at iter {i}, skip")
                continue

            # 使用 scaler 缩放并反向传播
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_sum += loss.item()
            epoch_steps    += 1
            
            # 修复内存泄漏核心：提取纯数值，强制剥离 Tensor 计算图
            for k in loss_keys:
                val = components.get(k, 0.0)
                if torch.is_tensor(val):
                    val = val.item()  # 提取标量，释放显存和内存
                comp_sum[k] += val

            if i % cfg.log.print_freq == 0:
                parts = " ".join(
                    f"{k}={components.get(k,0):.4f}" for k in loss_keys)
                print(f"[Epoch {epoch:03d}][{i:04d}/{len(source_loader)}] "
                      f"loss={loss.item():.4f}  {parts}  alpha={alpha:.3f}")

        scheduler.step()
        if epoch_steps == 0:
            continue

        epoch_avg = epoch_loss_sum / epoch_steps
        parts = " | ".join(
            f"{k}={comp_sum[k]/epoch_steps:.4f}" for k in loss_keys)
        print(f" Epoch {epoch:03d} done | avg={epoch_avg:.4f} | {parts}")

        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch, 'loss': epoch_avg}
        save_checkpoint(state, save_dir, "latest.pth")
        if epoch_avg < best_loss:
            best_loss = epoch_avg
            save_checkpoint(state, save_dir, "best.pth")
            print(f"  → new best: {best_loss:.4f}")

    print(" Training Finished")

if __name__ == "__main__":
    main()