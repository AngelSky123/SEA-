import torch
from torch.utils.data import DataLoader
import os
import time
import yaml
import itertools  # ⬆ 新增导入 itertools

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from losses.losses import compute_loss
from utils.config import get_config
from utils.checkpoint import save_checkpoint, load_checkpoint

# ===== GPU优化 =====
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== AMP =====
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


def main():
    cfg = get_config()

    # ===== 创建实验目录 =====
    exp_name = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # ===== 保存 config =====
    with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg.__dict__, f)

    print("===== CONFIG =====")
    print(cfg.__dict__)

    # ===== Dataset =====
    source_data = MMFiDataset(
        cfg.data.root,
        cfg.domain.source,
        cfg.data.seq_len
    )

    target_data = MMFiDataset(
        cfg.data.root,
        cfg.domain.target,
        cfg.data.seq_len
    )

    # ===== DataLoader（高性能）=====
    source_loader = DataLoader(
        source_data,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    target_loader = DataLoader(
        target_data,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # ===== Model =====
    model = WiFiPoseModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # ===== Resume =====
    start_epoch = 0
    if cfg.resume:
        model, optimizer, start_epoch = load_checkpoint(cfg.resume, model, optimizer)

    best_loss = 1e10

    # ===== Training =====
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()

        #  核心修复：将较短的目标域 DataLoader 转为无限循环迭代器
        target_iter = itertools.cycle(target_loader)

        #  核心修复：以数据量更大的源域 source_loader 为主循环
        for i, (xs, ys) in enumerate(source_loader):
            
            # 从目标域迭代器中获取一个 batch
            xt, _ = next(target_iter)

            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)

            # ===== Mixed Precision =====
            with autocast():
                pred, fs, ft, ds, dt = model(xs, xt)
                loss = compute_loss(pred, ys, fs, ft, ds, dt)

            # ===== NaN保护 =====
            if torch.isnan(loss) or torch.isinf(loss):
                print(" NaN/Inf loss, skip batch")
                continue

            optimizer.zero_grad()

            # ===== AMP backward =====
            scaler.scale(loss).backward()

            # ===== 梯度裁剪 =====
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            # ===== Log =====
            if i % cfg.log.print_freq == 0:
                print(f"[Epoch {epoch}] Iter {i} Loss {loss.item():.4f}")

        # ===== 保存checkpoint =====
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }

        save_checkpoint(state, save_dir, "latest.pth")
        save_checkpoint(state, save_dir, f"epoch_{epoch}.pth")

        # ===== best model =====
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(state, save_dir, "best.pth")

        print(f" Epoch {epoch} Done")

    print(" Training Finished")


if __name__ == "__main__":
    main()