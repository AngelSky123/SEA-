"""
注意力权重可视化
支持：CSI 跨传感器注意力热力图、时序注意力模式

用法：
    python -m visualization.attention_vis \
        --checkpoint ./outputs/best.pth \
        --data_root /data/MMFi \
        --target E04 \
        --save_dir ./vis_output
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel


def extract_attention_weights(model, x, device):
    """
    前向传播一个 batch，提取 CrossSensorAttention 的注意力权重。
    返回 dict: {'sensor': (B, heads, N, N)}
    """
    model.eval()
    x = x.to(device)
    attn_weights = {}

    with torch.no_grad():
        B, T, N, C = x.shape
        x_flat = x.view(B * T, N, C)

        # GATLayer
        h = model.encoder.gat.fc(x_flat)

        # CrossSensorAttention（权重缓存到 last_attn_weights）
        _ = model.encoder.attn(h)
        w = model.encoder.attn.last_attn_weights   # (B*T, N, N) 或 (B*T, heads, N, N)
        if w is not None:
            # MultiheadAttention 默认返回平均后的 (B, N, N)
            w = w.view(B, T, *w.shape[1:]).mean(dim=1)   # (B, N, N)
            attn_weights['sensor'] = w.cpu().numpy()

    return attn_weights


def plot_sensor_attention(attn, save_path, n_samples=4):
    """绘制前 n_samples 个样本的传感器注意力热力图。"""
    n = min(n_samples, len(attn))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle('Cross-sensor attention weights', fontsize=13, fontweight='bold')

    for i in range(n):
        ax = axes[i]
        im = ax.imshow(attn[i], cmap='viridis', aspect='auto',
                       vmin=0, vmax=attn[i].max())
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_xlabel('Key (sensor node)')
        if i == 0:
            ax.set_ylabel('Query (sensor node)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_attention_stats(attn, save_path):
    """
    注意力分布统计：
    - 对角线强度（自注意力强度）
    - 每行熵（注意力集中程度）
    """
    B, N, N_ = attn.shape

    # 对角线平均（自注意力强度）
    diag_vals = np.array([attn[b].diagonal().mean() for b in range(B)])

    # 每个 query 的注意力熵（越低 = 越集中）
    eps = 1e-8
    entropy = -np.sum(attn * np.log(attn + eps), axis=-1)  # (B, N)
    mean_entropy = entropy.mean(axis=0)   # (N,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Attention pattern statistics', fontsize=13, fontweight='bold')

    # 自注意力强度分布
    axes[0].hist(diag_vals, bins=30, color='#2E86AB', alpha=0.8, edgecolor='white')
    axes[0].axvline(diag_vals.mean(), color='red', linestyle='--',
                    label=f'mean={diag_vals.mean():.3f}')
    axes[0].set_xlabel('Avg diagonal (self-attention strength)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Self-attention strength distribution')
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    # 每节点平均熵
    axes[1].bar(range(len(mean_entropy)), mean_entropy,
                color='#E84855', alpha=0.8)
    axes[1].set_xlabel('Sensor node index')
    axes[1].set_ylabel('Avg attention entropy')
    axes[1].set_title('Per-node attention entropy\n(lower = more focused)')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_mean_attention_map(attn, save_path):
    """所有样本的平均注意力热力图（N×N）。"""
    mean_attn = attn.mean(axis=0)   # (N, N)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_attn, cmap='plasma', aspect='auto')
    ax.set_title('Mean cross-sensor attention map', fontsize=13, fontweight='bold')
    ax.set_xlabel('Key (sensor node index)')
    ax.set_ylabel('Query (sensor node index)')
    plt.colorbar(im, ax=ax, label='Attention weight')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root',  required=True)
    parser.add_argument('--target', nargs='+', default=['E04'])
    parser.add_argument('--save_dir', default='./vis_output')
    parser.add_argument('--n_batches', type=int, default=5,
                        help='用于统计的 batch 数量')
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = WiFiPoseModel(dim=args.dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])

    ds = MMFiDataset(args.data_root, args.target)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    all_sensor_attn = []

    for i, (x, _) in enumerate(loader):
        if i >= args.n_batches:
            break
        weights = extract_attention_weights(model, x, device)
        if 'sensor' in weights:
            all_sensor_attn.append(weights['sensor'])

    if not all_sensor_attn:
        print("  [Warning] No attention weights captured. "
              "Check that CrossSensorAttention.last_attn_weights is being set.")
        return

    all_sensor_attn = np.concatenate(all_sensor_attn, axis=0)  # (N_total, N, N)

    plot_sensor_attention(all_sensor_attn,
        os.path.join(args.save_dir, 'sensor_attention_samples.png'))
    plot_mean_attention_map(all_sensor_attn,
        os.path.join(args.save_dir, 'sensor_attention_mean.png'))
    plot_attention_stats(all_sensor_attn,
        os.path.join(args.save_dir, 'attention_statistics.png'))


if __name__ == '__main__':
    main()