"""
域对齐 t-SNE / UMAP 可视化（升级版）
支持：分层着色（域 + 动作类别）、before/after 对比、特征质量量化

用法：
    python -m visualization.tsne_analysis \
        --checkpoint ./outputs/run1/best.pth \
        --data_root /data/MMFi \
        --source E01 E02 E03 \
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
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from utils.config import get_config


# ── 颜色方案 ─────────────────────────────────────────────────────────────
DOMAIN_COLORS = {'source': '#2E86AB', 'target': '#E84855'}
ACTION_CMAP   = plt.get_cmap('tab20')


@torch.no_grad()
def extract_features(model, loader, device, domain_label):
    """提取 encoder 的 shared 特征，返回 (N, D) numpy 数组。"""
    model.eval()
    feats, labels = [], []

    dummy = torch.zeros(1, 1, 1, 1).to(device)   # 占位，不影响结果

    for x, y in loader:
        x = x.to(device)
        # 只需要 encoder + disentangle 的 shared 输出
        raw = model.encoder(x)                     # (B, T, N, D)
        shared, _ = model.disentangle(raw)
        feat = shared.mean(dim=(1, 2))             # (B, D)
        feats.append(feat.cpu().numpy())
        labels.append(np.full(len(x), domain_label))

    return np.concatenate(feats), np.concatenate(labels)


def run_tsne(feats, perplexity=30, seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=seed, n_iter=1000, init='pca')
    return tsne.fit_transform(feats)


def domain_confusion_score(emb, domain_labels):
    """
    域混淆分数：用轮廓系数衡量域分离程度。
    值越接近 0（甚至负数），说明域特征越混淆 → 对齐越好。
    """
    try:
        score = silhouette_score(emb, domain_labels)
    except Exception:
        score = float('nan')
    return score


def plot_tsne(emb_s, emb_t, title, save_path, action_labels_s=None, action_labels_t=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    all_emb = np.vstack([emb_s, emb_t])
    all_dom = np.array([0] * len(emb_s) + [1] * len(emb_t))

    # ── 左图：按域着色 ───────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(emb_s[:, 0], emb_s[:, 1], c=DOMAIN_COLORS['source'],
               s=8, alpha=0.5, label='Source', rasterized=True)
    ax.scatter(emb_t[:, 0], emb_t[:, 1], c=DOMAIN_COLORS['target'],
               s=8, alpha=0.5, label='Target', rasterized=True)
    score = domain_confusion_score(all_emb, all_dom)
    ax.set_title(f'Domain coloring  (silhouette={score:.3f})', fontsize=11)
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    # ── 右图：按动作类别着色（如果有标签）────────────────────────────────
    ax2 = axes[1]
    if action_labels_s is not None and action_labels_t is not None:
        all_act = np.concatenate([action_labels_s, action_labels_t])
        unique_acts = sorted(set(all_act))
        for i, act in enumerate(unique_acts):
            mask_s = (action_labels_s == act)
            mask_t = (action_labels_t == act)
            c = ACTION_CMAP(i / max(len(unique_acts) - 1, 1))
            ax2.scatter(emb_s[mask_s, 0], emb_s[mask_s, 1],
                        c=[c], s=8, alpha=0.6, marker='o', rasterized=True)
            ax2.scatter(emb_t[mask_t, 0], emb_t[mask_t, 1],
                        c=[c], s=8, alpha=0.6, marker='^', rasterized=True)

        legend_els = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=7, label='Source'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=7, label='Target'),
        ]
        ax2.legend(handles=legend_els, fontsize=9, markerscale=1.5)
        ax2.set_title('Action category coloring', fontsize=11)
    else:
        ax2.text(0.5, 0.5, 'No action labels available',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=11, color='gray')
        ax2.set_title('Action category coloring', fontsize=11)

    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root',  required=True)
    parser.add_argument('--source', nargs='+', default=['E01', 'E02', 'E03'])
    parser.add_argument('--target', nargs='+', default=['E04'])
    parser.add_argument('--save_dir', default='./vis_output')
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='每个域最多采样多少样本（加速 t-SNE）')
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 加载模型 ─────────────────────────────────────────────────────────
    model = WiFiPoseModel(dim=args.dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # ── 数据加载 ─────────────────────────────────────────────────────────
    def make_loader(envs):
        ds = MMFiDataset(args.data_root, envs)
        return DataLoader(ds, batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)

    src_loader = make_loader(args.source)
    tgt_loader = make_loader(args.target)

    print("  Extracting source features...")
    fs, ls = extract_features(model, src_loader, device, domain_label=0)
    print("  Extracting target features...")
    ft, lt = extract_features(model, tgt_loader, device, domain_label=1)

    # 随机降采样加速 t-SNE
    def subsample(arr, labels, n):
        if len(arr) <= n:
            return arr, labels
        idx = np.random.choice(len(arr), n, replace=False)
        return arr[idx], labels[idx]

    fs, ls = subsample(fs, ls, args.n_samples)
    ft, lt = subsample(ft, lt, args.n_samples)

    # ── t-SNE ─────────────────────────────────────────────────────────────
    print("  Running t-SNE...")
    all_feats = np.vstack([fs, ft])
    emb = run_tsne(all_feats)
    emb_s = emb[:len(fs)]
    emb_t = emb[len(fs):]

    score = plot_tsne(
        emb_s, emb_t,
        title='Domain Alignment: t-SNE of Shared Features',
        save_path=os.path.join(args.save_dir, 'tsne_domain_alignment.png'),
    )

    print(f"\n  Domain silhouette score: {score:.4f}")
    print(f"    (closer to 0 or negative = better domain alignment)")


if __name__ == '__main__':
    main()