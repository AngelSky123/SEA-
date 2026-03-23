"""
训练曲线可视化
用法：
    python -m visualization.plot_training_curves --log_dir ./outputs/20240101_120000
    python -m visualization.plot_training_curves --log_dir ./outputs/run1 ./outputs/run2 --labels baseline improved
"""

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 样式 ────────────────────────────────────────────────────────────────
COLORS = ['#2E86AB', '#E84855', '#3BB273', '#F4A261', '#9B59B6']
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def parse_log_file(log_path):
    """从训练日志文本中解析各损失曲线。"""
    data = {k: [] for k in ('epoch', 'total', 'pose', 'align', 'domain', 'orth', 'lr')}

    epoch_pattern  = re.compile(r'Epoch\s+(\d+)\s+done.*?avg_loss=([\d.]+)')
    comp_pattern   = re.compile(r'pose=([\d.]+).*?align=([\d.]+).*?domain=([\d.]+).*?orth=([\d.]+)')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            em = epoch_pattern.search(line)
            if em:
                data['epoch'].append(int(em.group(1)))
                data['total'].append(float(em.group(2)))
                cm = comp_pattern.search(line)
                if cm:
                    data['pose'].append(float(cm.group(1)))
                    data['align'].append(float(cm.group(2)))
                    data['domain'].append(float(cm.group(3)))
                    data['orth'].append(float(cm.group(4)))

    return {k: np.array(v) for k, v in data.items() if len(v) > 0}


def smooth(arr, window=5):
    """简单移动平均平滑。"""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def plot_curves(log_dirs, labels=None, save_path='training_curves.png'):
    if labels is None:
        labels = [os.path.basename(d) for d in log_dirs]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    axes = {
        'total':  fig.add_subplot(gs[0, :2]),
        'pose':   fig.add_subplot(gs[0, 2]),
        'align':  fig.add_subplot(gs[1, 0]),
        'domain': fig.add_subplot(gs[1, 1]),
        'orth':   fig.add_subplot(gs[1, 2]),
    }
    titles = {
        'total':  'Total loss',
        'pose':   'Pose loss (MSE)',
        'align':  'Align loss',
        'domain': 'Domain loss',
        'orth':   'Orthogonality loss',
    }

    for ax_key, ax in axes.items():
        ax.set_title(titles[ax_key], fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    all_parsed = []
    for d in log_dirs:
        log_file = os.path.join(d, 'train.log')
        if not os.path.exists(log_file):
            # 尝试在目录内搜索 .log 文件
            candidates = [f for f in os.listdir(d) if f.endswith('.log')]
            log_file = os.path.join(d, candidates[0]) if candidates else None
        parsed = parse_log_file(log_file) if log_file and os.path.exists(log_file) else {}
        all_parsed.append(parsed)

    for idx, (parsed, label) in enumerate(zip(all_parsed, labels)):
        color = COLORS[idx % len(COLORS)]
        epochs = parsed.get('epoch', np.array([]))
        if len(epochs) == 0:
            print(f"  [Warning] no epoch data found for: {log_dirs[idx]}")
            continue

        for key in ('total', 'pose', 'align', 'domain', 'orth'):
            vals = parsed.get(key)
            if vals is None or len(vals) == 0:
                continue
            ax = axes[key]
            # 原始曲线（淡色）
            ax.plot(epochs, vals, color=color, alpha=0.25, linewidth=0.8)
            # 平滑曲线
            ax.plot(epochs, smooth(vals), color=color, linewidth=2,
                    label=label)
            # 标注最终值
            ax.annotate(f'{vals[-1]:.4f}', xy=(epochs[-1], smooth(vals)[-1]),
                        fontsize=8, color=color,
                        xytext=(4, 0), textcoords='offset points')

    for ax in axes.values():
        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9, framealpha=0.7)

    fig.suptitle('Training Curve Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', nargs='+', required=True,
                        help='一个或多个实验输出目录（含 train.log）')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='各实验的图例名称（与 log_dir 一一对应）')
    parser.add_argument('--save', default='training_curves.png')
    args = parser.parse_args()

    plot_curves(args.log_dir, args.labels, args.save)