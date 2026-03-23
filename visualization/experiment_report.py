"""
多实验对比报告生成器
输入多个 checkpoint，自动计算各指标并生成对比表格 + 折线图

用法：
    python -m visualization.experiment_report \
        --checkpoints ./outputs/run1/best.pth ./outputs/run2/best.pth \
        --labels baseline ours \
        --data_root /data/MMFi \
        --target E04 \
        --save_dir ./vis_output
"""

import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from utils_metrics import compute_metrics


COLORS  = ['#2E86AB', '#E84855', '#3BB273', '#F4A261', '#9B59B6']
MARKERS = ['o', 's', '^', 'D', 'v']


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, data_root, envs, dim, seq_len=20,
                         batch_size=32, device='cpu'):
    model = WiFiPoseModel(dim=dim).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds     = MMFiDataset(data_root, envs, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_pred, all_gt = [], []
    for x, y in loader:
        x = x.to(device)
        pred, *_ = model(x, x, alpha=0.0)
        all_pred.append(pred.cpu())
        all_gt.append(y[:, 0].cpu())

    metrics = compute_metrics(all_pred, all_gt)
    return metrics


def plot_metric_bars(results, metric_key, save_path, ylabel=None):
    """绘制单个指标的柱状对比图。"""
    labels = [r['label'] for r in results]
    values = [r['metrics'][metric_key] for r in results]
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=10)

    ax.set_ylabel(ylabel or metric_key, fontsize=11)
    ax.set_title(f'Model comparison: {metric_key}', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 标注最优
    best_idx = np.argmin(values) if 'MPJPE' in metric_key else np.argmax(values)
    bars[best_idx].set_edgecolor('#FFD700')
    bars[best_idx].set_linewidth(2.5)
    ax.annotate('★ best', xy=(best_idx, values[best_idx]),
                xytext=(0, 14), textcoords='offset points',
                ha='center', fontsize=9, color='#B8860B')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comprehensive_comparison(results, save_path):
    """4 合 1 综合对比图。"""
    metrics_to_plot = ['MPJPE', 'PA-MPJPE', 'PCK@0.05']
    n_metrics = len(metrics_to_plot)
    labels = [r['label'] for r in results]

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, n_metrics + 1, width_ratios=[*([1]*n_metrics), 0.6],
                            wspace=0.35)

    for mi, metric in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[mi])
        values = [r['metrics'].get(metric, 0) for r in results]
        colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
        bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='white')
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=20, labelsize=9)

        best_idx = np.argmin(values) if 'MPJPE' in metric else np.argmax(values)
        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(2.5)

    # 汇总表格
    ax_tbl = fig.add_subplot(gs[-1])
    ax_tbl.axis('off')

    col_labels = ['Model'] + metrics_to_plot
    table_data = []
    for r in results:
        row = [r['label']]
        for m in metrics_to_plot:
            val = r['metrics'].get(m, float('nan'))
            row.append(f'{val:.4f}')
        table_data.append(row)

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # 高亮最优
    for mi, metric in enumerate(metrics_to_plot, start=1):
        vals = [r['metrics'].get(metric, float('nan')) for r in results]
        best = int(np.argmin(vals) if 'MPJPE' in metric else np.argmax(vals))
        tbl[best + 1, mi].set_facecolor('#FFF3CD')

    ax_tbl.set_title('Summary', fontsize=11, fontweight='bold', pad=10)

    fig.suptitle('Experiment Comparison Report', fontsize=15, fontweight='bold', y=1.03)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def save_json_report(results, save_path):
    report = []
    for r in results:
        report.append({
            'label':      r['label'],
            'checkpoint': r['checkpoint'],
            'metrics':    {k: float(v) for k, v in r['metrics'].items()},
        })
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {save_path}")


def print_table(results):
    metrics = list(results[0]['metrics'].keys())
    col_w   = max(16, max(len(r['label']) for r in results) + 2)
    header  = f"{'Model':<{col_w}}" + ''.join(f"{m:>14}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['label']:<{col_w}}"
        for m in metrics:
            row += f"{r['metrics'][m]:>14.4f}"
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--labels',      nargs='+', default=None)
    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--target',      nargs='+', default=['E04'])
    parser.add_argument('--save_dir',    default='./vis_output')
    parser.add_argument('--dim',         type=int, default=64)
    parser.add_argument('--seq_len',     type=int, default=20)
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = []
    for ckpt, label in zip(args.checkpoints, args.labels):
        print(f"\n  Evaluating: {label}  ({ckpt})")
        metrics = evaluate_checkpoint(
            ckpt, args.data_root, args.target,
            dim=args.dim, seq_len=args.seq_len, device=device
        )
        results.append({'label': label, 'checkpoint': ckpt, 'metrics': metrics})
        print(f"    {metrics}")

    print_table(results)

    # 各指标单独柱状图
    for metric in ['MPJPE', 'PA-MPJPE', 'PCK@0.05']:
        plot_metric_bars(
            results, metric,
            os.path.join(args.save_dir, f'compare_{metric.replace("@","_").replace("-","_")}.png'),
            ylabel=metric,
        )

    # 综合对比图
    plot_comprehensive_comparison(
        results,
        os.path.join(args.save_dir, 'experiment_comparison.png'),
    )

    # JSON 报告
    save_json_report(results, os.path.join(args.save_dir, 'report.json'))


if __name__ == '__main__':
    main()