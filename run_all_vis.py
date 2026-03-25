"""
一键运行所有可视化分析脚本

用法：
    python run_all_vis.py \
        --checkpoint ./outputs/20240101_120000/best.pth \
        --data_root /home/a123456/SEA-/MMFi \
        --save_dir ./vis_output \
        [--source E01 E02 E03] \
        [--target E04] \
        [--dim 64]

生成内容：
    vis_output/
    ├── tsne_domain_alignment.png     # 域对齐 t-SNE
    ├── joint_error_heatmap.png       # 逐关节误差条形图
    ├── body_part_radar.png           # 身体部位雷达图
    ├── skeleton_error_heatmap.png    # 骨架误差热力图
    ├── sensor_attention_samples.png  # 传感器注意力（样本）
    ├── sensor_attention_mean.png     # 传感器注意力（均值）
    ├── attention_statistics.png      # 注意力统计分析
    └── single_pose_sample.png        # 单帧姿态可视化示例
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from visualization.pose_vis import plot_pose, plot_pose_sequence, plot_error_vector_field
from visualization.tsne_analysis import extract_features, run_tsne, plot_tsne
from visualization.joint_error_analysis import (
    collect_errors, plot_joint_heatmap, plot_radar,
    plot_skeleton_error, print_summary, BODY_GROUPS
)
from visualization.attention_vis import (
    extract_attention_weights, plot_sensor_attention,
    plot_mean_attention_map, plot_attention_stats
)


def banner(text):
    print(f"\n{'─'*50}")
    print(f"  {text}")
    print(f"{'─'*50}")


def main():
    parser = argparse.ArgumentParser(description='一键可视化分析')
    parser.add_argument('--checkpoint', required=True, help='模型权重路径')
    parser.add_argument('--data_root',  required=True, help='MMFi 数据根目录')
    parser.add_argument('--save_dir',   default='./vis_output')
    parser.add_argument('--source', nargs='+', default=['E01', 'E02', 'E03'])
    parser.add_argument('--target', nargs='+', default=['E04'])
    parser.add_argument('--dim',    type=int, default=64)
    parser.add_argument('--seq_len',type=int, default=20)
    parser.add_argument('--n_tsne_samples', type=int, default=2000,
                        help='t-SNE 降采样数')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # ── 加载模型 ─────────────────────────────────────────────────────────
    banner("Loading model")
    model = WiFiPoseModel(dim=args.dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Loaded: {args.checkpoint}")
    epoch = ckpt.get('epoch', '?')
    loss  = ckpt.get('loss',  '?')
    print(f"  Checkpoint epoch={epoch}, loss={loss}")

    # ── 数据集 ────────────────────────────────────────────────────────────
    tgt_ds     = MMFiDataset(args.data_root, args.target, args.seq_len)
    src_ds     = MMFiDataset(args.data_root, args.source, args.seq_len)
    tgt_loader = DataLoader(tgt_ds, batch_size=32, shuffle=False, num_workers=4)
    src_loader = DataLoader(src_ds, batch_size=32, shuffle=False, num_workers=4)

    # ──────────────────────────────────────────────────────────────────────
    # 1. 单帧姿态可视化
    # ──────────────────────────────────────────────────────────────────────
    banner("1/4  Pose visualization")
    with torch.no_grad():
        x_sample, y_sample = tgt_ds[0]
        x_in = x_sample.unsqueeze(0).to(device)
        pred, *_ = model(x_in, x_in, alpha=0.0)

    gt_np   = y_sample[0].numpy()
    pred_np = pred[0].cpu().numpy()

    plot_pose(gt_np, pred_np,
              save_path=os.path.join(args.save_dir, 'single_pose_sample.png'))
    plot_error_vector_field(gt_np, pred_np,
              save_path=os.path.join(args.save_dir, 'error_vector_field.png'))

    # 序列可视化（取前 30 帧）
    seq_len = min(30, len(tgt_ds))
    gt_seq, pred_seq = [], []
    with torch.no_grad():
        for fi in range(seq_len):
            xf, yf = tgt_ds[fi]
            xf_in  = xf.unsqueeze(0).to(device)
            pf, *_ = model(xf_in, xf_in, alpha=0.0)
            gt_seq.append(yf[0].numpy())
            pred_seq.append(pf[0].cpu().numpy())

    plot_pose_sequence(
        np.stack(gt_seq), np.stack(pred_seq),
        save_path=os.path.join(args.save_dir, 'pose_sequence.png'),
    )

    # ──────────────────────────────────────────────────────────────────────
    # 2. 逐关节误差分析
    # ──────────────────────────────────────────────────────────────────────
    banner("2/4  Joint error analysis")
    errors    = collect_errors(model, tgt_loader, device)
    errors_mm = errors * 1000.0
    errors_mean = errors_mm.mean(axis=0)
    group_errors = {grp: errors_mm[:, idx].mean()
                    for grp, idx in BODY_GROUPS.items()}
    print_summary(errors_mean, group_errors)

    plot_joint_heatmap(errors_mean,
        os.path.join(args.save_dir, 'joint_error_heatmap.png'))
    plot_radar(group_errors,
        os.path.join(args.save_dir, 'body_part_radar.png'))
    plot_skeleton_error(errors_mean,
        os.path.join(args.save_dir, 'skeleton_error_heatmap.png'))

    # ──────────────────────────────────────────────────────────────────────
    # 3. t-SNE 域对齐分析
    # ──────────────────────────────────────────────────────────────────────
    banner("3/4  t-SNE domain alignment")
    print("  Extracting source features...")
    fs, ls = extract_features(model, src_loader, device, domain_label=0)
    print("  Extracting target features...")
    ft, lt = extract_features(model, tgt_loader, device, domain_label=1)

    def subsample(arr, labels, n):
        if len(arr) <= n: return arr, labels
        idx = np.random.choice(len(arr), n, replace=False)
        return arr[idx], labels[idx]

    fs, ls = subsample(fs, ls, args.n_tsne_samples)
    ft, lt = subsample(ft, lt, args.n_tsne_samples)

    print("  Running t-SNE (this may take a minute)...")
    emb   = run_tsne(np.vstack([fs, ft]))
    emb_s = emb[:len(fs)]
    emb_t = emb[len(fs):]

    score = plot_tsne(emb_s, emb_t,
        title='Domain Alignment: Shared Feature t-SNE',
        save_path=os.path.join(args.save_dir, 'tsne_domain_alignment.png'))
    print(f"  Domain silhouette score: {score:.4f}  (closer to 0 = better alignment)")

    # ──────────────────────────────────────────────────────────────────────
    # 4. 注意力权重分析
    # ──────────────────────────────────────────────────────────────────────
    banner("4/4  Attention weight analysis")
    all_attn = []
    for i, (x, _) in enumerate(tgt_loader):
        if i >= 5: break
        w = extract_attention_weights(model, x, device)
        if 'sensor' in w:
            all_attn.append(w['sensor'])

    if all_attn:
        all_attn = np.concatenate(all_attn, axis=0)
        plot_sensor_attention(all_attn,
            os.path.join(args.save_dir, 'sensor_attention_samples.png'))
        plot_mean_attention_map(all_attn,
            os.path.join(args.save_dir, 'sensor_attention_mean.png'))
        plot_attention_stats(all_attn,
            os.path.join(args.save_dir, 'attention_statistics.png'))
    else:
        print("  [Skip] No attention weights captured.")

    # ──────────────────────────────────────────────────────────────────────
    banner("Done")
    print(f"  All outputs saved to: {os.path.abspath(args.save_dir)}")
    files = sorted(os.listdir(args.save_dir))
    for f in files:
        size = os.path.getsize(os.path.join(args.save_dir, f))
        print(f"    {f:<45} {size//1024:>5} KB")


if __name__ == '__main__':
    main()