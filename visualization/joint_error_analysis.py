"""
逐关节误差分析：热力图 + 雷达图 + 身体部位分组统计

用法：
    python -m visualization.joint_error_analysis \
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel


# ── H36M 17 关节定义 ─────────────────────────────────────────────────────
JOINT_NAMES = [
    "Hip",        # 0
    "R-Hip",      # 1
    "R-Knee",     # 2
    "R-Ankle",    # 3
    "L-Hip",      # 4
    "L-Knee",     # 5
    "L-Ankle",    # 6
    "Spine",      # 7
    "Thorax",     # 8
    "Neck/Nose",  # 9
    "Head",       # 10
    "L-Shoulder", # 11
    "L-Elbow",    # 12
    "L-Wrist",    # 13
    "R-Shoulder", # 14
    "R-Elbow",    # 15
    "R-Wrist",    # 16
]

BODY_GROUPS = {
    "Torso": [0, 7, 8],
    "Head":  [9, 10],
    "L-Arm": [11, 12, 13],
    "R-Arm": [14, 15, 16],
    "L-Leg": [4, 5, 6],
    "R-Leg": [1, 2, 3],
}

SKELETON_EDGES = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]


@torch.no_grad()
def collect_errors(model, loader, device):
    """
    返回逐关节误差矩阵 (N, J)。

    修复：DataLoader 返回三元组，正确解包（原版 for x, y in loader 会报错）。
    """
    model.eval()
    all_errors = []

    for x, y, _ in loader:   # 修复：三元组解包
        x   = x.to(device)
        pred, *_ = model(x, x, alpha=0.0)   # (B, J, 3)
        gt  = y[:, 0].to(device)            # (B, J, 3)，取第 0 帧
        err = torch.norm(pred - gt, dim=-1) # (B, J)
        all_errors.append(err.cpu().numpy())

    return np.concatenate(all_errors, axis=0)   # (N, J)


def plot_joint_heatmap(errors_mean, save_path):
    """绘制 17 关节误差热力图（关节 × 1 条形）。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_idx   = np.argsort(errors_mean)[::-1]
    sorted_err   = errors_mean[sorted_idx]
    sorted_names = [JOINT_NAMES[i] for i in sorted_idx]

    cmap   = plt.get_cmap("RdYlGn_r")
    norm   = mcolors.Normalize(vmin=sorted_err.min(), vmax=sorted_err.max())
    colors = [cmap(norm(v)) for v in sorted_err]

    bars = ax.barh(sorted_names, sorted_err, color=colors, height=0.7)
    ax.bar_label(bars, fmt="%.1f mm", padding=4, fontsize=9)

    ax.set_xlabel("Mean per-joint position error (mm)", fontsize=11)
    ax.set_title("Per-joint MPJPE (sorted)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Error (mm)", shrink=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_radar(group_errors, save_path):
    """绘制身体部位误差雷达图。"""
    groups = list(group_errors.keys())
    values = [group_errors[g] for g in groups]
    N      = len(groups)

    angles       = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot  = values + [values[0]]
    angles      += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values_plot, "o-", linewidth=2, color="#2E86AB")
    ax.fill(angles, values_plot, alpha=0.2, color="#2E86AB")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups, size=11)
    ax.set_title("Body-part grouped MPJPE (mm)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.grid(color="gray", linestyle="--", alpha=0.4)

    for angle, val in zip(angles[:-1], values):
        ax.annotate(f"{val:.1f}", xy=(angle, val),
                    fontsize=9, ha="center", color="#1a1a2e",
                    xytext=(0, 10), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_skeleton_error(errors_mean, save_path):
    """在骨架图上用颜色编码各关节误差（2D 投影）。"""
    REF_POS = np.array([
        [ 0.0,  0.0],   # 0 Hip
        [ 0.2, -0.5],   # 1 R-Hip
        [ 0.2, -1.1],   # 2 R-Knee
        [ 0.2, -1.7],   # 3 R-Ankle
        [-0.2, -0.5],   # 4 L-Hip
        [-0.2, -1.1],   # 5 L-Knee
        [-0.2, -1.7],   # 6 L-Ankle
        [ 0.0,  0.5],   # 7 Spine
        [ 0.0,  1.0],   # 8 Thorax
        [ 0.0,  1.4],   # 9 Neck/Nose
        [ 0.0,  1.7],   # 10 Head
        [-0.4,  1.0],   # 11 L-Shoulder
        [-0.7,  0.5],   # 12 L-Elbow
        [-0.9,  0.0],   # 13 L-Wrist
        [ 0.4,  1.0],   # 14 R-Shoulder
        [ 0.7,  0.5],   # 15 R-Elbow
        [ 0.9,  0.0],   # 16 R-Wrist
    ])

    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Skeleton error heatmap", fontsize=13, fontweight="bold")

    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.Normalize(vmin=errors_mean.min(), vmax=errors_mean.max())

    for i, j in SKELETON_EDGES:
        x = [REF_POS[i, 0], REF_POS[j, 0]]
        y = [REF_POS[i, 1], REF_POS[j, 1]]
        ax.plot(x, y, color="#cccccc", linewidth=2, zorder=1)

    sc = ax.scatter(REF_POS[:, 0], REF_POS[:, 1],
                    c=errors_mean, cmap=cmap, norm=norm,
                    s=200, zorder=2, edgecolors="white", linewidths=1.2)

    for i, (x, y) in enumerate(REF_POS):
        ax.annotate(JOINT_NAMES[i], xy=(x, y),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, color="#333333")

    plt.colorbar(sc, ax=ax, label="Error (mm)",
                 orientation="horizontal", shrink=0.7, pad=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary(errors_mean, group_errors):
    print("\n" + "=" * 45)
    print(f"{'Joint':<16} {'Error (mm)':>12}")
    print("-" * 45)
    for name, err in zip(JOINT_NAMES, errors_mean):
        print(f"{name:<16} {err:>12.2f}")
    print("=" * 45)
    print(f"{'Overall MPJPE':<16} {errors_mean.mean():>12.2f}")
    print("-" * 45)
    print("\nBy body part:")
    for grp, val in group_errors.items():
        print(f"  {grp:<12} {val:.2f} mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--target",     nargs="+", default=["E04"])
    parser.add_argument("--save_dir",   default="./vis_output")
    # 修复：dim 默认值与 default.yaml 保持一致
    parser.add_argument("--dim",        type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WiFiPoseModel(dim=args.dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    ds     = MMFiDataset(args.data_root, args.target)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    print("  Collecting per-joint errors...")
    errors     = collect_errors(model, loader, device)   # (N, J)
    errors_mm  = errors * 1000.0
    errors_mean = errors_mm.mean(axis=0)                 # (J,)

    group_errors = {
        grp: errors_mm[:, idx].mean()
        for grp, idx in BODY_GROUPS.items()
    }

    print_summary(errors_mean, group_errors)

    plot_joint_heatmap(errors_mean,
        os.path.join(args.save_dir, "joint_error_heatmap.png"))
    plot_radar(group_errors,
        os.path.join(args.save_dir, "body_part_radar.png"))
    plot_skeleton_error(errors_mean,
        os.path.join(args.save_dir, "skeleton_error_heatmap.png"))


if __name__ == "__main__":
    main()