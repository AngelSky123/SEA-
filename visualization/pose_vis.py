"""
单帧 3D 姿态可视化（升级版）
新增：多帧序列对比、误差向量场、关节误差着色
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


EDGES = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16),
]

JOINT_NAMES = [
    'Hip','R-Hip','R-Knee','R-Ankle',
    'L-Hip','L-Knee','L-Ankle',
    'Spine','Thorax','Neck','Head',
    'L-Shldr','L-Elbow','L-Wrist',
    'R-Shldr','R-Elbow','R-Wrist',
]


def _convert_coords(pose):
    """物理坐标 → 绘图坐标（修正朝向）。"""
    p = np.zeros_like(pose)
    p[:, 0] =  pose[:, 0]
    p[:, 1] =  pose[:, 2]
    p[:, 2] = -pose[:, 1]
    return p


def _set_equal_axes(ax, points):
    max_range = np.ptp(points, axis=0).max() / 2.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)


def plot_pose(gt, pred, save_path='single_pose.png'):
    """标准单帧姿态对比图（GT vs Pred）。"""
    gt_p   = _convert_coords(gt)
    pred_p = _convert_coords(pred)

    # 计算逐关节误差用于着色
    errors = np.linalg.norm(pred - gt, axis=-1)
    cmap   = plt.get_cmap('RdYlGn_r')
    norm   = mcolors.Normalize(vmin=0, vmax=errors.max() + 1e-8)

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.05)

    for col, (pose_data, title, color, linestyle) in enumerate([
        (gt_p,   'Ground Truth', '#27AE60', '-'),
        (pred_p, 'Prediction',   '#E74C3C', '--'),
    ]):
        ax = fig.add_subplot(gs[col], projection='3d')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # 骨骼连线
        for i, j in EDGES:
            ax.plot([pose_data[i,0], pose_data[j,0]],
                    [pose_data[i,1], pose_data[j,1]],
                    [pose_data[i,2], pose_data[j,2]],
                    c=color, alpha=0.7, linewidth=2, linestyle=linestyle)

        # 关节点（Pred 图用误差着色）
        if col == 1:
            sc = ax.scatter(pose_data[:,0], pose_data[:,1], pose_data[:,2],
                            c=errors, cmap=cmap, norm=norm, s=50, zorder=5)
        else:
            ax.scatter(pose_data[:,0], pose_data[:,1], pose_data[:,2],
                       c=color, s=50, zorder=5)

        _set_equal_axes(ax, np.vstack([gt_p, pred_p]))
        ax.set_xlabel('X'); ax.set_ylabel('Y (depth)'); ax.set_zlabel('Z (up)')
        ax.view_init(elev=15, azim=45)
        ax.tick_params(labelsize=7)

    # 色条（右侧）
    ax_cb = fig.add_subplot(gs[2])
    ax_cb.axis('off')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cb, fraction=0.8, pad=0.05)
    cbar.set_label('Joint error (m)', fontsize=9)

    # MPJPE 标注
    mpjpe = errors.mean()
    fig.text(0.5, 0.01, f'MPJPE = {mpjpe*1000:.1f} mm',
             ha='center', fontsize=11, color='#555555')

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pose_sequence(gt_seq, pred_seq, save_path='pose_sequence.png',
                       stride=5, max_frames=6):
    """
    绘制一段动作序列中均匀抽取的多帧姿态对比。
    gt_seq / pred_seq : (T, J, 3)
    """
    T = min(len(gt_seq), len(pred_seq))
    frame_idx = np.linspace(0, T - 1, min(max_frames, T), dtype=int)
    n = len(frame_idx)

    fig = plt.figure(figsize=(3 * n, 6))
    gs  = gridspec.GridSpec(1, n, wspace=0.02)
    fig.suptitle('Pose sequence: GT (green) vs Pred (red)', fontsize=13,
                 fontweight='bold', y=1.02)

    for col, fi in enumerate(frame_idx):
        ax = fig.add_subplot(gs[col], projection='3d')
        ax.set_title(f'Frame {fi}', fontsize=9)

        gt_p   = _convert_coords(gt_seq[fi])
        pred_p = _convert_coords(pred_seq[fi])

        for pose_data, color, ls in [(gt_p, '#27AE60', '-'), (pred_p, '#E74C3C', '--')]:
            for i, j in EDGES:
                ax.plot([pose_data[i,0], pose_data[j,0]],
                        [pose_data[i,1], pose_data[j,1]],
                        [pose_data[i,2], pose_data[j,2]],
                        c=color, alpha=0.7, linewidth=1.5, linestyle=ls)
            ax.scatter(pose_data[:,0], pose_data[:,1], pose_data[:,2],
                       c=color, s=20, zorder=5)

        _set_equal_axes(ax, np.vstack([gt_p, pred_p]))
        ax.view_init(elev=15, azim=45)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_error_vector_field(gt, pred, save_path='error_vector_field.png'):
    """
    绘制关节误差向量场（2D 投影）：
    显示每个关节的预测偏移方向和大小。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Joint error vector field', fontsize=13, fontweight='bold')

    views = [
        ('Front view',  (0, 2)),   # X-Z
        ('Side view',   (1, 2)),   # Y-Z
    ]

    errors = np.linalg.norm(pred - gt, axis=-1)
    cmap   = plt.get_cmap('RdYlGn_r')
    norm   = mcolors.Normalize(vmin=0, vmax=errors.max() + 1e-8)

    for ax, (title, (xi, yi)) in zip(axes, views):
        # GT 骨骼（绿色）
        for i, j in EDGES:
            ax.plot([gt[i, xi], gt[j, xi]], [gt[i, yi], gt[j, yi]],
                    c='#27AE60', alpha=0.5, linewidth=1.5)

        # 误差向量（箭头）
        for k in range(len(gt)):
            dx = pred[k, xi] - gt[k, xi]
            dy = pred[k, yi] - gt[k, yi]
            c  = cmap(norm(errors[k]))
            ax.annotate('', xy=(pred[k, xi], pred[k, yi]),
                        xytext=(gt[k, xi], gt[k, yi]),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
            ax.scatter(gt[k, xi], gt[k, yi], c=[c], s=40, zorder=5)

        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.2)
        ax.set_xlabel(['X', 'Y'][views.index((title, (xi, yi)))])
        ax.set_ylabel('Z (up)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='Joint error (m)',
                 orientation='vertical', shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")