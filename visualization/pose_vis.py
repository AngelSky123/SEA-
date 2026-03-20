import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_pose(gt, pred, save_path="single_pose.png"):
    # 1. 严格使用你代码中提供的正确拓扑连线
    edges = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],
             [9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    # 2. 严格使用你代码中的坐标翻转与映射逻辑 (解决倒立和躺平)
    def convert_coords(pose):
        plot_pose = np.zeros_like(pose)
        plot_pose[:, 0] = pose[:, 0]     # X 保持不变
        plot_pose[:, 1] = pose[:, 2]     # 物理 Z (深度) 变成画板 Y
        plot_pose[:, 2] = -pose[:, 1]    # 物理 Y (高度) 加负号翻转，变成画板 Z
        return plot_pose

    pose_gt_plot = convert_coords(gt)
    pose_pred_plot = convert_coords(pred)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. 绘制 Ground Truth (参考你的代码，使用绿色实线)
    ax.scatter(pose_gt_plot[:, 0], pose_gt_plot[:, 1], pose_gt_plot[:, 2], c='green', s=40, label='Ground Truth')
    for i, j in edges:
        ax.plot([pose_gt_plot[i, 0], pose_gt_plot[j, 0]], 
                [pose_gt_plot[i, 1], pose_gt_plot[j, 1]], 
                [pose_gt_plot[i, 2], pose_gt_plot[j, 2]], c='green', alpha=0.6)

    # 4. 绘制 Prediction (参考你的代码，使用红色虚线和三角标记)
    ax.scatter(pose_pred_plot[:, 0], pose_pred_plot[:, 1], pose_pred_plot[:, 2], c='red', s=40, marker='^', label='Prediction')
    for i, j in edges:
        ax.plot([pose_pred_plot[i, 0], pose_pred_plot[j, 0]], 
                [pose_pred_plot[i, 1], pose_pred_plot[j, 1]], 
                [pose_pred_plot[i, 2], pose_pred_plot[j, 2]], c='red', alpha=0.6, linestyle='--')

    ax.set_title("3D Pose Visualization", fontsize=14)
    ax.set_xlabel('X (Right/Left)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Up/Down)')
    
    # 5. 绝对等比例锁定 (你的原版逻辑)
    all_points = np.vstack([pose_gt_plot, pose_pred_plot])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=15, azim=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()