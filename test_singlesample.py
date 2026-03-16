import os
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import SEAplusplus

def load_single_sample(csi_dir, pose_path, L=297):
    pose_array = np.load(pose_path)
    if pose_array.ndim == 3: 
        pose_gt = np.mean(pose_array, axis=0)
    else:
        pose_gt = pose_array

    csi_files = sorted([f for f in os.listdir(csi_dir) if f.endswith('.mat')])
    sequence_frames = []
    
    for f in csi_files:
        mat_dict = sio.loadmat(os.path.join(csi_dir, f))
        data_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
        csi_keys = [k for k in data_keys if 'csi' in k.lower() or 'data' in k.lower()]
        valid_key = csi_keys[0] if len(csi_keys) > 0 else data_keys[0]
        
        csi = mat_dict[valid_key]
        amp = np.abs(csi)  
        
        if amp.size == 3420:
            frame_feature = amp.reshape(342, -1).mean(axis=1)
        else:
            frame_feature = amp.flatten()[:342]
            
        sequence_frames.append(frame_feature)
        
    seq_amp = np.stack(sequence_frames, axis=1) 
    
    L_current = seq_amp.shape[1]
    if L_current > L:
        seq_amp = seq_amp[:, :L]
    elif L_current < L:
        pad_width = ((0, 0), (0, L - L_current))
        seq_amp = np.pad(seq_amp, pad_width, mode='constant', constant_values=0)
        
    seq_amp = np.nan_to_num(seq_amp, nan=0.0, posinf=0.0, neginf=0.0)
    pose_gt = np.nan_to_num(pose_gt, nan=0.0, posinf=0.0, neginf=0.0)
    
    seq_amp = seq_amp.astype(np.float32)
    # 【与训练完全对齐】：抛弃 Z-score，采用物理级压缩
    seq_amp = np.log1p(seq_amp) / 3.0
    
    return seq_amp, pose_gt

def plot_3d_pose(pose_gt, pose_pred, title_info):
    edges = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],
             [9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    def convert_coords(pose):
        plot_pose = np.zeros_like(pose)
        plot_pose[:, 0] = pose[:, 0]    
        plot_pose[:, 1] = pose[:, 2]    
        plot_pose[:, 2] = -pose[:, 1]   
        return plot_pose

    pose_gt_plot = convert_coords(pose_gt)
    pose_pred_plot = convert_coords(pose_pred)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pose_gt_plot[:, 0], pose_gt_plot[:, 1], pose_gt_plot[:, 2], c='green', s=40, label='Ground Truth')
    for i, j in edges:
        ax.plot([pose_gt_plot[i, 0], pose_gt_plot[j, 0]], 
                [pose_gt_plot[i, 1], pose_gt_plot[j, 1]], 
                [pose_gt_plot[i, 2], pose_gt_plot[j, 2]], c='green', alpha=0.6)

    ax.scatter(pose_pred_plot[:, 0], pose_pred_plot[:, 1], pose_pred_plot[:, 2], c='red', s=40, marker='^', label='Prediction')
    for i, j in edges:
        ax.plot([pose_pred_plot[i, 0], pose_pred_plot[j, 0]], 
                [pose_pred_plot[i, 1], pose_pred_plot[j, 1]], 
                [pose_pred_plot[i, 2], pose_pred_plot[j, 2]], c='red', alpha=0.6, linestyle='--')

    ax.set_title(title_info, fontsize=14)
    ax.set_xlabel('X (Right/Left)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Up/Down)')
    
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
    plt.savefig('pose_visualization.png', dpi=300)
    print("✨ 可视化结果已保存为 'pose_visualization.png'")

def main():
    sample_dir = "/home/a123456/SEA-/MMFi/E04/S31/A12"
    csi_dir = os.path.join(sample_dir, "wifi-csi")
    pose_path = os.path.join(sample_dir, "ground_truth.npy")
    model_path = "sea_model.pth" 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    print("[*] 正在初始化并加载模型权重...")
    model = SEAplusplus(num_sensors=342, d_patch=11, d_model=128, num_branches=3, num_joints=17)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    seq_amp, pose_gt = load_single_sample(csi_dir, pose_path)
    x_tensor = torch.from_numpy(seq_amp).unsqueeze(0).to(device)

    print("[*] 正在进行 3D 姿态推理...")
    with torch.no_grad():
        pose_pred = model(x_tensor, train=False)
        pose_pred_np = pose_pred.squeeze(0).cpu().numpy()
        
        # 抛弃一切对齐，严格计算原始物理空间误差！
        distances_abs = np.linalg.norm(pose_pred_np - pose_gt, axis=-1)
        mpjpe_abs = np.mean(distances_abs) * 1000
        pck_50_abs = np.mean(distances_abs < 0.05) * 100
        pck_20_abs = np.mean(distances_abs < 0.02) * 100

    print("\n" + "="*45)
    print(f" 🎯 单样本 ({sample_dir.split('/')[-1]}) 真实物理空间测试结果:")
    print("-" * 45)
    print(f" 1. 绝对 MPJPE (最严格真实误差): {mpjpe_abs:.2f} mm")
    print(f" 2. 绝对 PCK@50 (<50mm 准确率) : {pck_50_abs:.2f} %")
    print(f" 3. 绝对 PCK@20 (<20mm 准确率) : {pck_20_abs:.2f} %")
    print("="*45 + "\n")

    title_info = f"Absolute 3D Pose | Raw MPJPE: {mpjpe_abs:.1f}mm"
    plot_3d_pose(pose_gt, pose_pred_np, title_info)

if __name__ == "__main__":
    main()