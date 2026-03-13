import os
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from model import SEAplusplus

def load_single_sample(csi_dir, pose_path, L=297):
    # 1. 加载真实的 3D Pose
    pose_array = np.load(pose_path)
    if pose_array.ndim == 3: 
        pose_gt = np.mean(pose_array, axis=0)
    else:
        pose_gt = pose_array

    # 2. 加载 CSI 序列
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
    
    # 3. 长度对齐
    L_current = seq_amp.shape[1]
    if L_current > L:
        seq_amp = seq_amp[:, :L]
    elif L_current < L:
        pad_width = ((0, 0), (0, L - L_current))
        seq_amp = np.pad(seq_amp, pad_width, mode='constant', constant_values=0)
        
    # 4. 数据清洗与 Z-score 归一化
    seq_amp = np.nan_to_num(seq_amp, nan=0.0, posinf=0.0, neginf=0.0)
    pose_gt = np.nan_to_num(pose_gt, nan=0.0, posinf=0.0, neginf=0.0)
    
    seq_amp = seq_amp.astype(np.float32)
    seq_amp = (seq_amp - np.mean(seq_amp)) / (np.std(seq_amp) + 1e-5)
    
    return seq_amp, pose_gt

def compute_similarity_transform(S1, S2):
    """普氏分析 (Procrustes Analysis) 核心算法"""
    trans1 = S1.mean(axis=0)
    trans2 = S2.mean(axis=0)
    S1 = S1 - trans1
    S2 = S2 - trans2
    scale1 = np.linalg.norm(S1)
    scale2 = np.linalg.norm(S2)
    
    if scale1 == 0 or scale2 == 0:
        return S1, S2
        
    S1 = S1 / scale1
    S2 = S2 / scale2
    
    U, s, Vh = np.linalg.svd(np.dot(S1.T, S2))
    R = np.dot(U, Vh)
    
    if np.linalg.det(R) < 0:
        Vh[-1] *= -1
        s[-1] *= -1
        R = np.dot(U, Vh)
        
    s_opt = sum(s) * (scale2 / scale1)
    S1_opt = S1 * scale1 * s_opt
    S1_opt = np.dot(S1_opt, R) + trans2
    S2_opt = S2 * scale2 + trans2
    
    return S1_opt, S2_opt

def plot_3d_pose(pose_gt, pose_pred, title_info):
    edges = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],
             [9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    # 坐标系转换 (Camera -> Matplotlib)
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
    sample_dir = "/home/a123456/SEA-/MMFi/E04/S34/A26"
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

    print(f"[*] 正在处理样本: {sample_dir} ...")
    if not os.path.exists(csi_dir) or not os.path.exists(pose_path):
        print(" 错误: 找不到对应的数据！")
        return

    seq_amp, pose_gt = load_single_sample(csi_dir, pose_path)
    
    x_tensor = torch.from_numpy(seq_amp).unsqueeze(0).to(device)

    print("[*] 正在进行 3D 姿态推理...")
    with torch.no_grad():
        pose_pred = model(x_tensor, train=False)
        pose_pred_np = pose_pred.squeeze(0).cpu().numpy()
        
        # ==========================================
        # 计算单样本四大 SOTA 指标
        # ==========================================
        # 0. 获取根节点对齐的相对坐标
        pred_rel = pose_pred_np - pose_pred_np[0]
        gt_rel = pose_gt - pose_gt[0]
        
        # 1. MPJPE (相对姿态误差)
        distances_rel = np.linalg.norm(pred_rel - gt_rel, axis=-1)
        mpjpe = np.mean(distances_rel) * 1000
        
        # 2. PA-MPJPE (普氏对齐误差)
        pred_opt, gt_opt = compute_similarity_transform(pred_rel, gt_rel)
        pa_distances = np.linalg.norm(pred_opt - gt_opt, axis=-1)
        pa_mpjpe = np.mean(pa_distances) * 1000
        
        # 3 & 4. PCK@20 & PCK@50
        pck_20 = np.mean(distances_rel < 0.02) * 100
        pck_50 = np.mean(distances_rel < 0.05) * 100

    print("\n" + "="*45)
    print(f"  单样本 ({sample_dir.split('/')[-1]}) 测试结果:")
    print("-" * 45)
    print(f" 1. PA-MPJPE (绝对极小误差) : {pa_mpjpe:.2f} mm")
    print(f" 2. MPJPE    (相对姿态误差) : {mpjpe:.2f} mm")
    print(f" 3. PCK@50   (<50mm 比例)   : {pck_50:.2f} %")
    print(f" 4. PCK@20   (<20mm 比例)   : {pck_20:.2f} %")
    print("="*45 + "\n")

    # 画图时传入的是未经中心化篡改的绝对坐标，但在标题上展示相对姿态的 MPJPE 误差
    title_info = f"3D Pose | MPJPE: {mpjpe:.1f}mm | PA-MPJPE: {pa_mpjpe:.1f}mm"
    plot_3d_pose(pose_gt, pose_pred_np, title_info)

if __name__ == "__main__":
    main()