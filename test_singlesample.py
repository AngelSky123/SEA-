import os
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入你的模型
from model import SEAplusplus

def load_single_sample(csi_dir, pose_path, L=297):
    """
    加载并预处理单个样本，逻辑与 data_loader.py 完全一致
    """
    # 1. 加载真实的 3D Pose
    pose_array = np.load(pose_path)
    if pose_array.ndim == 3: 
        pose_gt = np.mean(pose_array, axis=0) # [17, 3]
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
        
    # 4. 数据清洗与 Z-score 归一化 (核心！必须和训练时一样)
    seq_amp = np.nan_to_num(seq_amp, nan=0.0, posinf=0.0, neginf=0.0)
    pose_gt = np.nan_to_num(pose_gt, nan=0.0, posinf=0.0, neginf=0.0)
    
    seq_amp = seq_amp.astype(np.float32)
    seq_amp = (seq_amp - np.mean(seq_amp)) / (np.std(seq_amp) + 1e-5)
    
    return seq_amp, pose_gt

def plot_3d_pose(pose_gt, pose_pred, mpjpe_error):
    # 1. 恢复为绝对正确的 MM-Fi (Human3.6M) 17关节连线
    edges = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],
             [9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制 Ground Truth (绿色)
    ax.scatter(pose_gt[:, 0], pose_gt[:, 1], pose_gt[:, 2], c='green', s=40, label='Ground Truth')
    for i, j in edges:
        ax.plot([pose_gt[i, 0], pose_gt[j, 0]], 
                [pose_gt[i, 1], pose_gt[j, 1]], 
                [pose_gt[i, 2], pose_gt[j, 2]], c='green', alpha=0.6)

    # 绘制 Prediction (红色)
    ax.scatter(pose_pred[:, 0], pose_pred[:, 1], pose_pred[:, 2], c='red', s=40, marker='^', label='Prediction')
    for i, j in edges:
        ax.plot([pose_pred[i, 0], pose_pred[j, 0]], 
                [pose_pred[i, 1], pose_pred[j, 1]], 
                [pose_pred[i, 2], pose_pred[j, 2]], c='red', alpha=0.6, linestyle='--')

    ax.set_title(f"3D Human Pose Estimation (MPJPE: {mpjpe_error:.2f} mm)", fontsize=14)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    
    # ==========================================
    # 【核心修复】：解决 3D 比例失调导致人体被“压扁”的问题
    # 找出所有点的最大范围，强制 X, Y, Z 轴具有完全相同的绝对物理比例尺
    # ==========================================
    all_points = np.vstack([pose_gt, pose_pred])
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
    
    # 调整视角：如果发现人是躺着的，可以解开这行的注释调整摄像机角度
    # ax.view_init(elev=20, azim=45)

    ax.legend()
    plt.tight_layout()
    plt.savefig('pose_visualization.png', dpi=300)
    print("✨ 可视化结果已保存为 'pose_visualization.png'")

def main():
    # ==========================
    # 1. 路径设置 (请根据你需要测试的具体动作进行修改)
    # ==========================
    # 比如我们测试 E03 环境下，S01 用户的 A01 动作
    sample_dir = "/home/a123456/SEA-/MMFi/E03/S21/A01"
    csi_dir = os.path.join(sample_dir, "wifi-csi")
    pose_path = os.path.join(sample_dir, "ground_truth.npy")
    model_path = "sea_model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    # ==========================
    # 2. 加载模型
    # ==========================
    print("[*] 正在初始化并加载模型权重...")
    model = SEAplusplus(num_sensors=342, d_patch=32, d_model=64, num_branches=3, num_joints=17)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ==========================
    # 3. 加载并处理单条数据
    # ==========================
    print(f"[*] 正在处理样本: {sample_dir} ...")
    if not os.path.exists(csi_dir) or not os.path.exists(pose_path):
        print("❌ 错误: 找不到对应的 csi 目录或 ground_truth.npy 文件，请检查路径！")
        return

    seq_amp, pose_gt = load_single_sample(csi_dir, pose_path)
    
    # 转换为 Tensor，并增加 Batch 维度: [342, 297] -> [1, 342, 297]
    x_tensor = torch.from_numpy(seq_amp).unsqueeze(0).to(device)
    y_tensor = torch.from_numpy(pose_gt).to(device)

    # ==========================
    # 4. 模型推理
    # ==========================
    print("[*] 正在进行 3D 姿态推理...")
    with torch.no_grad():
        pose_pred = model(x_tensor, train=False)
        
        # 计算该样本的 MPJPE 误差
        distances = torch.norm(pose_pred - y_tensor.unsqueeze(0), dim=-1)
        mpjpe_error_m = torch.mean(distances).item()
        mpjpe_error_mm = mpjpe_error_m * 1000

    print(f"🎯 该样本的 MPJPE 误差为: {mpjpe_error_mm:.2f} 毫米")

    # ==========================
    # 5. 可视化
    # ==========================
    # 将 Tensor 转回 Numpy 以便画图 [1, 17, 3] -> [17, 3]
    pose_pred_np = pose_pred.squeeze(0).cpu().numpy()
    pose_gt_np = pose_gt

    plot_3d_pose(pose_gt_np, pose_pred_np, mpjpe_error_mm)

if __name__ == "__main__":
    main()