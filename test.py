import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os

from model import SEAplusplus
from data_loader import MMFiDataset

def compute_similarity_transform(S1, S2):
    """
    计算普氏分析 (Procrustes Analysis) 的变换矩阵。
    找到最优的旋转、平移和缩放，使预测骨架 S1 最大限度贴合真实骨架 S2。
    """
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
    
    # 奇异值分解求最优旋转矩阵
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

def evaluate_all_metrics(preds, gts):
    """
    批量计算 3D HPE 的四大核心指标: PA-MPJPE, MPJPE, PCK@20, PCK@50
    preds: 预测坐标数组 shape (N, 17, 3)
    gts:   真实坐标数组 shape (N, 17, 3)
    """
    N, num_joints, _ = preds.shape
    
    # ==================================================
    # 0. Root-Relative 对齐 (学术界 MPJPE 的默认前置操作)
    # 强制将根节点 (Joint 0) 对齐到原点 (0,0,0)
    # ==================================================
    preds_rel = preds - preds[:, 0:1, :]
    gts_rel = gts - gts[:, 0:1, :]
    
    # 计算每个关节的 3D 欧式距离 (米)
    distances_rel = np.linalg.norm(preds_rel - gts_rel, axis=-1) # shape: (N, 17)
    
    # ==================================================
    # 1. MPJPE (Root-Relative)
    # ==================================================
    mpjpe_error = np.mean(distances_rel) * 1000  # 转为毫米
    
    # ==================================================
    # 2. PA-MPJPE (Procrustes Aligned)
    # ==================================================
    pa_distances = np.zeros((N, num_joints))
    for i in range(N):
        pred_opt, gt_opt = compute_similarity_transform(preds_rel[i], gts_rel[i])
        pa_distances[i] = np.linalg.norm(pred_opt - gt_opt, axis=-1)
    pa_mpjpe_error = np.mean(pa_distances) * 1000 # 转为毫米
    
    # ==================================================
    # 3. PCK@20 (Percentage of Correct Keypoints < 20mm)
    # 距离误差在 20 毫米以内的极其精准的关节比例
    # ==================================================
    threshold_20 = 0.02 # 20 mm = 0.02 m
    pck_20 = np.mean(distances_rel < threshold_20) * 100
    
    # ==================================================
    # 4. PCK@50 (Percentage of Correct Keypoints < 50mm)
    # 距离误差在 50 毫米以内的较精准的关节比例
    # ==================================================
    threshold_50 = 0.05 # 50 mm = 0.05 m
    pck_50 = np.mean(distances_rel < threshold_50) * 100

    return pa_mpjpe_error, mpjpe_error, pck_20, pck_50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/a123456/SEA-/MMFi', type=str)
    parser.add_argument('--target_env', default='E04', type=str)
    parser.add_argument('--model_path', default='sea_model.pth', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    print("[*] 正在初始化 SEA++ 模型...")
    # 保持与你当前训练好的模型架构参数完全一致
    model = SEAplusplus(num_sensors=342, d_patch=11, d_model=128, num_branches=3, num_joints=17).to(device)

    if not os.path.exists(args.model_path):
        print(f"[!] 错误: 未找到权重 '{args.model_path}'")
        return
    
    print(f"[*] 加载权重文件 '{args.model_path}'...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print(f"[*] 加载目标域 ({args.target_env}) 测试数据...")
    test_dataset = MMFiDataset(args.root, envs=args.target_env, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[*] 成功加载测试数据: {len(test_dataset)} 个样本")
    
    all_preds = []
    all_gts = []

    print("[*] 正在进行批量推理...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            # 收集预测和真实坐标，转为 numpy 以便进行严谨的数学分析
            poses_pred = model(x, train=False).cpu().numpy()
            poses_gt = y.numpy()
            
            all_preds.append(poses_pred)
            all_gts.append(poses_gt)
            
    # 将所有的 batch 拼接到一起
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    print("[*] 正在计算 SOTA 评估四大指标...")
    pa_mpjpe, mpjpe, pck_20, pck_50 = evaluate_all_metrics(all_preds, all_gts)

    print("\n" + "="*50)
    print(f"  目标环境 [{args.target_env}] 的最终测试结果:")
    print("-" * 50)
    print(f" 1. PA-MPJPE (绝对极小姿态误差) : {pa_mpjpe:.2f} mm")
    print(f" 2. MPJPE    (相对姿态平均误差) : {mpjpe:.2f} mm")
    print(f" 3. PCK@50   (<50mm 关节比例)   : {pck_50:.2f} %")
    print(f" 4. PCK@20   (<20mm 关节比例)   : {pck_20:.2f} %")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()