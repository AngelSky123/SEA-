import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os

from model import SEAplusplus
from data_loader import MMFiDataset

def evaluate_absolute_metrics(preds, gts):
    distances_abs = np.linalg.norm(preds - gts, axis=-1)
    mpjpe_error = np.mean(distances_abs) * 1000
    pck_20 = np.mean(distances_abs < 0.02) * 100
    pck_50 = np.mean(distances_abs < 0.05) * 100
    return mpjpe_error, pck_20, pck_50

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/a123456/SEA-/MMFi', type=str)
    parser.add_argument('--target_env', default='E04', type=str)
    parser.add_argument('--model_path', default='sea_model.pth', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

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
    
    all_preds = []
    all_gts = []

    print("[*] 正在进行批量推理...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            poses_pred = model(x, train=False).cpu().numpy()
            poses_gt = y.numpy()
            
            all_preds.append(poses_pred)
            all_gts.append(poses_gt)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    print("[*] 正在计算最严格的绝对定位指标...")
    mpjpe, pck_20, pck_50 = evaluate_absolute_metrics(all_preds, all_gts)

    print("\n" + "="*50)
    print(f" 🎯 目标环境 [{args.target_env}] 真实物理空间最终结果:")
    print("-" * 50)
    print(f" 1. 绝对 MPJPE (完全无对齐误差): {mpjpe:.2f} mm")
    print(f" 2. 绝对 PCK@50 (<50mm 准确率): {pck_50:.2f} %")
    print(f" 3. 绝对 PCK@20 (<20mm 准确率): {pck_20:.2f} %")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()