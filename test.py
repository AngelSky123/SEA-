import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

# 导入你现有的模型和数据加载器
from model import SEAplusplus
from data_loader import MMFiDataset

def mpjpe(pred, gt):
    """
    计算平均关节位置误差 (Mean Per Joint Position Error)
    pred: 预测的 3D 关节坐标 [Batch, Num_Joints, 3]
    gt: 真实的 3D 关节坐标 [Batch, Num_Joints, 3]
    返回: 平均误差标量
    """
    # 计算每个关节的三维欧式距离，然后再求所有关节和 Batch 的平均
    distances = torch.norm(pred - gt, dim=-1)
    return torch.mean(distances)

def main():
    parser = argparse.ArgumentParser(description="SEA++ 3D Human Pose Estimation Testing")
    parser.add_argument('--root', default='/path/to/MMFi', type=str, help='MMFi数据集的根目录')
    parser.add_argument('--target_env', default='E02', type=str, help='测试的目标环境 (如 E02)')
    parser.add_argument('--model_path', default='sea_model.pth', type=str, help='训练好的模型权重路径')
    parser.add_argument('--batch_size', default=16, type=int, help='测试时的 Batch Size')
    args = parser.parse_args()

    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    # 1. 实例化与训练时结构完全一致的模型
    print("[*] 正在初始化 SEA++ 模型...")
    # 注意：这里的 d_patch=32, d_model=64 必须和你在 train_uda.py 中设置的一模一样
    model = SEAplusplus(num_sensors=342, d_patch=32, d_model=64, num_branches=3, num_joints=17)
    model = model.to(device)

    # 2. 加载训练好的权重
    if not os.path.exists(args.model_path):
        print(f"[!] 错误: 未找到模型权重文件 '{args.model_path}'，请先运行训练脚本！")
        return
    
    print(f"[*] 正在加载权重文件 '{args.model_path}'...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 开启评估模式，这会关闭 Dropout 和 BatchNorm 等训练期特定的操作
    model.eval()

    # 3. 加载测试数据集
    print(f"[*] 正在准备目标域 ({args.target_env}) 的测试数据...")
    test_dataset = MMFiDataset(args.root, env=args.target_env, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    num_samples = len(test_dataset)
    print(f"[*] 成功加载测试数据，共计 {num_samples} 个样本。")

    # 4. 运行测试循环
    print("[*] 开始执行测试推理...")
    mpjpe_total = 0.0
    
    with torch.no_grad(): # 测试阶段不需要计算梯度，极大节省显存和加速
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            
            # 前向传播预测 3D 姿态
            # 设置 train=False，模型将跳过域对齐损失的计算，只返回姿态预测
            poses_pred = model(x, train=False)
            
            # 计算当前 Batch 的 MPJPE
            batch_error = mpjpe(poses_pred, y).item()
            
            # 累加总误差（乘以 batch 的实际大小防止最后一个 batch 不足）
            mpjpe_total += batch_error * y.size(0)
            
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"    - 进度: [{batch_idx + 1}/{len(test_loader)}] Batch 误差: {batch_error:.4f}")

    # 5. 输出最终结果
    final_mpjpe = mpjpe_total / num_samples
    print("="*50)
    print(f" 测试完成！")
    print(f"目标环境 [{args.target_env}] 的最终 MPJPE (平均关节误差) 为: {final_mpjpe:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()