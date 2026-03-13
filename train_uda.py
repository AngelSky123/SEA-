import torch
import torch.optim as optim
import torch.nn as nn
from model import SEAplusplus
from data_loader import get_loaders
import argparse
import itertools

# Human3.6M 连线标准
EDGES = [(0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), (8,9),
         (9,10), (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]

def bone_length_loss(pred, gt):
    """计算预测骨骼长度与真实骨骼长度的误差"""
    loss = 0
    for u, v in EDGES:
        pred_len = torch.norm(pred[:, u, :] - pred[:, v, :], dim=-1)
        gt_len = torch.norm(gt[:, u, :] - gt[:, v, :], dim=-1)
        loss += torch.mean(torch.abs(pred_len - gt_len))
    return loss / len(EDGES)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/a123456/SEA-/MMFi', type=str)
    parser.add_argument('--source_envs', nargs='+', default=['E01', 'E02', 'E03'], help='源域环境列表')
    parser.add_argument('--target_env', default='E04', type=str, help='目标测试环境')
    parser.add_argument('--epochs', default=300, type=int) 
    parser.add_argument('--lr', default=0.0005, type=float) 
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    model = SEAplusplus(num_sensors=342, d_patch=11, d_model=128, num_branches=3, num_joints=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f" 训练源域 (监督学习): {args.source_envs}")
    print(f" 对齐目标域 (无监督 UDA): {args.target_env}")
    
    source_loader, target_loader, test_loader = get_loaders(args.root, args.source_envs, args.target_env, args.batch_size)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss_val = 0
        
        target_iter = itertools.cycle(target_loader)
        
        for step, (x_s, y_s) in enumerate(source_loader):
            x_t, _ = next(target_iter)
            
            x_s, y_s = x_s.to(device), y_s.to(device)
            x_t = x_t.to(device)

            optimizer.zero_grad()
            poses_s, align_loss = model(x_s, x_t, train=True)
            
            # 1. 整体姿态损失
            sup_loss = criterion(poses_s, y_s)
            
            # 2. 骨骼长度损失 (物理限制)
            b_loss = bone_length_loss(poses_s, y_s) 
            
            # 3. 根节点绝对定位损失 (死盯宏观定位)
            root_pred = poses_s[:, 0, :]
            root_gt = y_s[:, 0, :]
            root_loss = torch.mean(torch.norm(root_pred - root_gt, dim=-1))
            
            # ==========================================
            # 【核心改进】：UDA 动态对齐权重 (Curriculum Learning)
            # 前 50 epoch 完全不考虑跨房间对齐，让模型疯狂记住绝对位置
            # 50 epoch 之后，慢慢加入对齐，防止负迁移
            # ==========================================
            alpha = max(0.0, min(1.0, (epoch - 50) / 100.0))
            
            total_loss = sup_loss + 0.5 * b_loss + 2.0 * root_loss + alpha * align_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss_val += total_loss.item()
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch+1:03d}/{args.epochs} | LR: {current_lr:.6f} | Alpha: {alpha:.2f} | Avg Loss: {total_loss_val/len(source_loader):.4f}')
        
    torch.save(model.state_dict(), 'sea_model.pth')
    print(" Model successfully saved to sea_model.pth")