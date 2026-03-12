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

def mpjpe(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/a123456/SEA-/MMFi', type=str)
    parser.add_argument('--source_envs', nargs='+', default=['E01', 'E02', 'E03'], help='源域环境列表')
    parser.add_argument('--target_env', default='E04', type=str, help='目标测试环境')
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--epochs', default=50, type=int) 
    # 【修改】：学习率调低到 0.0001
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # d_patch=11：将时间分辨率提高，每 11 帧打包一次（297/11=27个动作块），捕捉更细腻的动作
    # d_model=128：特征维度翻倍，让网络能记住更复杂的空间映射关系
    model = SEAplusplus(num_sensors=342, d_patch=11, d_model=128, num_branches=3, num_joints=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    # 【新增代码 1】：加入余弦退火学习率调度器，让学习率像平滑的波浪一样慢慢降到接近 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"[*] 训练源域 (Source Domains): {args.source_envs}")
    print(f"[*] 评估目标域 (Target Domain): {args.target_env}")
    
    source_loader, target_loader, test_loader = get_loaders(args.root, args.source_envs, args.target_env, args.batch_size)
    
    # 打印数据集大小，确认读取到了真实数据
    print(f"🔥 实际加载的源域训练样本数: {len(source_loader.dataset)}")
    print(f"🔥 实际加载的目标域测试样本数: {len(test_loader.dataset)}")

    if args.mode == 'train':
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
                
                sup_loss = criterion(poses_s, y_s)
                b_loss = bone_length_loss(poses_s, y_s) # 计算物理骨骼误差
                
                # 将坐标误差、骨骼误差和域对齐误差加在一起
                # 0.5 是权重，可以调
                total_loss = sup_loss + 0.5 * b_loss + align_loss
                
                total_loss.backward()
                
                # 【防护 3】：梯度裁剪，死死锁住梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss_val += total_loss.item()
                # 【新增代码 2】：每个 Epoch 跑完后，让调度器更新一次学习率
            scheduler.step()

            # 【优化打印】：顺便把当前学习率打印出来，看着它慢慢变小
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f} | Avg Loss: {total_loss_val/len(source_loader):.4f}')
                
            print(f'Epoch {epoch+1}/{args.epochs} | Avg Loss: {total_loss_val/len(source_loader):.4f}')
            
        torch.save(model.state_dict(), 'sea_model.pth')
        print("Model saved to sea_model.pth")
        
    else:
        model.load_state_dict(torch.load('sea_model.pth', map_location=device))
        model.eval()
        mpjpe_total = 0.0
        num = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                poses = model(x, train=False)
                mpjpe_total += mpjpe(poses, y).item() * y.size(0)
                num += y.size(0)
        print(f'Test MPJPE: {mpjpe_total / num:.4f} 米 ({mpjpe_total / num * 1000:.2f} 毫米)')