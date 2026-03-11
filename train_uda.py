import torch
import torch.optim as optim
import torch.nn as nn
from model import SEAplusplus
from data_loader import get_loaders
import argparse
import itertools

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

    model = SEAplusplus(num_sensors=342, d_patch=32, d_model=64, num_branches=3, num_joints=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

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
                total_loss = sup_loss + align_loss
                
                total_loss.backward()
                
                # 【防护 3】：梯度裁剪，死死锁住梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss_val += total_loss.item()
                
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