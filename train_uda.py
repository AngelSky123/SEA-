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
    parser.add_argument('--root', default='/path/to/MMFi', type=str)
    parser.add_argument('--source_env', default='E01', type=str)
    parser.add_argument('--target_env', default='E02', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--epochs', default=5, type=int) # 调低 epoch 测试
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SEAplusplus().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()

    source_loader, target_loader, test_loader = get_loaders(args.root, args.source_env, args.target_env, args.batch_size)

    if args.mode == 'train':
        for epoch in range(args.epochs):
            model.train()
            total_loss_val = 0
            
            # 使用 cycle 保证小数据集循环以适配大数据集
            target_iter = itertools.cycle(target_loader)
            
            for step, (x_s, y_s) in enumerate(source_loader):
                x_t, _ = next(target_iter)
                
                # 转移到 GPU/CPU
                x_s, y_s = x_s.to(device), y_s.to(device)
                x_t = x_t.to(device)

                optimizer.zero_grad()
                poses_s, align_loss = model(x_s, x_t, train=True)
                
                sup_loss = criterion(poses_s, y_s)
                total_loss = sup_loss + align_loss
                
                total_loss.backward()
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
                # 计算 MPJPE 误差累加
                mpjpe_total += mpjpe(poses, y).item() * y.size(0)
                num += y.size(0)
        print(f'Test MPJPE: {mpjpe_total / num:.4f}')