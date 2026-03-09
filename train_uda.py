import torch
import torch.optim as optim
import torch.nn as nn
from model import SEAplusplus
from data_loader import get_loaders
import argparse

def mpjpe(pred, gt):  # MPJPE指标
    return torch.mean(torch.norm(pred - gt, dim=-1))

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/path/to/MMFi', type=str)
parser.add_argument('--source_env', default='E01', type=str)
parser.add_argument('--target_env', default='E02', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=32, type=int)
args = parser.parse_args()

model = SEAplusplus()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.SmoothL1Loss()

source_loader, target_loader, test_loader = get_loaders(args.root, args.source_env, args.target_env, args.batch_size)

if args.mode == 'train':
    for epoch in range(args.epochs):
        model.train()
        for (x_s, y_s), (x_t, _) in zip(source_loader, target_loader):
            optimizer.zero_grad()
            poses_s, align_loss = model(x_s, x_t)
            sup_loss = criterion(poses_s, y_s)
            total_loss = sup_loss + align_loss
            total_loss.backward()
            optimizer.step()
        print(f'轮次 {epoch}: 损失 {total_loss.item()}')
    torch.save(model.state_dict(), 'sea_model.pth')
else:
    model.load_state_dict(torch.load('sea_model.pth'))
    model.eval()
    mpjpe_total = 0
    num = 0
    for x, y in test_loader:
        poses = model(x, None, train=False)
        mpjpe_total += mpjpe(poses, y) * y.size(0)
        num += y.size(0)
    print(f'测试 MPJPE: {mpjpe_total / num}')