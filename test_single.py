import torch
import argparse

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from visualization.pose_vis import plot_pose
from utils_metrics import mpjpe, pa_mpjpe, pck
import config

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='E04')
    parser.add_argument('--subject', default='S35')
    parser.add_argument('--action', default='A01')
    #  替换为 --work：专门用于指定测试权重路径
    parser.add_argument('--work', required=True, help="模型权重的路径")
    
    # 兼容可能传入的 --config 等其他未定义参数，防止报错
    args, _ = parser.parse_known_args()

    dataset = MMFiDataset(config.DATA_ROOT, [args.env])

    model = WiFiPoseModel().to(device)
    
    #  使用 args.work 读取路径
    if args.work:
        checkpoint = torch.load(args.work, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f" 成功加载权重: {args.work}")
    else:
        raise ValueError(" 请通过 --work 参数指定测试模型的权重路径！")
        
    model.eval()

    print(f"Testing {args.env}-{args.subject}-{args.action}")

    for i in range(len(dataset)):
        csi, pose = dataset[i]

        x = csi.unsqueeze(0).to(device)
        gt = pose[0].to(device)

        with torch.no_grad():
            pred, _, _, _, _ = model(x, x)

        pred = pred[0]

        print("MPJPE:", mpjpe(pred, gt).item())
        print("PA-MPJPE:", pa_mpjpe(pred, gt).item())
        print("PCK@50:", pck(pred, gt, 0.05).item())
        print("PCK@20:", pck(pred, gt, 0.02).item())

        plot_pose(
            gt.cpu().numpy(),
            pred.cpu().numpy(),
            save_path="single_pose.png"
        )

        print(" Visualization saved: single_pose.png")
        break # 测完第一个就退出

if __name__ == "__main__":
    main()