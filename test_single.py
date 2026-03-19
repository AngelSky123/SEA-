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
    parser.add_argument('--env', default='E01')
    parser.add_argument('--subject', default='S01')
    parser.add_argument('--action', default='A01')
    args = parser.parse_args()

    dataset = MMFiDataset(config.DATA_ROOT, [args.env])

    model = WiFiPoseModel().to(device)
    model.load_state_dict(torch.load("model.pth"))
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
        break

if __name__ == "__main__":
    main()