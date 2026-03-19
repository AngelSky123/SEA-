import torch
from torch.utils.data import DataLoader

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from utils_metrics import compute_metrics
from utils.config import get_config

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    cfg = get_config()

    test_data = MMFiDataset(
        cfg.data.root,
        cfg.domain.target,
        cfg.data.seq_len
    )

    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = WiFiPoseModel().to(device)
    model.load_state_dict(torch.load("latest.pth"))
    model.eval()

    all_pred, all_gt = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            pred, _, _, _, _ = model(x, x)
            all_pred.append(pred.cpu())
            all_gt.append(y)

    metrics = compute_metrics(all_pred, all_gt)

    print("===== RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()