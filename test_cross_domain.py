import torch
from torch.utils.data import DataLoader

from dataset.mmfi_dataset import MMFiDataset
from models.model import WiFiPoseModel
from utils_metrics import compute_metrics # 注意：确保你有这个模块
from utils.config import get_config

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    cfg = get_config()

    test_data = MMFiDataset(
        cfg.data.root,
        cfg.domain.target,
        cfg.data.seq_len
    )

    loader = DataLoader(
        test_data, 
        batch_size=16,       # 测试时不需要计算梯度，显存占用小，可以直接开到 32 或更大
        shuffle=False, 
        num_workers=8,       # 开启 8 个子进程疯狂读图
        pin_memory=True      # 开启锁页内存，加速数据往 GPU 搬运
    )

    model = WiFiPoseModel().to(device)
    
    #  修复点：使用 cfg.resume 读取路径，并提取 'model' 字典
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f" 成功加载权重: {cfg.resume}")
    else:
        raise ValueError(" 请通过 --resume 参数指定测试模型的权重路径！")
        
    model.eval()

    all_pred, all_gt = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            pred, _, _, _, _ = model(x, x)
            all_pred.append(pred.cpu())
            #  修复：像训练时一样，只取第 0 帧用于测试评估，并转移到 CPU 内存
            all_gt.append(y[:, 0].cpu())

    metrics = compute_metrics(all_pred, all_gt)

    print("===== RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()