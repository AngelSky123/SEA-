import torch

def mpjpe(pred, gt):
    # 计算均方根误差 (Batch, Joints, 3) -> 标量标量平均值
    return torch.mean(torch.norm(pred - gt, dim=-1))

def compute_similarity_transform(pred, gt):
    """
    针对单帧骨架 (J, 3) 计算 Procrustes 对齐
    """
    muX = pred.mean(0)
    muY = gt.mean(0)

    X0 = pred - muX
    Y0 = gt - muY

    normX = torch.norm(X0)
    normY = torch.norm(Y0)

    X0 /= normX
    Y0 /= normY

    H = X0.T @ Y0
    U, S, V = torch.svd(H)

    R = V @ U.T
    s = S.sum() * normY / normX

    return s * (pred - muX) @ R + muY

def pa_mpjpe(pred, gt):
    aligned = compute_similarity_transform(pred, gt)
    return torch.mean(torch.norm(aligned - gt, dim=-1))

def pck(pred, gt, threshold=0.05):
    dist = torch.norm(pred - gt, dim=-1)
    return (dist < threshold).float().mean()

def compute_metrics(all_pred, all_gt):
    """
    测试脚本 test_cross_domain.py 调用的统一评估入口
    """
    # 将列表拼接为完整的张量 (Total_Samples, Num_Joints, 3)
    preds = torch.cat(all_pred, dim=0)
    gts = torch.cat(all_gt, dim=0)

    # 1. 计算整体 MPJPE
    val_mpjpe = mpjpe(preds, gts).item()

    # 2. 计算整体 PCK
    val_pck = pck(preds, gts, threshold=0.05).item()

    # 3. 计算 PA-MPJPE (需要遍历 Batch 中的每一个样本单独做 SVD 对齐)
    pa_mpjpe_list = []
    for i in range(preds.shape[0]):
        # 取出单帧预测和真实骨架 (J, 3)
        pa_err = pa_mpjpe(preds[i], gts[i])
        pa_mpjpe_list.append(pa_err.item())
    
    val_pa_mpjpe = sum(pa_mpjpe_list) / len(pa_mpjpe_list)

    # 组装返回字典
    return {
        "MPJPE": val_mpjpe,
        "PA-MPJPE": val_pa_mpjpe,
        "PCK@0.05": val_pck
    }