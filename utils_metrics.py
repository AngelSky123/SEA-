import torch


def mpjpe(pred, gt):
    """
    均关节位置误差 (B, J, 3) 或 (J, 3) → 标量
    """
    return torch.mean(torch.norm(pred - gt, dim=-1))


def compute_similarity_transform(pred, gt):
    """
    对单帧骨架 (J, 3) 做 Procrustes 对齐。

    修复（v4）：修正旋转矩阵 R 的公式。
    SVD 分解 H = U @ diag(S) @ Vh，最优旋转为 R = U @ sign_fix @ Vh。
    原版写成 Vh.T @ sign_fix @ U.T，是 R 的转置，导致对齐后误差反而
    增大，出现 PA-MPJPE > MPJPE 的数学错误。
    对齐后的预测点：aligned = s * (pred - muX) @ R.T + muY
    （@ R.T 是因为 pred 是行向量，旋转需右乘 R.T）
    """
    muX = pred.mean(0)
    muY = gt.mean(0)

    X0 = pred - muX
    Y0 = gt   - muY

    normX = torch.norm(X0)
    normY = torch.norm(Y0)

    X0 = X0 / (normX + 1e-8)
    Y0 = Y0 / (normY + 1e-8)

    H = X0.T @ Y0                          # (3, 3)
    U, S, Vh = torch.linalg.svd(H)        # H = U @ diag(S) @ Vh

    # 处理反射：当 det(U @ Vh) < 0 时翻转最后列
    d = torch.det(U @ Vh)
    sign_fix = torch.diag(torch.tensor(
        [1.0] * (S.shape[0] - 1) + [d.sign().item()],
        device=pred.device
    ))

    # 修复：R = U @ sign_fix @ Vh（原版错写为 Vh.T @ sign_fix @ U.T）
    R = U @ sign_fix @ Vh                  # (3, 3)
    s = S.sum() * normY / (normX + 1e-8)

    # pred 为行向量矩阵 (J, 3)，旋转右乘 R.T
    return s * (pred - muX) @ R.T + muY


def pa_mpjpe(pred, gt):
    """Procrustes 对齐后的 MPJPE，输入 (J, 3)"""
    aligned = compute_similarity_transform(pred, gt)
    return torch.mean(torch.norm(aligned - gt, dim=-1))


def pck(pred, gt, threshold=0.05):
    """PCK：关节距离 < threshold 的比例，输入 (B, J, 3) 或 (J, 3)"""
    dist = torch.norm(pred - gt, dim=-1)
    return (dist < threshold).float().mean()


def compute_metrics(all_pred, all_gt):
    """
    测试脚本调用的统一评估入口。
    all_pred / all_gt : list of (B, J, 3) tensors
    """
    preds = torch.cat(all_pred, dim=0)   # (N, J, 3)
    gts   = torch.cat(all_gt,  dim=0)

    val_mpjpe = mpjpe(preds, gts).item()
    val_pck   = pck(preds, gts, threshold=0.05).item()

    # PA-MPJPE 需要逐样本 SVD，无法向量化
    pa_list = [pa_mpjpe(preds[i], gts[i]).item() for i in range(preds.shape[0])]
    val_pa_mpjpe = sum(pa_list) / len(pa_list)

    return {
        "MPJPE":     val_mpjpe,
        "PA-MPJPE":  val_pa_mpjpe,
        "PCK@0.05":  val_pck,
    }