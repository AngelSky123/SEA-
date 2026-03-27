import torch


def mpjpe(pred, gt):
    """
    均关节位置误差 (B, J, 3) 或 (J, 3) → 标量
    """
    return torch.mean(torch.norm(pred - gt, dim=-1))


def compute_similarity_transform(pred, gt):
    """
    对单帧骨架 (J, 3) 做 Procrustes 对齐。

    已验证的正确公式（H = X0.T @ Y0 约定下）：
        H   = X0.T @ Y0
        U,S,Vh = svd(H)
        d   = det(U @ Vh)          <- 不是 det(Vh.T @ U.T)
        R   = U @ sign_fix @ Vh    <- 不是 Vh.T @ sign_fix @ U.T
        ret = s * (pred-muX) @ R + muY   <- @ R，不是 @ R.T

    原版三处全错（det参数、R公式、返回值），导致 PA-MPJPE > MPJPE。
    """
    muX = pred.mean(0)
    muY = gt.mean(0)

    X0 = pred - muX
    Y0 = gt   - muY

    normX = torch.norm(X0)
    normY = torch.norm(Y0)

    X0 = X0 / (normX + 1e-8)
    Y0 = Y0 / (normY + 1e-8)

    H = X0.T @ Y0                           # (3, 3)
    U, S, Vh = torch.linalg.svd(H)

    # 修复1：det 的参数是 U @ Vh，不是 Vh.T @ U.T
    d = torch.det(U @ Vh)
    sign_fix = torch.diag(torch.tensor(
        [1.0] * (S.shape[0] - 1) + [d.sign().item()],
        device=pred.device
    ))

    # 修复2：R = U @ sign_fix @ Vh（不是 Vh.T @ sign_fix @ U.T）
    R = U @ sign_fix @ Vh

    s = S.sum() * normY / (normX + 1e-8)

    # 修复3：返回 @ R（不是 @ R.T）
    return s * (pred - muX) @ R + muY


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