import torch
import torch.nn.functional as F

EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def compute_variance_loss(features):
    """VICReg 方差正则：每维 std >= 1。"""
    if features.shape[0] < 2:
        return torch.tensor(0.0, device=features.device)
    return F.relu(1.0 - features.std(dim=0)).mean()


def compute_covariance_loss(features):
    """VICReg 协方差正则。"""
    B, D = features.shape
    if B < 2:
        return torch.tensor(0.0, device=features.device)
    f = features - features.mean(dim=0)
    cov = (f.T @ f) / (B - 1)
    off = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    return (off ** 2).mean() / D


def compute_io_consistency_loss(csi, pred):
    """
    输入-输出距离一致性（v8 核心）。

    CSI 输入差异大 → 预测差异也应该大。
    这是目标域上唯一的结构保持信号，不需要标签。

    用 rank-based 而非 raw distance，对 scale 更鲁棒。
    """
    B = pred.shape[0]
    if B < 4:
        return torch.tensor(0.0, device=pred.device)

    # 输入距离（用首帧 + 末帧，增加信息量）
    csi_first = csi[:, 0].reshape(B, -1)
    csi_last  = csi[:, -1].reshape(B, -1)
    csi_repr  = torch.cat([csi_first, csi_last], dim=-1)    # (B, 2*N*C)
    csi_dist  = torch.cdist(csi_repr, csi_repr, p=2)        # (B, B)

    # 输出距离
    pred_flat = pred.reshape(B, -1)
    pred_dist = torch.cdist(pred_flat, pred_flat, p=2)

    # 归一化到 [0, 1]
    mask = torch.triu(torch.ones(B, B, device=pred.device), diagonal=1).bool()
    csi_d  = csi_dist[mask]
    pred_d = pred_dist[mask]

    csi_d  = csi_d  / (csi_d.max().clamp(min=1e-6))
    pred_d = pred_d / (pred_d.max().clamp(min=1e-6))

    return F.mse_loss(pred_d, csi_d)


def compute_loss(pred, vel_pred, gt, fs, ft, ds, dt,
                 orth_loss=None, pred_t=None,
                 enc_feat_s=None, enc_feat_t=None,
                 csi_s=None, csi_t=None,
                 da_weight=1.0):
    """
    统一损失函数。

    da_weight: 域自适应损失的全局权重，用于 curriculum：
               Phase 1 (da_weight=0): 纯监督
               Phase 2 (da_weight=0→1): 渐进引入 DA
    """
    device = pred.device

    # ── 监督损失（始终激活）─────────────────────────────────────────
    pose_loss = F.mse_loss(pred, gt[:, 0])

    gt_vel   = gt[:, -1] - gt[:, 0]
    vel_loss = F.mse_loss(vel_pred, gt_vel)

    bone_loss = torch.tensor(0.0, device=device)
    for i, j in EDGES:
        pred_len = torch.norm(pred[:, i] - pred[:, j],    dim=-1)
        gt_len   = torch.norm(gt[:, 0, i] - gt[:, 0, j], dim=-1)
        bone_loss = bone_loss + F.mse_loss(pred_len, gt_len)
    bone_loss = bone_loss / len(EDGES)

    # ── DA 损失（由 da_weight 控制）───────────────────────────────
    align_loss  = torch.tensor(0.0, device=device)
    domain_loss = torch.tensor(0.0, device=device)
    orth        = torch.tensor(0.0, device=device)
    var_loss    = torch.tensor(0.0, device=device)
    cov_loss    = torch.tensor(0.0, device=device)
    io_loss     = torch.tensor(0.0, device=device)

    if da_weight > 0:
        # 特征对齐
        align_loss = ((fs.mean(dim=(1, 2)) - ft.mean(dim=(1, 2))) ** 2).mean()

        # 对抗域分类
        src_labels = torch.zeros(ds.size(0), dtype=torch.long, device=device)
        tgt_labels = torch.ones( dt.size(0), dtype=torch.long, device=device)
        domain_loss = (F.cross_entropy(ds, src_labels) +
                       F.cross_entropy(dt, tgt_labels))

        # 正交
        if orth_loss is not None:
            orth = orth_loss

        # VICReg（只对目标域特征约束，源域有监督信号不需要）
        if enc_feat_t is not None:
            var_loss = compute_variance_loss(enc_feat_t)
            cov_loss = compute_covariance_loss(enc_feat_t)

        # IO 一致性（源域 + 目标域）
        if csi_s is not None:
            io_loss = io_loss + compute_io_consistency_loss(csi_s, pred)
        if csi_t is not None and pred_t is not None:
            io_loss = io_loss + compute_io_consistency_loss(csi_t, pred_t)

    total = (pose_loss
             + 0.1  * vel_loss
             + 0.1  * bone_loss
             + da_weight * (0.01 * align_loss
                            + 0.01 * domain_loss
                            + 0.01 * orth
                            + 1.0  * var_loss
                            + 0.04 * cov_loss
                            + 5.0  * io_loss))

    return total, {
        "pose":   pose_loss.item(),
        "vel":    vel_loss.item(),
        "bone":   bone_loss.item(),
        "align":  align_loss.item(),
        "domain": domain_loss.item(),
        "orth":   orth.item() if isinstance(orth, torch.Tensor) else float(orth),
        "var":    var_loss.item(),
        "cov":    cov_loss.item(),
        "io":     io_loss.item(),
    }