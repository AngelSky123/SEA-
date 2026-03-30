import torch
import torch.nn.functional as F

EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def compute_diversity_loss(pred):
    """
    批内多样性损失：惩罚同一 batch 内预测骨架过于相似的情况。

    原理：计算 batch 内所有样本对之间的平均骨架距离，
    使用 margin hinge，只惩罚距离低于 margin 的样本对。
    margin=0.10 表示：不同样本的平均关节距离至少要有 10cm，
    否则施加惩罚。
    """
    B = pred.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=pred.device)

    flat = pred.view(B, -1)                                # (B, J*3)
    diff = flat.unsqueeze(0) - flat.unsqueeze(1)           # (B, B, J*3)
    dist = diff.norm(dim=-1)                               # (B, B)

    mask = torch.triu(torch.ones(B, B, device=pred.device), diagonal=1).bool()
    pairwise_dist = dist[mask]                             # (B*(B-1)/2,)

    margin = 0.10   # 10cm
    loss = F.relu(margin - pairwise_dist).mean()
    return loss


def compute_loss(pred, vel_pred, gt, fs, ft, ds, dt, orth_loss=None):
    """
    pred      : (B, J, 3)    — 预测第 0 帧姿态（相对坐标）
    vel_pred  : (B, J, 3)    — 预测帧间位移（首帧→末帧）
    gt        : (B, T, J, 3) — 真值序列（已中心化）
    fs / ft   : (B, T, N, D) — 源域 / 目标域 shared 特征
    ds / dt   : (B, 2)       — 域判别器输出（已过 GRL）

    修复（v5）：新增 diversity_loss，对抗均值塌陷。
    均值塌陷诊断标志：batch 内样本间预测距离 < 30mm，各关节 std < 5mm。
    diversity_loss 权重 1.0，与 pose_loss 同级，强制模型产生有区分度的预测。
    """

    # ── 1. 姿态 MSE ──────────────────────────────────────────────────
    pose_loss = F.mse_loss(pred, gt[:, 0])

    # ── 2. 帧间位移监督 ──────────────────────────────────────────────
    gt_vel   = gt[:, -1] - gt[:, 0]
    vel_loss = F.mse_loss(vel_pred, gt_vel)

    # ── 3. 骨骼长度一致性 ────────────────────────────────────────────
    bone_loss = torch.tensor(0.0, device=pred.device)
    for i, j in EDGES:
        pred_len = torch.norm(pred[:, i] - pred[:, j],    dim=-1)
        gt_len   = torch.norm(gt[:, 0, i] - gt[:, 0, j], dim=-1)
        bone_loss = bone_loss + F.mse_loss(pred_len, gt_len)
    bone_loss = bone_loss / len(EDGES)

    # ── 4. 特征对齐 ──────────────────────────────────────────────────
    align_loss = ((fs.mean(dim=(1, 2)) - ft.mean(dim=(1, 2))) ** 2).mean()

    # ── 5. 对抗域分类 ────────────────────────────────────────────────
    src_labels  = torch.zeros(ds.size(0), dtype=torch.long, device=ds.device)
    tgt_labels  = torch.ones( dt.size(0), dtype=torch.long, device=dt.device)
    domain_loss = (F.cross_entropy(ds, src_labels) +
                   F.cross_entropy(dt, tgt_labels))

    # ── 6. 正交损失 ──────────────────────────────────────────────────
    orth = (orth_loss if orth_loss is not None
            else torch.tensor(0.0, device=pred.device))

    # ── 7. 多样性损失（v5 新增，对抗均值塌陷）────────────────────────
    diversity_loss = compute_diversity_loss(pred)

    total = (pose_loss
             + 0.1  * vel_loss
             + 0.1  * bone_loss
             + 0.01 * align_loss
             + 0.01 * domain_loss
             + 0.01 * orth
             + 1.0  * diversity_loss)

    return total, {
        "pose":      pose_loss.item(),
        "vel":       vel_loss.item(),
        "bone":      bone_loss.item(),
        "align":     align_loss.item(),
        "domain":    domain_loss.item(),
        "orth":      orth.item() if isinstance(orth, torch.Tensor) else float(orth),
        "diversity": diversity_loss.item(),
    }