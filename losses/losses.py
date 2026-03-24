import torch
import torch.nn.functional as F

EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def compute_loss(pred, vel_pred, gt, fs, ft, ds, dt, orth_loss=None):
    """
    pred     : (B, J, 3)    — 预测第 0 帧姿态（相对坐标）
    vel_pred : (B, J, 3)    — 预测帧间位移（首帧→末帧）
    gt       : (B, T, J, 3) — 真值序列（已中心化）
    fs / ft  : (B, T, N, D) — 源域 / 目标域 shared 特征
    ds / dt  : (B, 2)       — 域判别器输出（已过 GRL）

    修复说明：
    1. gt_vel 语义明确为"首帧到末帧的关节位移"，与 vel_head 的设计对齐。
    2. vel_loss 权重从 0.5 降至 0.1，避免与 pose_loss 量级相近时喧宾夺主。
       速度监督的作用是提供运动方向信号，不应主导梯度。
    3. 损失函数签名与 train.py 调用保持一致。
    """

    # ── 1. 姿态 MSE（主损失）──────────────────────────────────────────────
    pose_loss = F.mse_loss(pred, gt[:, 0])

    # ── 2. 帧间位移监督 ─────────────────────────────────────────────────
    # 修复：gt_vel 明确定义为"末帧相对首帧的关节位移"，与 vel_head 的
    # 设计意图（PoseHead.vel_head: 预测 xT-x0 方向的关节位移）保持一致。
    gt_vel   = gt[:, -1] - gt[:, 0]         # (B, J, 3)
    vel_loss = F.mse_loss(vel_pred, gt_vel)

    # ── 3. 骨骼长度一致性 ────────────────────────────────────────────────
    bone_loss = torch.tensor(0.0, device=pred.device)
    for i, j in EDGES:
        pred_len = torch.norm(pred[:, i] - pred[:, j],    dim=-1)
        gt_len   = torch.norm(gt[:, 0, i] - gt[:, 0, j], dim=-1)
        bone_loss = bone_loss + F.mse_loss(pred_len, gt_len)
    bone_loss = bone_loss / len(EDGES)

    # ── 4. 特征对齐（shared 特征均值对齐）──────────────────────────────
    align_loss = ((fs.mean(dim=(1, 2)) - ft.mean(dim=(1, 2))) ** 2).mean()

    # ── 5. 对抗域分类 ────────────────────────────────────────────────────
    src_labels  = torch.zeros(ds.size(0), dtype=torch.long, device=ds.device)
    tgt_labels  = torch.ones( dt.size(0), dtype=torch.long, device=dt.device)
    domain_loss = (F.cross_entropy(ds, src_labels) +
                   F.cross_entropy(dt, tgt_labels))

    # ── 6. 正交损失 ──────────────────────────────────────────────────────
    orth = (orth_loss if orth_loss is not None
            else torch.tensor(0.0, device=pred.device))

    # 修复：vel_loss 权重从 0.5 降至 0.1
    # 速度监督提供运动方向信号，不应主导总梯度
    total = (pose_loss
             + 0.1  * vel_loss
             + 0.1  * bone_loss
             + 0.01 * align_loss
             + 0.01 * domain_loss
             + 0.01 * orth)

    return total, {
        "pose":   pose_loss.item(),
        "vel":    vel_loss.item(),
        "bone":   bone_loss.item(),
        "align":  align_loss.item(),
        "domain": domain_loss.item(),
        "orth":   orth.item() if isinstance(orth, torch.Tensor) else float(orth),
    }