import torch
import torch.nn.functional as F


def compute_loss(pred, gt, fs, ft, ds, dt, orth_loss=None):
    """
    pred  : (B, J, 3)    — 预测骨架（相对坐标）
    gt    : (B, T, J, 3) — 真值序列（已中心化，取第 0 帧）
    """

    # ── 1. 姿态回归损失（提高权重至主导地位）────────────────────────────
    pose_loss = F.mse_loss(pred, gt[:, 0])

    # ── 2. 骨骼长度一致性损失（约束骨架结构合理性）──────────────────────
    EDGES = [
        (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),
        (0,7),(7,8),(8,9),(9,10),
        (8,11),(11,12),(12,13),
        (8,14),(14,15),(15,16),
    ]
    bone_loss = torch.tensor(0.0, device=pred.device)
    for i, j in EDGES:
        pred_len = torch.norm(pred[:, i] - pred[:, j], dim=-1)
        gt_len   = torch.norm(gt[:, 0, i] - gt[:, 0, j], dim=-1)
        bone_loss = bone_loss + F.mse_loss(pred_len, gt_len)
    bone_loss = bone_loss / len(EDGES)

    # ── 3. 特征对齐损失 ──────────────────────────────────────────────────
    align_loss = ((fs.mean(dim=(1, 2)) - ft.mean(dim=(1, 2))) ** 2).mean()

    # ── 4. 对抗域分类损失 ────────────────────────────────────────────────
    source_labels = torch.zeros(ds.size(0), dtype=torch.long, device=ds.device)
    target_labels = torch.ones(dt.size(0),  dtype=torch.long, device=dt.device)
    domain_loss   = F.cross_entropy(ds, source_labels) + \
                    F.cross_entropy(dt, target_labels)

    # ── 5. 正交损失 ──────────────────────────────────────────────────────
    orth = orth_loss if orth_loss is not None \
           else torch.tensor(0.0, device=pred.device)

    # 调整权重：pose 是主任务，大幅提高其权重
    total = pose_loss \
            + 0.1  * bone_loss   \
            + 0.01 * align_loss  \
            + 0.01 * domain_loss \
            + 0.01 * orth

    return total, {
        "pose":   pose_loss.item(),
        "bone":   bone_loss.item(),
        "align":  align_loss.item(),
        "domain": domain_loss.item(),
        "orth":   orth.item() if isinstance(orth, torch.Tensor) else float(orth),
    }