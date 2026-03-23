import torch
import torch.nn as nn
from .encoder import Encoder
from .pose_head import PoseHead
from .domain_disc import DomainDisc
from .disentangle import DomainDisentangle


class WiFiPoseModel(nn.Module):
    """
    跨域 WiFi 姿态估计主模型。

    修复点：
      1. dim 通过参数传入，不再硬编码，与 config 保持一致。
      2. 集成 DomainDisentangle，将特征分为 shared / private。
      3. DomainDisc 内置 GRL，encoder 梯度被反转，实现真正的对抗训练。
      4. forward 返回 alpha 供训练循环动态控制 GRL 强度。
    """

    def __init__(self, dim=64):
        super().__init__()
        self.encoder = Encoder(in_dim=4, dim=dim)
        self.disentangle = DomainDisentangle(dim)
        self.head = PoseHead(dim)
        self.domain = DomainDisc(dim, num_env=2)

    def forward(self, xs, xt, alpha=1.0):
        """
        xs    : (B, T, N, C) — 源域 CSI
        xt    : (B, T, N, C) — 目标域 CSI
        alpha : GRL 反转强度（训练过程中从小到大线性增长）

        返回：
          pose    : (B, J, 3)    — 姿态预测（基于源域 shared 特征）
          fs_shared : (B, T, N, D) — 源域共享特征
          ft_shared : (B, T, N, D) — 目标域共享特征
          ds      : (B, 2)       — 源域判别器 logits（经过 GRL）
          dt      : (B, 2)       — 目标域判别器 logits（经过 GRL）
          orth_loss : scalar     — 正交损失（鼓励 shared/private 不相关）
        """
        # ── Encoder ──────────────────────────────────────────────────────
        fs_raw = self.encoder(xs)   # (B, T, N, D)
        ft_raw = self.encoder(xt)

        # ── Disentangle ───────────────────────────────────────────────────
        fs_shared, fs_private = self.disentangle(fs_raw)
        ft_shared, ft_private = self.disentangle(ft_raw)

        # ── Pose head（仅用 shared 特征）─────────────────────────────────
        pose = self.head(fs_shared)

        # ── 域判别器（用 private 特征 + GRL）────────────────────────────
        # private 特征包含域特有信息，让判别器在此基础上对抗
        fs_p_avg = fs_private.mean(dim=(1, 2))   # (B, D)
        ft_p_avg = ft_private.mean(dim=(1, 2))

        ds = self.domain(fs_p_avg, alpha)
        dt = self.domain(ft_p_avg, alpha)

        # ── 正交损失 ─────────────────────────────────────────────────────
        fs_s_avg = fs_shared.mean(dim=(1, 2))
        ft_s_avg = ft_shared.mean(dim=(1, 2))
        orth_s = DomainDisentangle.orthogonality_loss(fs_s_avg, fs_p_avg)
        orth_t = DomainDisentangle.orthogonality_loss(ft_s_avg, ft_p_avg)
        orth_loss = (orth_s + orth_t) * 0.5

        return pose, fs_shared, ft_shared, ds, dt, orth_loss