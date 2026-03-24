import torch
import torch.nn as nn
from .encoder import Encoder
from .pose_head import PoseHead
from .domain_disc import DomainDisc
from .disentangle import DomainDisentangle


class WiFiPoseModel(nn.Module):
    def __init__(self, dim=64, num_joints=17):
        super().__init__()
        self.encoder     = Encoder(in_dim=4, dim=dim)
        self.disentangle = DomainDisentangle(dim)
        self.head        = PoseHead(dim, num_joints=num_joints)
        self.domain      = DomainDisc(dim, num_env=2)

    def forward(self, xs, xt, alpha=1.0):
        """
        返回:
          pose      : (B, J, 3)      — 姿态预测（相对坐标）
          vel_pred  : (B, J, 3)      — 预测帧间速度
          fs_shared : (B, T, N, D)
          ft_shared : (B, T, N, D)
          ds        : (B, 2)
          dt        : (B, 2)
          orth_loss : scalar

        修复：域判别器作用于 shared 特征（而非 private 特征）。
        对抗域自适应的核心目标是迫使 encoder 产生"域判别器无法区分的"
        shared 特征，因此 GRL 必须接在 shared 特征上。
        对 private 特征做域判别没有对抗意义，反而会削弱 private 特征
        保留域私有信息的能力。
        """
        fs_raw = self.encoder(xs)
        ft_raw = self.encoder(xt)

        fs_shared, fs_private = self.disentangle(fs_raw)
        ft_shared, ft_private = self.disentangle(ft_raw)

        # PoseHead 返回 (pose, vel_pred)
        pose, vel_pred = self.head(fs_shared)

        # 修复：对 shared 特征（而非 private）做对抗域分类
        fs_s_avg = fs_shared.mean(dim=(1, 2))   # (B, D)
        ft_s_avg = ft_shared.mean(dim=(1, 2))   # (B, D)
        ds = self.domain(fs_s_avg, alpha)
        dt = self.domain(ft_s_avg, alpha)

        # 正交损失仍使用 shared 与 private 的关系
        fs_p_avg = fs_private.mean(dim=(1, 2))
        ft_p_avg = ft_private.mean(dim=(1, 2))
        orth_s = DomainDisentangle.orthogonality_loss(fs_s_avg, fs_p_avg)
        orth_t = DomainDisentangle.orthogonality_loss(ft_s_avg, ft_p_avg)
        orth_loss = (orth_s + orth_t) * 0.5

        return pose, vel_pred, fs_shared, ft_shared, ds, dt, orth_loss