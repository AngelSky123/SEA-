import torch
import torch.nn as nn
from .encoder import Encoder
from .pose_head import PoseHead
from .domain_disc import DomainDisc
from .disentangle import DomainDisentangle


class WiFiPoseModel(nn.Module):
    def __init__(self, in_dim=40, dim=64, num_joints=17, pose_head_old=False):
        super().__init__()
        self.encoder     = Encoder(in_dim=in_dim, dim=dim)
        self.disentangle = DomainDisentangle(dim)
        self.head        = PoseHead(dim, num_joints=num_joints,
                                    old_style=pose_head_old)
        self.domain      = DomainDisc(dim, num_env=2)

    def forward(self, xs, xt, alpha=1.0):
        fs_raw = self.encoder(xs)
        ft_raw = self.encoder(xt)

        fs_shared, fs_private = self.disentangle(fs_raw)
        ft_shared, ft_private = self.disentangle(ft_raw)

        pose_s, vel_pred = self.head(fs_shared)
        pose_t, _ = self.head(ft_shared)

        fs_s_avg = fs_shared.mean(dim=(1, 2))
        ft_s_avg = ft_shared.mean(dim=(1, 2))
        ds = self.domain(fs_s_avg, alpha)
        dt = self.domain(ft_s_avg, alpha)

        fs_p_avg = fs_private.mean(dim=(1, 2))
        ft_p_avg = ft_private.mean(dim=(1, 2))
        orth_s = DomainDisentangle.orthogonality_loss(fs_s_avg, fs_p_avg)
        orth_t = DomainDisentangle.orthogonality_loss(ft_s_avg, ft_p_avg)
        orth_loss = (orth_s + orth_t) * 0.5

        # encoder 层面均值特征（VICReg 用）
        enc_feat_s = fs_raw.mean(dim=(1, 2))
        enc_feat_t = ft_raw.mean(dim=(1, 2))

        return (pose_s, vel_pred, fs_shared, ft_shared, ds, dt,
                orth_loss, pose_t, enc_feat_s, enc_feat_t)