import torch.nn as nn
from .encoder import Encoder
from .pose_head import PoseHead
from .domain_disc import DomainDisc

class WiFiPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = PoseHead(64)
        self.domain = DomainDisc(64)

    def forward(self, xs, xt):
        fs = self.encoder(xs)
        ft = self.encoder(xt)

        pose = self.head(fs)

        ds = self.domain(fs.mean((1,2)))
        dt = self.domain(ft.mean((1,2)))

        return pose, fs, ft, ds, dt