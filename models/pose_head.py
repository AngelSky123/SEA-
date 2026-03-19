import torch.nn as nn

class PoseHead(nn.Module):
    def __init__(self, dim, joints=17):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim,128),
            nn.ReLU(),
            nn.Linear(128,joints*3)
        )

    def forward(self, x):
        x = x.mean(dim=(1,2))
        return self.fc(x).view(x.size(0), -1, 3)