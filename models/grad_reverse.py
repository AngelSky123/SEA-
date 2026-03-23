import torch
import torch.nn as nn


class GradReverseFunction(torch.autograd.Function):
    """梯度反转层：前向传播恒等，反向传播乘以 -alpha。"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverseFunction.apply(x, alpha)


class GradReversalLayer(nn.Module):
    """可作为 nn.Module 插入网络的梯度反转层。"""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return grad_reverse(x, self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha