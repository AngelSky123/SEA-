import torch

def mpjpe(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=-1))

def compute_similarity_transform(pred, gt):
    muX = pred.mean(0)
    muY = gt.mean(0)

    X0 = pred - muX
    Y0 = gt - muY

    normX = torch.norm(X0)
    normY = torch.norm(Y0)

    X0 /= normX
    Y0 /= normY

    H = X0.T @ Y0
    U, S, V = torch.svd(H)

    R = V @ U.T
    s = S.sum() * normY / normX

    return s * (pred - muX) @ R + muY

def pa_mpjpe(pred, gt):
    aligned = compute_similarity_transform(pred, gt)
    return torch.mean(torch.norm(aligned - gt, dim=-1))

def pck(pred, gt, threshold=0.05):
    dist = torch.norm(pred - gt, dim=-1)
    return (dist < threshold).float().mean()