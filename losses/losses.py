import torch.nn.functional as F

def compute_loss(pred, gt, fs, ft, ds, dt):

    pose_loss = F.mse_loss(pred, gt[:,0])

    align_loss = ((fs.mean(0)-ft.mean(0))**2).mean()

    domain_loss = F.cross_entropy(ds, ds.argmax(1)) + \
                  F.cross_entropy(dt, dt.argmax(1))

    return pose_loss + 0.1*align_loss + 0.01*domain_loss