import torch
import os

def save_checkpoint(state, save_dir, filename="checkpoint.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f" Saved checkpoint: {path}")

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint.get('epoch', 0)

    print(f" Loaded checkpoint from {path}, epoch {start_epoch}")

    return model, optimizer, start_epoch