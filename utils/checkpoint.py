import torch
import os


def save_checkpoint(state, save_dir, filename="checkpoint.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f" Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer=None):
    """
    加载 checkpoint，支持新旧结构部分复用。

    修复：load_state_dict 增加 strict=False，并打印详细加载报告。
    旧版 strict=True 在模型结构有任何变化时直接报错，不符合 README 的承诺。
    """
    checkpoint = torch.load(path, map_location="cpu")

    state_dict = checkpoint["model"]
    result = model.load_state_dict(state_dict, strict=False)

    # 打印详细加载报告
    if result.missing_keys:
        print(f"  [Checkpoint] Missing keys ({len(result.missing_keys)}) "
              f"— randomly initialized:")
        for k in result.missing_keys:
            print(f"    - {k}")
    if result.unexpected_keys:
        print(f"  [Checkpoint] Unexpected keys ({len(result.unexpected_keys)}) "
              f"— ignored:")
        for k in result.unexpected_keys:
            print(f"    + {k}")
    if not result.missing_keys and not result.unexpected_keys:
        print("  [Checkpoint] All keys matched perfectly.")

    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"  [Checkpoint] Optimizer state not loaded ({e}), "
                  f"will start fresh.")

    start_epoch = checkpoint.get("epoch", 0)
    print(f"  Loaded checkpoint from {path}, resuming at epoch {start_epoch + 1}")

    return model, optimizer, start_epoch