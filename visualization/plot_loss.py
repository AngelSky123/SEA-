"""
简易训练损失曲线绘制工具。

修复：
- 使用非交互式 Agg 后端，避免在无显示器的服务器环境 plt.show() 挂起。
- 统一为 savefig 接口，与项目其他可视化模块风格一致。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(losses, save_path="loss_curve.png", title="Training Loss"):
    """
    losses    : list or array of loss values
    save_path : 图片保存路径（默认 loss_curve.png）
    title     : 图标题
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5, color="#2E86AB")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")