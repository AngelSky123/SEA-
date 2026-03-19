import matplotlib.pyplot as plt

def plot_attention(attn_weights, save_path=None):
    # attn_weights: [B, heads, N, N]
    attn = attn_weights.mean(1)[0].detach().cpu().numpy()

    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title("Sensor Attention Map")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()