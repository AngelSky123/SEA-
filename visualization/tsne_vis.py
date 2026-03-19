from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def tsne_plot(fs, ft, save_path=None):
    fs = fs.detach().cpu().numpy()
    ft = ft.detach().cpu().numpy()

    X = np.concatenate([fs, ft], axis=0)
    y = np.array([0]*len(fs) + [1]*len(ft))

    X_emb = TSNE(n_components=2).fit_transform(X)

    plt.scatter(X_emb[y==0,0], X_emb[y==0,1], label='Source', alpha=0.5)
    plt.scatter(X_emb[y==1,0], X_emb[y==1,1], label='Target', alpha=0.5)

    plt.legend()
    plt.title("t-SNE Domain Alignment")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()