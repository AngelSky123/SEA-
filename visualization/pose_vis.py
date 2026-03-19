import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# COCO skeleton (你可以改)
SKELETON = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),
    (7,10),(10,11),(11,12),
    (7,13),(13,14),(14,15)
]

def plot_pose(gt, pred=None, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw(pose, color):
        for i,j in SKELETON:
            x = [pose[i,0], pose[j,0]]
            y = [pose[i,1], pose[j,1]]
            z = [pose[i,2], pose[j,2]]
            ax.plot(x,y,z,color=color)

    draw(gt, 'blue')

    if pred is not None:
        draw(pred, 'red')

    ax.set_title("GT (blue) vs Pred (red)")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()