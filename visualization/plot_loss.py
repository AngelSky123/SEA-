import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()