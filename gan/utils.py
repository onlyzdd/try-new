import matplotlib.pyplot as plt


def plot_25(X, name):
    _, axs = plt.subplots(5, 5)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(X[i, :, :, 0], cmap='gray_r')
        ax.axis('off')
    plt.savefig(f'./imgs/{name}.png')
