import matplotlib.pyplot as plt

from astropy.visualization import ZScaleInterval
from astropy.visualization.stretch import LinearStretch


def plot_stamp(x, figsize=(7.5, 3)):
    transform = LinearStretch() + ZScaleInterval()

    fig, axes = plt.subplots(ncols=3, figsize=figsize)
    for col in range(3):
        img = x[:, :, col]
        img = transform(img)
        axes[col].imshow(img, cmap='gray', origin='lower')
        axes[col].axis('off')
    axes[0].set_title('Ref')
    axes[1].set_title('New')
    axes[2].set_title('Sub')

    return axes
