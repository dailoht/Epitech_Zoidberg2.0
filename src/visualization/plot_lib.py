"""
The module provides visualization tools using `matplotlib` library. It includes
functions to set default visualization parameters, plot a single image, and
plot multiple images organized by their labels.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


def default_viz(figsize=(7, 5), dpi=100):
    """
    Sets default parameters for Matplotlib visualizations.

    Args:
        figsize (tuple, optional): Size of the figure in inches. Defaults to
            (7, 5).
        dpi (int, optional): Resolution of the figure. Defaults to 100.
    """
    mpl.rc('axes', labelsize=12, titlesize=12)
    mpl.rc('figure', figsize=figsize, dpi=dpi)
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', labelsize=10)
    mpl.rc('image', cmap='hsv')


def plot_image(image):
    """
    Plots an image.

    Args:
        image (numpy.ndarray): Image to be plotted.
    """
    plt.imshow(image, cmap="gray")
    plt.axis("off")


def plot_images(instances, figsize=(12, 8)):
    """
    Plots a dictionary of images.

    Args:
        instances (dict): Dictionary containing image labels as keys and a
            list of images as values.
        figsize (tuple, optional): Size of the figure in inches. Defaults to
            (12, 8).
    """
    for label, images in instances.items():
        for image in images:
            plt.figure(figsize=figsize)
            idx_label = list(instances.keys()).index(label)
            idx_image = images.index(image)
            idx_subplot = idx_label * 3 + idx_image + 1
            plt.subplot(len(images), 3, idx_subplot)
            plt.title(label)
            plot_image(image)


if __name__ == "__main__":
    default_viz()
