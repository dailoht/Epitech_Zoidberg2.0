import matplotlib as mpl
import matplotlib.pyplot as plt


def default_viz(figsize=(7, 5), dpi=100):
    mpl.rc('axes', labelsize=12, titlesize=12)
    mpl.rc('figure', figsize=figsize, dpi=dpi)
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', labelsize=10)
    mpl.rc('image', cmap='hsv')


def plot_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")


def plot_images(instances, figsize=(12,8)):
    for label, images in instances.items():
        for image in images:
            plt.figure(figsize=figsize)
            idx_subplot = list(instances.keys()).index(label) * 3 + images.index(image) + 1
            plt.subplot(len(images), 3,idx_subplot)
            plt.title(label)
            plot_image(image)
