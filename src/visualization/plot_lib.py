import matplotlib as mpl


def default_viz(figsize=(7, 5), dpi=100):
    mpl.rc('axes', labelsize=12, titlesize=12)
    mpl.rc('figure', figsize=figsize, dpi=dpi)
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', labelsize=10)
    mpl.rc('image', cmap='hsv')