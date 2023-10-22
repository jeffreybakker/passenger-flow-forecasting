import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


def full_extent(ax, pad=0.0):
    """
    Get the full extent of an axes, including axes labels, tick labels, and
    titles.

    :Authors:
    Joe Kington @ StackOverflow: https://stackoverflow.com/users/325565/joe-kington

    See relevant StackOverflow answer:
    https://stackoverflow.com/questions/14712665/matplotlib-subplot-background-axes-face-labels-colour-or-figure-axes-coor/14720600#14720600
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def save_subplot(fig, ax, filename):
    # Get the boundaries of the subplot (ax)
    extent = full_extent(ax).transformed(fig.dpi_scale_trans.inverted())

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches=extent, transparent=True)
