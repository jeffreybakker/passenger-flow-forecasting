from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def style_plot(ax: Axes):
    ax.grid(True, color='black', alpha=0.2)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_alpha(0.7)
    ax.spines['bottom'].set_alpha(0.7)
