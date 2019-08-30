""" Helper functions to add common formatting done to plots
"""
import matplotlib as mpl
import numpy as np
from typing import Dict

def set_square_axis(ax: mpl.axes.Axes):
    """ Set the axis limits to be equal for x and y

    Arguments:
        ax {mpl.axes.Axes} -- axis to modify
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    max_val = max([x_max, y_max])
    min_val = min([x_min, y_min])

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

def add_equality_line(ax: mpl.axes.Axes, **plot_kwargs):
    """ Add line at y=x to plot spanning plot limits
    kwargs sent to ax.plot() function
    
    Arguments:
        ax {mpl.axes.Axes} -- matplotlib axis to add line to
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    max_val = max([x_max, y_max])
    min_val = min([x_min, y_min])

    ax.plot([min_val, max_val], [min_val, max_val], **plot_kwargs)

def add_grid_despine_linewidth(ax: mpl.axes.Axes, linewidth: int = 3, grid: bool = True):
    """ Makes graphs 'pretty'. Remove top bottom, center ticks, thicken axes
    
    Arguments:
        ax {mpl.axes.Axes}
        linewidth {int} -- width fo ticks and axis lines
        grid {bool} -- whether to add grid or not
    """

    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='inout', length=10, width=linewidth)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(linewidth)
    if grid:
        ax.grid(linestyle='--', alpha=0.3, linewidth=linewidth)

def tick_at_end_of_axis(
        ax: mpl.axes.Axes,
        n_ticks: int = None,
        x_0: float = None,
        y_0: float = None,
        set_x_max: float = None,
        set_y_max: float = None,
        round_tick_end: bool = True,
        round_base: int = 5,
        hide_min: bool = False,
        round_ticks: bool = False
    ):
    """ Make ticks end at axis limits, formats x and y ticks

    Arguments:
        ax {mpl.axes.Axes}

    Keyword Arguments:
        n_ticks {int} -- set to number of ticks to create
            default uses what exists on axis (default: {None})
        x_0 {float} -- set to where the minimum x tick should be
            defaults to axis min (default: {None})
        y_0 {float} --  set to where the minimum y tick should be,
            defaults to axis min (default: {None})
        round_tick_end {bool} -- if true, round the last tick(default: {True})
        round_base {int} -- if rounding tick end, set
            nearest value to round up to (default: {5})
        hide_min {bool} -- removes the minimum value from x and y ticks (default: {False})
        round_ticks {bool} -- Whether to round value of each tick using
            default python round function (default: {False})
    """

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    if x_0 is None:
        x_0 = x_min
    if y_0 is None:
        y_0 = y_min
    if set_x_max:
        x_max = set_x_max
    if set_y_max:
        y_max = set_y_max


    if round_tick_end:
        round_to_base = lambda x: round_base * np.ceil(x/round_base)
        x_max = round_to_base(x_max)
        y_max = round_to_base(y_max)

    if n_ticks:
        x_ticks = np.linspace(x_0, x_max, n_ticks)
        y_ticks = np.linspace(y_0, y_max, n_ticks)
    else:
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        x_ticks[-1] = x_max
        y_ticks[-1] = y_max
    if hide_min:
        x_ticks = x_ticks[1:]
        y_ticks = y_ticks[1:]

    if round_ticks:
        x_ticks = [round(x) for x in x_ticks]
        y_ticks = [round(y) for y in y_ticks]

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

def rename_legend_labels(ax: mpl.axes.Axes, label_map: Dict[str, str], **legend_kwargs) -> None:
    """ Renames labels in legend based on dictionary maping old to new names
    
    Arguments:
        ax {mpl.axes.Axes}
        label_map {Dict[str, str]} -- dictionary in format {oldname: newname}
    
    Returns:
        None
    """
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [label_map[l] if l in label_map.keys() else l for l in labels]
    ax.legend(handles, new_labels, **legend_kwargs)

def change_axis_linewidth(ax: mpl.axes.Axes, linewidth: float):
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(linewidth)
    ax.tick_params(direction='out', length=10, width=linewidth)

def change_dendrogram_linewidth(clustergrid, linewidth: float):
    for a in clustergrid.ax_row_dendrogram.collections:
        a.set_linewidth(linewidth)
    for a in clustergrid.ax_col_dendrogram.collections:
        a.set_linewidth(linewidth)

def correct_heatmap_cutoff(ax: mpl.axes.Axes):
    """ Corrects for y clipping in heatmaps
    
    Arguments:
        ax {mpl.axes.Axes} -- axis with heatmap
    """
    y_lims = ax.get_ylim()
    ax.set_ylim([y_lims[0] + 0.5, y_lims[1] - .5])
