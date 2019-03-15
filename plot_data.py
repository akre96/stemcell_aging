""" Create plots from step 7 output data

Returns:
    None - Shows plots
"""

import typing
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def filter_threshold(input_df: pd.DataFrame, threshold: float, threshold_column: str) -> pd.DataFrame:
    """Filter dataframe based on numerical thresholds

    Arguments:
        input_df {pd.DataFrame} -- Input dataframe
        threshold {float} -- minimum value to allowed in output dataframe
        threshold_column {str} -- column to filter by

    Returns:
        pd.DataFrame -- thresholded dataframe
    """

    threshold_filtered_df = input_df[input_df[threshold_column] > threshold]
    return threshold_filtered_df

def count_clones(input_df: pd.DataFrame, analysed_cell_types: typing.List[str]) -> typing.Tuple:
    """ Count clones and generate statistics

    Arguments:
        input_df {pd.DataFrame} -- long formatted step7 output
        analysed_cell_types {typing.List[str]} -- list of cell types to analyze

    Returns:
        typing.Tuple -- cell_counts, cell_stats, time_points
    """

    time_points = sorted(set(input_df.day.values))
    mice = set(input_df.mouse_id.values)

    clone_counts: typing.Dict = {}
    for cell_type in analysed_cell_types:
        clone_counts[cell_type] = {}
        for day in time_points:
            clone_counts[cell_type][day] = []
            for mouse in mice:
                mouse_df = input_df[(input_df.mouse_id == mouse) & (input_df.cell_type == cell_type) & (input_df.day == day)]
                clone_counts[cell_type][day].append(len(set(mouse_df.code.values)))
    clone_counts['total'] = {}
    for day in time_points:
        clone_counts['total'][day] = []
        for mouse in mice:
            mouse_df = input_df[(input_df.mouse_id == mouse) & (input_df.cell_type.isin(analysed_cell_types)) & (input_df.day == day)]
            clone_counts['total'][day].append(len(set(mouse_df.code.values)))

    avg_clone_counts = {}
    sem_clone_counts = {}
    sample_size_clone_counts = {}
    for cell_type in analysed_cell_types + ['total']:
        avg_clone_counts[cell_type] = [np.mean(clone_counts[cell_type][day]) for day in time_points]
        sem_clone_counts[cell_type] = [stats.sem(clone_counts[cell_type][day]) for day in time_points]
        sample_size_clone_counts[cell_type] = [len(clone_counts[cell_type][day]) for day in time_points]

    print('Mice at day with cell type:')
    print(sample_size_clone_counts)

    clone_stats = {
        'mean': avg_clone_counts,
        'sem': sem_clone_counts,
        'n': sample_size_clone_counts,
    }

    return (clone_counts, clone_stats, time_points)

def plot_clone_count(clone_stats: typing.Dict, time_points: typing.List[int], threshold: float, analysed_cell_types: typing.List[str]) -> typing.Tuple:
    """ Plots clone counts, based on stats from count_clones function

    Arguments:
        clone_stats {typing.Dict} -- dictionary of statistics from data, mean and sem required
        time_points {typing.List[int]} -- list of time points to plot
        threshold {float} -- threshold value, used in title of plot
        analysed_cell_types {typing.List[str]} -- list of cell types analyzed

    Returns:
        typing.Tuple -- fig,ax for further modification if required
    """

    fig, axis = plt.subplots()

    ind = np.arange(len(time_points))
    width = .25
    colors = ['r', 'g', 'y', 'k', 'b', 'c'] # Assumes no more than 5 cell types
    axes = []
    labels = []
    # Plot individual cell types
    for i, cell_type in enumerate(analysed_cell_types + ['total']):
        axes.append(axis.bar(ind + width * i, clone_stats['mean'][cell_type], width, color=colors[i])[0]) # Used for setting legend
        _, caplines, _ = axis.errorbar(ind + width * i, clone_stats['mean'][cell_type], yerr=clone_stats['sem'][cell_type], lolims=True, color=colors[i][0], capsize=0, ls='None')
        caplines[0].set_marker('_')
        labels.append(cell_type)

    # checks if odd number of bars/time point
    if (len(analysed_cell_types) + 1) & 1:
        axis.set_xticks(ind + width)
    else:
        axis.set_xticks(ind + width/(len(analysed_cell_types) + 1))

    time_points_months = [int(round(x/30)) for x in time_points]
    axis.set_xticklabels(time_points_months)
    plt.xlabel('Time (months)')

    plt.ylabel('Clone count ')
    axis.legend(axes, labels)
    axis.set_title('Clone count by cell type, threshold= '+str(threshold) +' % engraftment')
    return (fig, axis)

def plot_clone_count_by_thresholds(input_df: pd.DataFrame, thresholds: typing.List[float], threshold_column: str, analysed_cell_types: typing.List[str]) -> None:
    """Wrapper of plot_clone_counts to plot multiple for desired threshold values

    Arguments:
        input_df {pd.DataFrame} -- long formatted data from step7 output
        thresholds {typing.List[float]} -- list of thresholds to plot
        threshold_column {str} -- column of input_df to apply threshold to
        analysed_cell_types {typing.List[str]} -- cell types to consider in analysis

    Returns:
        None -- plots created, run plt.show() to observe
    """

    for thresh in thresholds:
        print('Plotting at threshold: ' + str(thresh))
        threshold_df = filter_threshold(input_df, thresh, threshold_column)
        _, clone_stats, time_points = count_clones(threshold_df, analysed_cell_types)

        plot_clone_count(clone_stats, time_points, thresh, analysed_cell_types)


def main():
    """ Create plots
    """

    test_input_df = pd.read_csv('Ania_M_all_percent-engraftment_100818_long.csv')
    #test_input_df = pd.read_csv('output/step7_to_long.csv')
    analysed_cell_types = ['gr', 'b']
    thresholds = [0.0, 0.01, 0.05, 0.1, 0.2]
    threshold_column = 'percent_engraftment'

    plot_clone_count_by_thresholds(test_input_df, thresholds, threshold_column, analysed_cell_types)
    plt.show()

if __name__ == "__main__":
    main()
