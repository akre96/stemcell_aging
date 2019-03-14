import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_odd(x: int) -> bool:
    return x & 1

def filter_threshold(input_df: pd.DataFrame, threshold: float, threshold_column: str) -> pd.DataFrame:
    threshold_filtered_df = input_df[input_df[threshold_column] > threshold]
    return threshold_filtered_df

def count_clones(input_df: pd.DataFrame, analysed_cell_types: typing.List[str]) -> typing.Tuple:
    time_points = set(input_df.day.values)
    clone_counts: typing.Dict = {}
    for cell_type in analysed_cell_types:
        clone_counts[cell_type] = []
        for day in time_points:
            time_point_clones = input_df[(input_df.cell_type == cell_type) & (input_df.day == day)]
            clone_counts[cell_type].append(len(set(time_point_clones.code.values)))
    return (clone_counts, time_points)

def plot_clone_count(clone_counts: typing.Dict, time_points: typing.List[int], threshold: float) -> typing.Tuple:
    analysed_cell_types = clone_counts.keys()
    fig, ax = plt.subplots()

    ind = np.arange(len(time_points))
    width = .25
    colors = ['r', 'g', 'y', 'k', 'b', 'c'] # Assumes no more than 5 cell types
    axes = []
    labels = []

    # Plot individual cell types
    for i, cell_type in enumerate(analysed_cell_types):
        print(cell_type)
        axes.append(ax.bar(ind + width * i, clone_counts[cell_type], width, color=colors[i])[0]) # Used for setting legend
        labels.append(cell_type)

    # Plot Total change to clone_counts section
    total_clones=[]

    for time_point_index in range(len(time_points)):
        clones_at_timepoint = sum([clone_counts[cell_type][time_point_index] for cell_type in analysed_cell_types])
        total_clones.append(clones_at_timepoint)

    axes.append(ax.bar(ind + width * len(analysed_cell_types), total_clones, width, color=colors[len(analysed_cell_types)])[0])
    labels.append('Total')

    if is_odd(len(analysed_cell_types) + 1):
        ax.set_xticks(ind + width)
    else:
        ax.set_xticks(ind + width/(len(analysed_cell_types) + 1))
    ax.set_xticklabels(time_points)
    ax.legend(axes, labels)
    ax.set_title('Clone count by cell type, threshold= '+str(threshold) +' % engraftment')
    return (fig, ax)

def plot_clone_count_by_thresholds(input_df: pd.DataFrame, thresholds: typing.List[float], threshold_column: str, analysed_cell_types: typing.List[str]) -> None:
    for th in thresholds:
        threshold_df = filter_threshold(input_df, th, threshold_column)
        clone_count, time_points = count_clones(threshold_df, analysed_cell_types)
        plot_clone_count(clone_count, time_points, th)


def main():
    test_input_df = pd.read_csv('output/Ania_M3000_percent-engraftment_100818_long.csv')
    analysed_cell_types = ['gr', 'b']
    thresholds = [0.0, 0.01, 0.05, 0.1, 0.2]
    threshold_column = 'percent_engraftment'

    plot_clone_count_by_thresholds(test_input_df, thresholds, threshold_column, analysed_cell_types)
    plt.show()
    
    

if __name__ == "__main__":
    main()
