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
            clones = len(set(mouse_df.code.values))
            clone_counts['total'][day].append(len(set(mouse_df.code.values)))

    
    avg_clone_counts = {}
    std_clone_counts = {}
    sample_size_clone_counts = {}
    for cell_type in analysed_cell_types + ['total']:
        avg_clone_counts[cell_type] = [np.mean(clone_counts[cell_type][day]) for day in time_points ]
        std_clone_counts[cell_type] = [np.std(clone_counts[cell_type][day]) for day in time_points ]
        sample_size_clone_counts[cell_type] = [len(clone_counts[cell_type][day]) for day in time_points ]

    print('Mice at day with cell type:')
    print(sample_size_clone_counts)

    return (avg_clone_counts, std_clone_counts, sample_size_clone_counts, time_points)

def plot_clone_count(avg_clone_counts: typing.Dict, std_clone_counts: typing.Dict, time_points: typing.List[int], threshold: float, analysed_cell_types: typing.List[str]) -> typing.Tuple:
    fig, ax = plt.subplots()

    ind = np.arange(len(time_points))
    width = .25
    colors = ['r', 'g', 'y', 'k', 'b', 'c'] # Assumes no more than 5 cell types
    axes = []
    labels = []
    # Plot individual cell types
    for i, cell_type in enumerate(analysed_cell_types + ['total']):
        axes.append(ax.bar(ind + width * i, avg_clone_counts[cell_type], width, color=colors[i])[0]) # Used for setting legend
        _ , caplines, _ = ax.errorbar(ind + width * i, avg_clone_counts[cell_type], yerr=std_clone_counts[cell_type], lolims=True, color=colors[i][0], capsize=0, ls='None')
        caplines[0].set_marker('_')
        labels.append(cell_type)

    if is_odd(len(analysed_cell_types) + 1):
        ax.set_xticks(ind + width)
    else:
        ax.set_xticks(ind + width/(len(analysed_cell_types) + 1))

    time_points_months = [int(round(x/30)) for x in time_points]
    ax.set_xticklabels(time_points_months)
    plt.xlabel('Time (months)')

    plt.ylabel('Clone count ')
    ax.legend(axes, labels)
    ax.set_title('Clone count by cell type, threshold= '+str(threshold) +' % engraftment')
    return (fig, ax)

def plot_clone_count_by_thresholds(input_df: pd.DataFrame, thresholds: typing.List[float], threshold_column: str, analysed_cell_types: typing.List[str]) -> None:
    for th in thresholds:
        print('Plotting at threshold: ' + str(th))
        threshold_df = filter_threshold(input_df, th, threshold_column)
        avg_clone_count, std_clone_count, _ , time_points = count_clones(threshold_df, analysed_cell_types)
        plot_clone_count(avg_clone_count, std_clone_count, time_points, th, analysed_cell_types)


def main():
    test_input_df = pd.read_csv('Ania_M_all_percent-engraftment_100818_long.csv')
    #test_input_df = pd.read_csv('output/step7_to_long.csv')
    analysed_cell_types = ['gr', 'b']
    thresholds = [0.0, 0.01, 0.05, 0.1, 0.2]
    threshold_column = 'percent_engraftment'

    #threshold_df = filter_threshold(test_input_df, thresholds[1], threshold_column)
    #avg_clone_count, std_clone_count, _ , time_points = count_clones(threshold_df, analysed_cell_types)
    #plot_clone_count(avg_clone_count, std_clone_count, time_points, thresholds[1], analysed_cell_types)
    plot_clone_count_by_thresholds(test_input_df, thresholds, threshold_column, analysed_cell_types)
    plt.show()
    
    

if __name__ == "__main__":
    main()
