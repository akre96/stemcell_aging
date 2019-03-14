import typing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    test_input_df = pd.read_csv('output/Ania_M3000_percent-engraftment_100818_long.csv')
    analysed_cell_types = ['gr', 'b']
    percent_engraftment_min_threshold = 0.01

    threshold_filtered_df = test_input_df[test_input_df.percent_engraftment > percent_engraftment_min_threshold]
    time_points = set(threshold_filtered_df.day.values)

    clone_counts: typing.Dict = {}
    for cell_type in analysed_cell_types:
        clone_counts[cell_type] = []
        for day in time_points:
            time_point_clones = threshold_filtered_df[(threshold_filtered_df.cell_type == cell_type) & (threshold_filtered_df.day == day)]
            clone_counts[cell_type].append(len(set(time_point_clones.code.values)))
    print(clone_counts)


    fig, ax = plt.subplots()
    ind = np.arange(len(time_points))
    width = .35
    colors = ['r', 'g', 'y', 'k', 'b']
    axes = []
    labels = []
    for i, cell_type in enumerate(analysed_cell_types):
        print(cell_type)
        axes.append(ax.bar(ind + width * i, clone_counts[cell_type], width, color=colors[i])[0])
        labels.append(cell_type)

    ax.set_xticks(ind + width/(len(analysed_cell_types)))
    ax.set_xticklabels(time_points)
    ax.legend(axes, labels)
    ax.set_title('Clone count by cell type, threshold= '+str(percent_engraftment_min_threshold) +' % engraftment')
    plt.show()
    
    

if __name__ == "__main__":
    main()
