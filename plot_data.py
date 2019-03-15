""" Create plots from step 7 output data

Returns:
    None - Shows plots
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def filter_threshold(input_df: pd.DataFrame,
                     threshold: float,
                     analyzed_cell_types: List[str],
                     threshold_column: str = "percent_engraftment",
                     ) -> pd.DataFrame:
    """Filter dataframe based on numerical thresholds, adds month column

    Arguments:
        input_df {pd.DataFrame} -- Input dataframe
        threshold {float} -- minimum value to allowed in output dataframe
        analyzed_cell_types {List[str]} -- List of cells to filter for
        threshold_column {str} -- column to filter by

    Returns:
        pd.DataFrame -- thresholded dataframe
    """

    analyzed_cell_types_df = input_df.loc[input_df.cell_type.isin(analyzed_cell_types)]
    threshold_filtered_df = analyzed_cell_types_df[analyzed_cell_types_df[threshold_column] > threshold]
    threshold_filtered_df['month'] = pd.to_numeric(round(threshold_filtered_df['day']/30), downcast='integer')
    return threshold_filtered_df


def count_clones(input_df: pd.DataFrame) -> pd.DataFrame:
    """ Count unique clones per cell type

    Arguments:
        input_df {pd.DataFrame} -- long formatted step7 output

    Returns:
        pd.DataFrame -- DataFrame with columns 'mouse_id','day', 'cell_type', 'code' where
        'code' contains count of unique barcodes
    """

    clone_counts = pd.DataFrame(
        input_df.groupby(['mouse_id', 'day', 'cell_type'])['code'].nunique()
        ).reset_index()
    total_clone_counts = pd.DataFrame(input_df.groupby(['mouse_id', 'day'])['code'].nunique()).reset_index()
    total_clone_counts['cell_type'] = 'Total'
    clone_counts = clone_counts.append(total_clone_counts, sort=True)

    return clone_counts


def plot_clone_count(clone_counts: pd.DataFrame,
                     threshold: float,
                     analyzed_cell_types: List[str]
                    ) -> Tuple:
    """ Plots clone counts, based on stats from count_clones function

    Arguments:
        clone_counts {pd.DataFrame} -- dictionary of statistics from data, mean and sem required
        time_points {List[int]} -- list of time points to plot
        threshold {float} -- threshold value, used in title of plot
        analysed_cell_types {List[str]} -- list of cell types analyzed

    Returns:
        Tuple -- fig,ax for further modification if required
    """

    fig, axis = plt.subplots()
    clone_counts['month'] = pd.to_numeric(round(clone_counts['day']/30), downcast='integer')
    sns.barplot(x='month',
                y='code',
                hue='cell_type',
                hue_order=analyzed_cell_types + ['Total'],
                data=clone_counts,
                ax=axis,
                capsize=.08,
                errwidth=0.5
               )
    axis.set_title('Clone Counts By Cell Type with % Engraftment > ' + str(threshold))

    return (fig, axis)

def find_enriched_clones_at_time(input_df: pd.DataFrame,
                                 enrichement_month: int,
                                 threshold: float,
                                 threshold_column: str = 'percent_engraftment',
                                 ) -> np.array:
    """ Finds clones enriched at a specific time point

    Arguments:
        input_df {pd.DataFrame} -- long format data, formatted with filter_threshold()
        enrichement_month {int} -- month of interest
        threshold {int} -- threshold for significant engraftment
        threshold_column {str} -- column on which to apply threshold

    Returns:
        np.array -- [description]
    """

    filter_index = (input_df[threshold_column] > threshold) & (input_df['month'] == enrichement_month)
    enriched_at_month_df = input_df[filter_index]
    enriched_clones = enriched_at_month_df['code'].unique()
    return enriched_clones

def plot_clone_engraftment(input_df: pd.DataFrame) -> plt.axis:
    axis = sns.lineplot(x='month',
                        y='percent_engraftment',
                        hue='cell_type',
                        # units='code',
                        # estimator=None,
                        data=input_df,
                        legend='brief',
                        sort=True,
                        )
    return (axis)

def plot_clone_count_by_thresholds(input_df: pd.DataFrame,
                                   thresholds: List[float],
                                   analysed_cell_types: List[str]
                                  ) -> None:
    """Wrapper of plot_clone_counts to plot multiple for desired threshold values

    Arguments:
        input_df {pd.DataFrame} -- long formatted data from step7 output
        thresholds {List[float]} -- list of thresholds to plot
        threshold_column {str} -- column of input_df to apply threshold to
        analysed_cell_types {List[str]} -- cell types to consider in analysis

    Returns:
        None -- plots created, run plt.show() to observe
    """

    for thresh in thresholds:
        print('Plotting at threshold: ' + str(thresh))
        threshold_df = filter_threshold(input_df, thresh, analysed_cell_types)
        clone_counts = count_clones(threshold_df)

        plot_clone_count(clone_counts, thresh, analysed_cell_types)

def plot_clone_enriched_at_time(filtered_df: pd.DataFrame,
                                enrichement_month: int,
                                enrichment_threshold: float
                                ) -> None:
    plt.subplot(2,1,1)
    enriched_clones = find_enriched_clones_at_time(filtered_df, enrichement_month, enrichment_threshold)
    enriched_df = filtered_df[filtered_df['code'].isin(enriched_clones)]
    axis = plot_clone_engraftment(enriched_df)
    axis.set_title('Clones With % Engraftment > '
                   + str(enrichment_threshold)
                   + ' At Month: '
                   + str(enrichement_month))
    plt.subplot(2,1,2)
    sns.violinplot(x='month',
                   y='percent_engraftment',
                   hue='cell_type',
                   data=enriched_df,
                   split=True,
                   )


def main():
    """ Create plots
    """

    test_input_df = pd.read_csv('Ania_M_all_percent-engraftment_100818_long.csv')
    analysed_cell_types = ['gr', 'b']

    filtered_df = filter_threshold(test_input_df, 0.0, analysed_cell_types)
    plot_clone_enriched_at_time(filtered_df, 4, 0.01)
    # thresholds = [0.0, 0.01, 0.05, 0.1, 0.2]
    # plot_clone_count_by_thresholds(test_input_df, thresholds, analysed_cell_types)
    plt.show()


if __name__ == "__main__":
    main()
