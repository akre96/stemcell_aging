""" Create plots from step 7 output data

Returns:
    None - Shows plots
"""

from typing import List, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvenn import venn

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
                     analyzed_cell_types: List[str],
                     group: str = 'all',
                     save: bool = False,
                     save_path: str = './output',
                     save_format: str = 'png',
                    ) -> Tuple:
    """ Plots clone counts, based on stats from count_clones function

    Arguments:
        clone_counts {pd.DataFrame} -- dictionary of statistics from data, mean and sem required
        time_points {List[int]} -- list of time points to plot
        threshold {float} -- threshold value, used in title of plot
    plt.show()
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
    plt.suptitle('Clone Counts By Cell Type with Abundance > ' + str(threshold) + ' % WBC')
    label = 'Group: ' + group
    plt.title(label)
    plt.xlabel('Month')
    plt.ylabel('Number of Clones')
    if save:
        fname = save_path + os.sep + 'clone_count_t' + str(threshold).replace('.','-') + '_' + group + '.' + save_format
        plt.savefig(fname, format=save_format)

    return (fig, axis)

def find_enriched_clones_at_time(input_df: pd.DataFrame,
                                 enrichment_month: int,
                                 enrichment_threshold: float,
                                 cell_type: str,
                                 threshold_column: str = 'percent_engraftment',
                                 ) -> pd.DataFrame:
    """ Finds clones enriched at a specific time point

    Arguments:
        input_df {pd.DataFrame} -- long format data, formatted with filter_threshold()
        enrichment_month {int} -- month of interest
        threshold {int} -- threshold for significant engraftment
        cell_type {str} -- Cell type to select for
        threshold_column {str} -- column on which to apply threshold

    Returns:
        pd.DataFrame -- [description]
    """

    filter_index = (input_df[threshold_column] > enrichment_threshold) & (input_df['month'] == enrichment_month) & (input_df['cell_type'] == cell_type)
    enriched_at_month_df = input_df.loc[filter_index]
    enriched_clones = enriched_at_month_df['code']

    cell_df = input_df.loc[input_df.cell_type == cell_type]
    should_be_empty_index = (cell_df.month == enrichment_month) & (cell_df.percent_engraftment < enrichment_threshold)
    stray_clones = cell_df[should_be_empty_index]['code'].unique()

    enriched_clones_df = cell_df.loc[(~cell_df['code'].isin(stray_clones)) & (cell_df['code'].isin(enriched_clones))]
    return enriched_clones_df


def combine_enriched_clones_at_time(input_df: pd.DataFrame, enrichement_month: int, threshold: float, analyzed_cell_types: List[str]) -> pd.DataFrame:
    all_enriched_df = pd.DataFrame()
    for cell_type in analyzed_cell_types:
        enriched_cell_df = find_enriched_clones_at_time(input_df, enrichement_month, threshold, cell_type)
        all_enriched_df = all_enriched_df.append(enriched_cell_df)
    return all_enriched_df
        

def plot_clone_count_by_thresholds(input_df: pd.DataFrame,
                                   thresholds: List[float],
                                   analysed_cell_types: List[str],
                                   group: str = 'all',
                                   save: bool = False,
                                   save_path: str = './output/'
                                  ) -> None:
    """Wrapper of plot_clone_counts to plot multiple for desired threshold values

    Arguments:
        input_df {pd.DataFrame} -- long formatted data from step7 output
        thresholds {List[float]} -- list of thresholds to plot
        analysed_cell_types {List[str]} -- cell types to consider in analysis

    Returns:
        None -- plots created, run plt.show() to observe
    """

    # Filter by group if specified
    if group != 'all':
        input_df = input_df.loc[input_df.group == group]

    # Plot at thresholds
    for thresh in thresholds:
        print('Plotting at threshold: ' + str(thresh))
        threshold_df = filter_threshold(input_df, thresh, analysed_cell_types)
        clone_counts = count_clones(threshold_df)

        plot_clone_count(clone_counts,
                         thresh,
                         analysed_cell_types,
                         group=group,
                         save=save,
                         save_path=save_path)

def plot_clone_enriched_at_time(filtered_df: pd.DataFrame,
                                enrichement_months: List[int],
                                enrichment_threshold: float,
                                analyzed_cell_types: List[str] = ['gr', 'b'],
                                group: str = 'all',
                                save: bool = False,
                                save_path: str = './output',
                                save_format: str = 'png',
                                ) -> None:
    if group != 'all':
        filtered_df = filtered_df.loc[filtered_df.group == group]
    print('Plotting group: ' + group)
    for month in enrichement_months:
        plt.figure()
        plt.subplot(2, 1, 1)
        enriched_df = combine_enriched_clones_at_time(filtered_df, month, enrichment_threshold, analyzed_cell_types)

        print('Plotting clones enriched at month '+str(month))

        sns.lineplot(x='month',
                     y='percent_engraftment',
                     hue='cell_type',
                     data=enriched_df,
                     legend='brief',
                     sort=True,
                    )
        plt.suptitle('Clones With Abundance > '
                       + str(enrichment_threshold)
                       + ' % WBC At Month: '
                       + str(month))
        plt.title('Group: ' + group)
        plt.xlabel('')
        plt.subplot(2, 1, 2)
        sns.swarmplot(x='month',
                      y='percent_engraftment',
                      hue='cell_type',
                      data=enriched_df,
                      dodge=True,
                     )
        if save:
            fname = save_path + os.sep + 'dominant_clones_t' + str(enrichment_threshold).replace('.','-') + '_' + 'm' + str(month) + '_' + group + '.' + save_format
            plt.savefig(fname, format=save_format)

def clustermap_clone_abundance(filtered_df: pd.DataFrame,
                               cell_types: List[str],
                               normalize: bool = False,
                               group: str = 'all',
                               save: bool = False,
                               save_path: str = './output',
                               save_format: str = 'png',
                              ) -> plt.axis:
    if group != 'all':
        filtered_df = filtered_df.loc[filtered_df.group == group]

    norm_val = None
    norm_label = ''
    norm_title = ''
    if normalize:
        norm_val = 0
        norm_label = 'norm_'
        norm_title = 'Row Normalized '

    for cell in cell_types:
        cell_df = filtered_df[filtered_df.cell_type == cell]
        pivot_filtered = cell_df.pivot_table(index='code',
                                             columns='month',
                                             values='percent_engraftment',
                                             fill_value=0
                                            )
        clustergrid = sns.clustermap(pivot_filtered, col_cluster=False, z_score=norm_val, method='ward')
        clustergrid.fig.suptitle(norm_title + 'Clone Abundance Change in ' + cell + ' cells, Group: ' + group)
        if save:
            fname = save_path + os.sep + 'abundance_heatmap_' + norm_label + cell + '_' + group + '.' + save_format
            plt.savefig(fname, format=save_format)

    return clustergrid


def venn_barcode_in_time(present_clones_df: pd.DataFrame,
                         analysed_cell_types: List[str],
                         group: str = 'all',
                         save: bool = False,
                         save_path: str = './output',
                         save_format: str = 'png',
                        ) -> plt.axis:

    if group != 'all':
        present_clones_df = present_clones_df.loc[present_clones_df.group == group]

    for cell_type in analysed_cell_types:
        print('Venn diagram for: ' + cell_type)
        total_labels = venn.get_labels([
            present_clones_df[(present_clones_df.month == 4) & (present_clones_df.cell_type == cell_type)].code.values,
            present_clones_df[(present_clones_df.month == 9) & (present_clones_df.cell_type == cell_type)].code.values,
            present_clones_df[(present_clones_df.month == 12) & (present_clones_df.cell_type == cell_type)].code.values,
            present_clones_df[(present_clones_df.month == 14) & (present_clones_df.cell_type == cell_type)].code.values,
        ], fill=['number'])

        labels_per_mouse = {}
        mice = present_clones_df.mouse_id.unique()
        for mouse in mice:
            mouse_present_df = present_clones_df[(present_clones_df.mouse_id == mouse) & (present_clones_df.cell_type == cell_type)]
            labels = venn.get_labels([
                mouse_present_df[mouse_present_df.month == 4].code.values,
                mouse_present_df[mouse_present_df.month == 9].code.values,
                mouse_present_df[mouse_present_df.month == 12].code.values,
                mouse_present_df[mouse_present_df.month == 14].code.values,
            ], fill=['number'])
            labels_per_mouse[mouse] = labels

        
        mean_labels = {}
        median_labels = {}
        for section in labels.keys():
            agg_mouse_sections = [int(labels_per_mouse[mouse][section]) for mouse in mice]
            mean_labels[section] = np.round(np.mean(agg_mouse_sections), decimals=1)
            median_labels[section] = np.median(agg_mouse_sections)

        fname_prefix = save_path + os.sep + 'present_clones_venn_' + cell_type + '_' + group

        _, axis_total = venn.venn4(total_labels, names=['4 Month', '9 Month', '12 Month', '14 Month'])
        axis_total.set_title(cell_type + ' Total Present Clones at Time Point, Group: '+group)
        if save:
            fname = fname_prefix + '_total.' + save_format
            plt.savefig(fname, format=save_format)

        _, axis_mean = venn.venn4(mean_labels, names=['4 Month', '9 Month', '12 Month', '14 Month'])
        axis_mean.set_title(cell_type + ' Mean Present Clones at Time Point, Group: ' + group)
        if save:
            fname = fname_prefix + '_mean.' + save_format
            plt.savefig(fname, format=save_format)

        _, axis_median = venn.venn4(median_labels, names=['4 Month', '9 Month', '12 Month', '14 Month'])
        axis_median.set_title(cell_type + ' Median Present Clones at Time Point, Group: ' + group)
        if save:
            fname = fname_prefix + '_median.' + save_format
            plt.savefig(fname, format=save_format)


def append_mouse_id_group_info(input_df: pd.DataFrame, group_info_df: pd.DataFrame) -> pd.DataFrame:
    appended_df = pd.merge(input_df, group_info_df, how='left', on=['mouse_id'])
    return appended_df

def main():
    """ Create plots
    """


    input_df = pd.read_csv('Ania_M_all_percent-engraftment_100818_long.csv')

    analysed_cell_types = ['gr', 'b']

    presence_threshold = 0.01
    present_clones_df = filter_threshold(input_df, presence_threshold, analysed_cell_types)
    
    # Venn diagram of present clones
    #venn_barcode_in_time(present_clones_df,
                         #analysed_cell_types,
                         #save=True,
                         #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Venn_Presence_At_Time',
                         #save_format='png',
                         #group='no_change'
                        #)
    #venn_barcode_in_time(present_clones_df,
                         #analysed_cell_types,
                         #save=True,
                         #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Venn_Presence_At_Time',
                         #save_format='png',
                         #group='aging_phenotype'
                        #)

    # heatmap present clones
    #clustermap_clone_abundance(present_clones_df,
                               #analysed_cell_types,
                               #normalize=True,
                               #save=True,
                               #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Heatmap_Clone_Abundance',
                               #save_format='png',
                               #group='aging_phenotype',
                              #)
    #clustermap_clone_abundance(present_clones_df,
                               #analysed_cell_types,
                               #normalize=True,
                               #save=True,
                               #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Heatmap_Clone_Abundance',
                               #save_format='png',
                               #group='no_change',
                              #)

    # Count clones by threshold
    #clone_count_thresholds = [0.01, 0.02, 0.05, 0.2, 0.5]
    #plot_clone_count_by_thresholds(present_clones_df,
                                   #clone_count_thresholds,
                                   #analysed_cell_types,
                                   #save=True,
                                   #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                   #group='aging_phenotype')
    #plot_clone_count_by_thresholds(present_clones_df,
                                   #clone_count_thresholds,
                                   #analysed_cell_types,
                                   #save=True,
                                   #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                   #group='no_change')

    # Abundant clones at specific time
    #plot_clone_enriched_at_time(present_clones_df,
                                #[4, 14],
                                #0.2,
                                #save=True,
                                #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Dominant_Clone_Abundance_Over_Time',
                                #save_format='png',
                                #group='no_change',
                               #)
    #plot_clone_enriched_at_time(present_clones_df,
                                #[4, 14],
                                #0.2,
                                #save=True,
                                #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Dominant_Clone_Abundance_Over_Time',
                                #save_format='png',
                                #group='aging_phenotype',
                               #)

    #plt.show()


if __name__ == "__main__":
    main()
