""" Create plots from step 7 output data

Returns:
    None - Shows plots
"""

from typing import List, Tuple, Dict
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvenn import venn
from aggregate_functions import filter_threshold, count_clones, combine_enriched_clones_at_time, find_enriched_clones_at_time, clones_enriched_at_last_timepoint


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
        fname = save_path + os.sep + 'clone_count_t' + str(threshold).replace('.', '-') + '_' + group + '.' + save_format
        plt.savefig(fname, format=save_format)

    return (fig, axis)


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
    """ Create a Line + Swarm plot of clones dominant at specified time
    
    Arguments:
        filtered_df {pd.DataFrame} -- Step7 output put through filter_threshold()
        enrichement_months {List[int]} -- Months too look at for enrichment at,
        creates one set of plots per month
        enrichment_threshold {float} -- Cutoff for dominant cell percent_engraftment
    
    Keyword Arguments:
        analyzed_cell_types {List[str]} -- Cell types to categorize by (default: {['gr', 'b']})
        group {str} -- Phenotypic group to filter by (default: {'all'})
        save {bool} --  True to save a figure (default: {False})
        save_path {str} -- Path of saved output (default: {'./output'})
        save_format {str} -- Format to save output figure (default: {'png'})
    
    Returns:
        None -- Run plt.show() to display figures created
    """

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
                              ) -> None:
    """ Plots a clustered heatmap of clone engraftment over time by cell type

    Arguments:
        filtered_df {pd.DataFrame} -- long format output of step7 passed through filter_threshold()
        cell_types {List[str]} -- cell_types to filter by, one plot per cell_type

    Keyword Arguments:
        normalize {bool} --  Whether to normalize by rows or not (default: {False})
        group {str} -- Phenotypic group to filter by (default: {'all'})
        save {bool} --  True to save a figure (default: {False})
        save_path {str} -- Path of saved output (default: {'./output'})
        save_format {str} -- Format to save output figure (default: {'png'})

    Returns:
        None -- plt.show() to view plot
    """

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
        clustergrid = sns.clustermap(pivot_filtered,
                                     col_cluster=False,
                                     z_score=norm_val,
                                     method='ward')

        clustergrid.fig.suptitle(norm_title + 'Clone Abundance Change in ' + cell + ' cells, Group: ' + group)

        if save:
            fname = save_path + os.sep + 'abundance_heatmap_' + norm_label + cell + '_' + group + '.' + save_format
            plt.savefig(fname, format=save_format)


def venn_barcode_in_time(present_clones_df: pd.DataFrame,
                         analysed_cell_types: List[str],
                         group: str = 'all',
                         save: bool = False,
                         save_path: str = './output',
                         save_format: str = 'png',
                        ) -> None:
    """ Create venn diagrams of barcode existance in seperate time points.

    Presence counted by total, mean, and median across mice
    
    Arguments:
        present_clones_df {pd.DataFrame} -- Clones filtered for presence by filter_threshold()
        analysed_cell_types {List[str]} -- Cell types to filter by, one plot set per cell_type
    
    Keyword Arguments:
        group {str} -- Phenotypic group to filter by (default: {'all'})
        save {bool} --  True to save a figure (default: {False})
        save_path {str} -- Path of saved output (default: {'./output'})
        save_format {str} -- Format to save output figure (default: {'png'})
    
    Returns:
        None -- run plt.show() to view plots
    """

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

def plot_lineage_bias_line(lineage_bias_df: pd.DataFrame, title_addon: str = ''):
    plt.figure()
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_df, hue='group') 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in All Mice, Overall Trend')
    plt.title(title_addon)

    plt.figure()
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_df, hue='mouse_id', style='group', units='code', estimator=None)
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in All Mice by Clone')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'aging_phenotype']
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', units='code', estimator=None) 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in aging_phenotype')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'no_change']
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', units='code', estimator=None) 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in no_change')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plot_lineage_bias_swarm_by_group(lineage_bias_df: pd.DataFrame) -> None:
    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'aging_phenotype']
    ax = sns.swarmplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id',dodge=True)
    ax.legend_.remove()
    plt.title('Myeloid (+) / Lymphoid (-) Bias in aging_phenotype')

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'no_change']
    ax = sns.swarmplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', dodge=True)
    ax.legend_.remove()
    plt.title('Myeloid (+) / Lymphoid (-) Bias in no_change')


def main():
    """ Create plots

    Contains many commented out templates for generating different types of plots
    """


    input_df = pd.read_csv('Ania_M_all_percent-engraftment_100818_long.csv')

    analysed_cell_types = ['gr', 'b']

    presence_threshold = 0.01
    present_clones_df = filter_threshold(input_df, presence_threshold, analysed_cell_types)
    
    lineage_bias_df = pd.read_csv('lineage_bias_from_counts.csv')

    #threshold = .2
    #plt.figure()
    #dominant_b_4m = find_enriched_clones_at_time(lineage_bias_df,
                                                  #enrichment_month=4,
                                                  #enrichment_threshold=threshold,
                                                  #cell_type=None,
                                                  #threshold_column='b_percent_engraftment',
                                                  #lineage_bias=True,
                                                 #)
    #sns.lineplot(x='month', y='lineage_bias', data=dominant_b_4m, hue='mouse_id', units='code',style='group', estimator=None)
    #plt.title('abundant at 4 month in b cells, threshold: ' + str(threshold))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #plt.figure()
    #dominant_gr_4m = find_enriched_clones_at_time(lineage_bias_df,
                                                  #enrichment_month=4,
                                                  #enrichment_threshold=threshold,
                                                  #cell_type=None,
                                                  #threshold_column='gr_percent_engraftment',
                                                  #lineage_bias=True,
                                                 #)
    #sns.lineplot(x='month', y='lineage_bias', data=dominant_gr_4m, hue='mouse_id', units='code',style='group', estimator=None)
    #plt.title('abundant at 4 month in gr cells, threshold: ' + str(threshold))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #plt.figure()
    #dominant_b_12m = find_enriched_clones_at_time(lineage_bias_df,
                                                  #enrichment_month=12,
                                                  #enrichment_threshold=threshold,
                                                  #cell_type='b',
                                                  #threshold_column='b_percent_engraftment',
                                                  #lineage_bias=True,
                                                 #)
    #sns.lineplot(x='month', y='lineage_bias', data=dominant_b_12m, hue='mouse_id', units='code',style='group', estimator=None)
    #plt.title('abundant at 12 month in b cells, threshold: ' + str(threshold))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #plt.figure()
    #dominant_gr_12m = find_enriched_clones_at_time(lineage_bias_df,
                                                  #enrichment_month=12,
                                                  #enrichment_threshold=threshold,
                                                  #cell_type='gr',
                                                  #threshold_column='gr_percent_engraftment',
                                                  #lineage_bias=True,
                                                 #)
    #sns.lineplot(x='month', y='lineage_bias', data=dominant_gr_12m, hue='mouse_id', units='code',style='group', estimator=None)
    #plt.title('abundant at 12 month in gr cells, threshold: ' + str(threshold))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Lineage Bias Line Plots
    #filt_lineage_bias_df = clones_enriched_at_last_timepoint(lineage_bias_df, threshold=1, lineage_bias=True, cell_type='any')
    #plot_lineage_bias_line(filt_lineage_bias_df, title_addon='Filtered by clones with 1% WBC last time point in any cell type')


    #filt_lineage_bias_df = clones_enriched_at_last_timepoint(lineage_bias_df, threshold=.2, lineage_bias=True, cell_type='gr')
    #plot_lineage_bias_line(filt_lineage_bias_df, title_addon='Filtered by clones with >.2 engraftment last time point in gr cell type')

    #filt_lineage_bias_df = clones_enriched_at_last_timepoint(lineage_bias_df, threshold=.5, lineage_bias=True, cell_type='b')
    #plot_lineage_bias_line(filt_lineage_bias_df, title_addon='Filtered by clones with >.5 engraftment last time point in b cell type')

    # Lineage Bias Swarmplots
    filt_lineage_bias_df = clones_enriched_at_last_timepoint(lineage_bias_df, threshold=1, lineage_bias=True, cell_type='any')
    #plot_lineage_bias_swarm_by_group(lineage_bias_df)
    sns.scatterplot(x='month', y='lineage_bias', data=filt_lineage_bias_df, size='sum_percent_engraftment', hue='mouse_id', style='group')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

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
