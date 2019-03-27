""" Create plots from step 7 output data

Returns:
    None - Shows plots
"""

from typing import List, Tuple, Dict
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pyvenn import venn
from aggregate_functions import filter_threshold, count_clones, \
     combine_enriched_clones_at_time, count_clones_at_percentile, \
     clones_enriched_at_last_timepoint, filter_mice_with_n_timepoints, \
     find_top_percentile_threshold, get_data_from_mice_missing_at_time, \
     get_max_by_mouse_timepoint, filter_cell_type_threshold


def plot_clone_count(clone_counts: pd.DataFrame,
                     threshold: float,
                     analyzed_cell_types: List[str],
                     group: str = 'all',
                     line: bool = False,
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

    clone_counts['month'] = pd.to_numeric(round(clone_counts['day']/30), downcast='integer')
    if line:
        for cell_type in clone_counts.cell_type.unique():
            fig, axis = plt.subplots()
            clone_counts_cell = clone_counts[clone_counts.cell_type == cell_type]
            sns.lineplot(x='month',
                         y='code',
                         hue='mouse_id',
                         data=clone_counts_cell,
                         ax=axis,
                         legend=False
                        )
            plt.suptitle('Clone Counts in '+ cell_type +' Cells with Abundance > ' + str(threshold) + ' % WBC')
            label = 'Group: ' + group
            plt.title(label)
            plt.xlabel('Month')
            plt.ylabel('Number of Clones')
            if save:
                fname = save_path + os.sep + 'clone_count_t' + str(threshold).replace('.', '-') + '_' + cell_type + '_' + group + '.' + save_format
                print('Saving to: ' + fname)
                plt.savefig(fname, format=save_format)
    else:
        fig, axis = plt.subplots()
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
            print('Saving to: ' + fname)
            plt.savefig(fname, format=save_format)

    return (fig, axis)


def plot_clone_count_by_thresholds(input_df: pd.DataFrame,
                                   thresholds: List[float],
                                   analysed_cell_types: List[str],
                                   group: str = 'all',
                                   line: bool = False,
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
                         line=line,
                         save_path=save_path)

def plot_clone_enriched_at_time(filtered_df: pd.DataFrame,
                                enrichement_months: List[int],
                                enrichment_thresholds: Dict[str, float],
                                analyzed_cell_types: List[str] = ['gr', 'b'],
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

    sns.set_palette(sns.color_palette("hls", 2))
    for month in enrichement_months:
        print('\n Month '+ str(month) +'\n')
        enriched_df = combine_enriched_clones_at_time(filtered_df, month, enrichment_thresholds, analyzed_cell_types)
        print('Number of Mice in No Change Group: ' + str(enriched_df.loc[enriched_df.group == 'no_change'].mouse_id.nunique()))
        print('Number of Mice in Aging Phenotype Group: ' + str(enriched_df.loc[enriched_df.group == 'aging_phenotype'].mouse_id.nunique()))

        if month == 12:
            print('EXCLUDING MICE WITH 14 MONTH DATA')
            enriched_df = get_data_from_mice_missing_at_time(enriched_df, exclusion_timepoint=14, timepoint_column='month')
        for cell_type in analyzed_cell_types:
            plt.figure()
            plt.subplot(2, 1, 1)
            print('Plotting clones enriched at month '+str(month)+' Cell Type: ' + cell_type)

            cell_df = enriched_df.loc[enriched_df.cell_type == cell_type]
            sns.lineplot(x='month',
                        y='percent_engraftment',
                        hue='group',
                        data=cell_df,
                        legend='brief',
                        sort=True,
                        )
            plt.suptitle(cell_type + ' Clones with Abundance > '
                        + str(round(enrichment_thresholds[cell_type], 2))
                        + ' % WBC At Month: ' + str(month))
            plt.xlabel('')
            plt.subplot(2, 1, 2)
            ax = sns.swarmplot(x='month',
                        y='percent_engraftment',
                        hue='group',
                        data=cell_df,
                        dodge=True,
                        )
            ax.legend_.remove()
            if save:
                fname = save_path \
                        + os.sep \
                        + 'dominant_clones_' + cell_type + '_' \
                        + str(round(enrichment_thresholds[cell_type], 2)).replace('.', '-') \
                        + '_' + 'm' + str(month) + '.' + save_format
                print('Saving to: ' + fname)
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
            print('Saving to: ' + fname)
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
    print('Filtering for only mice with 4 timepoints')
    print('Length of input before: '+str(len(present_clones_df)))
    present_clones_df = filter_mice_with_n_timepoints(present_clones_df, n_timepoints=4)
    print('Length of input after: '+str(len(present_clones_df)))
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
            print('Saving to: ' + fname)
            plt.savefig(fname, format=save_format)

        _, axis_mean = venn.venn4(mean_labels, names=['4 Month', '9 Month', '12 Month', '14 Month'])
        axis_mean.set_title(cell_type + ' Mean Present Clones at Time Point, Group: ' + group)
        if save:
            fname = fname_prefix + '_mean.' + save_format
            print('Saving to: ' + fname)
            plt.savefig(fname, format=save_format)

        _, axis_median = venn.venn4(median_labels, names=['4 Month', '9 Month', '12 Month', '14 Month'])
        axis_median.set_title(cell_type + ' Median Present Clones at Time Point, Group: ' + group)
        if save:
            fname = fname_prefix + '_median.' + save_format
            print('Saving to: ' + fname)
            plt.savefig(fname, format=save_format)

def plot_lineage_bias_line(lineage_bias_df: pd.DataFrame,
                           title_addon: str = '',
                           percentile: float = 0,
                           threshold: float = 0,
                           save: bool = False,
                           save_path: str = './output',
                           save_format: str = 'png'
                          ) -> None:
    fname_prefix = save_path + os.sep + 'lineplot_bias'
    if percentile:
        fname_prefix += '_p' + str(round(100*percentile, ndigits=2)).replace('.', '-')
    elif threshold:
        fname_prefix += '_t' + str(round(threshold, ndigits=2)).replace('.', '-')

    plt.figure()
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_df, hue='group', palette=sns.color_palette('hls', 2))
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in All Mice, Overall Trend')
    plt.title(title_addon)

    fname = fname_prefix + '_all_average.' + save_format
    if save:
        print('Saving to: ' + fname)
        plt.savefig(fname, format=save_format)

    plt.figure()
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_df, hue='mouse_id', style='group', units='code', estimator=None)
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in All Mice by Clone')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fname = fname_prefix + '_all.' + save_format
    if save:
        print('Saving to: ' + fname)
        plt.savefig(fname, format=save_format)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'aging_phenotype']
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', units='code', estimator=None) 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in aging_phenotype')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fname = fname_prefix + '_aging-phenotype.' + save_format
    if save:
        print('Saving to: ' + fname)
        plt.savefig(fname, format=save_format)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'no_change']
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', units='code', estimator=None) 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in no_change')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fname = fname_prefix + '_no-change.' + save_format
    if save:
        plt.savefig(fname, format=save_format)

def plot_lineage_bias_swarm_by_group(lineage_bias_df: pd.DataFrame) -> None:
    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'aging_phenotype']
    ax = sns.swarmplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', dodge=True)
    ax.legend_.remove()
    plt.title('Myeloid (+) / Lymphoid (-) Bias in aging_phenotype')

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'no_change']
    ax = sns.swarmplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', dodge=True)
    ax.legend_.remove()
    plt.title('Myeloid (+) / Lymphoid (-) Bias in no_change')

def plot_counts_at_percentile(input_df: pd.DataFrame,
                              percentile: float = 0.9,
                              thresholds: Dict[str,float] = None,
                              analyzed_cell_types: List[str] = ['gr', 'b'],
                              group: str = 'all',
                              line: bool = False,
                              save: bool = False,
                              save_path: str = 'output',
                              save_format: str = 'png',
                             ) -> None:
    if group != 'all':
        input_df = input_df.loc[input_df.group == group]

    if not thresholds:
        thresholds = find_top_percentile_threshold(input_df, percentile, analyzed_cell_types)
    clone_counts = count_clones_at_percentile(input_df, percentile, analyzed_cell_types=analyzed_cell_types, thresholds=thresholds)

    if line:
        _, axis = plt.subplots()
        sns.lineplot(x='month',
                    y='code',
                    hue='cell_type',
                    data=clone_counts,
                    ax=axis,
                    )
        title_string = 'Average/Mouse Clone Counts for Cells Filtered Above Percentile Based Threshold'
        plt.suptitle(title_string)
        label = 'Group: ' + group + ', Percentile: ' + str(round(100 * percentile, ndigits=2))
        plt.title(label)
        plt.xlabel('Month')
        plt.ylabel('Number of Clones')

        if save:
            fname = save_path + os.sep + 'clone_count_p' + str(percentile).replace('.', '-') + '_' + 'Average' + '_' + group + '.' + save_format
            print('Saving to: ' + fname)
            plt.savefig(fname, format=save_format)
        for cell_type in clone_counts.cell_type.unique():
            _, axis = plt.subplots()
            clone_counts_cell = clone_counts[clone_counts.cell_type == cell_type]
            sns.lineplot(x='month',
                        y='code',
                        hue='mouse_id',
                        data=clone_counts_cell,
                        ax=axis,
                        legend=False
                        )
            if cell_type == 'Total':
                title_string = 'Total Clone Counts for Cells Filtered Above Percentile Based Threshold'
            else:
                title_string = 'Clone Counts in ' + cell_type + ' > ' + str(round(thresholds[cell_type], ndigits=2)) + '% WBC'
            plt.suptitle(title_string)
            label = 'Group: ' + group + ', Percentile: ' + str(round(100 * percentile, ndigits=2))
            plt.title(label)
            plt.xlabel('Month')
            plt.ylabel('Number of Clones')

            if save:
                fname = save_path + os.sep + 'clone_count_p' + str(percentile).replace('.', '-') + '_' + cell_type + '_' + group + '.' + save_format
                print('Saving to: ' + fname)
                plt.savefig(fname, format=save_format)
    else:
        _, axis = plt.subplots()
        sns.barplot(x='month',
                    y='code',
                    hue='cell_type',
                    hue_order=analyzed_cell_types + ['Total'],
                    data=clone_counts,
                    ax=axis,
                    capsize=.08,
                    errwidth=0.5
                )
        title_string = 'Clone Counts by Cell Type'
        for cell_type in analyzed_cell_types:
            title_string += ' ' + cell_type + ' > ' + str(round(thresholds[cell_type], ndigits=2)) + '% WBC'
        plt.suptitle(title_string)
        label = 'Group: ' + group + ', Percentile: ' + str(round(100 * percentile, ndigits=2))
        plt.title(label)
        plt.xlabel('Month')
        plt.ylabel('Number of Clones')
        if save:
            fname = save_path + os.sep + 'clone_count_p' + str(percentile).replace('.', '-') + '_' + group + '.' + save_format
            print('Saving to: ' + fname)
            plt.savefig(fname, format=save_format)

def plot_lineage_bias_abundance_3d(lineage_bias_df: pd.DataFrame, analyzed_cell_types: List[str] = ['gr','b'], group: str = 'all'):
    fig = plt.figure()
    fig.suptitle('Group: ' + group)
    if group != 'all':
        lineage_bias_df = lineage_bias_df.loc[lineage_bias_df.group == group]
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(lineage_bias_df.month, lineage_bias_df.lineage_bias, lineage_bias_df[analyzed_cell_types[0]+ '_percent_engraftment'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Lineage Bias Myeloid(+)/Lymphoid(-)')
    ax.set_zlabel('Abundance in '+analyzed_cell_types[0])
    plt.title(analyzed_cell_types[0])
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(lineage_bias_df.month, lineage_bias_df.lineage_bias, lineage_bias_df[analyzed_cell_types[1]+ '_percent_engraftment'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Lineage Bias Myeloid(+)/Lymphoid(-)')
    ax.set_zlabel('Abundance in '+analyzed_cell_types[1])
    plt.title(analyzed_cell_types[1])
    
def plot_max_engraftment(input_df: pd.DataFrame, title: str = '', percentile: float = 0, save: bool = False, save_path: str = '', save_format: str = 'png') -> None:
    max_df = get_max_by_mouse_timepoint(input_df)

    plt.figure()
    sns.pointplot(x='month', y='percent_engraftment', hue='cell_type', hue_order=['gr','b'], data=max_df)
    plt.suptitle('Max Engraftment of All Mice')
    plt.title(title)
    if save:
        if percentile:
            fname = save_path + os.sep + 'max_engraftment' + str(round(100*percentile, 2)).replace('.', '-') + '_all' + '.' + save_format
        else:
            fname = save_path + os.sep + 'max_engraftment' + '_all' + '.' + save_format
        print('Saving to: ' + fname)
        plt.savefig(fname, format=save_format)

    plt.figure()
    group = 'no_change'
    sns.pointplot(x='month', y='percent_engraftment', hue='cell_type', hue_order=['gr','b'], data=max_df.loc[max_df.group == group])
    plt.suptitle('Max Engraftment of ' + group)
    plt.title(title)
    if save:
        if percentile:
            fname = save_path + os.sep + 'max_engraftment' + str(round(100*percentile, 2)).replace('.', '-') + '_' + group + '.' + save_format
        else:
            fname = save_path + os.sep + 'max_engraftment' + '_' + group + '.' + save_format
        print('Saving to: ' + fname)
        plt.savefig(fname, format=save_format)

    plt.figure()
    group = 'aging_phenotype'
    sns.pointplot(x='month', y='percent_engraftment', hue='cell_type', hue_order=['gr','b'], data=max_df.loc[max_df.group == group])
    plt.suptitle('Max Engraftment of ' + group)
    plt.title(title)
    if save:
        if percentile:
            fname = save_path + os.sep + 'max_engraftment' + str(round(100*percentile, 2)).replace('.', '-') + '_' + group + '.' + save_format
        else:
            fname = save_path + os.sep + 'max_engraftment' + '_' + group + '.' + save_format
        print('Saving to: ' + fname)
        plt.savefig(fname, format=save_format)

def main():
    """ Create plots set options via command line arguments

    Available graph types:
        default:            Subject to change based on what is being actively developed
        cluster:            Clustered heatmap of present clone engraftment
        venn:               Venn Diagram of clone existance at timepoint
        clone_count:        Bar charts of clone counts by cell type at different thresholds
        lineage_bias_line:  lineplots of lineage bias over time at different abundance from last timepoint
        top_perc_bias:      line plot of lineage bias over time with top percentile of clones by abundance during last time point
        engraftment_time:   lineplot/swarmplot of abundance of clones with high values at 4, 12, and 14 months
        counts_at_perc:     line or barplot of clone counts where cell-types are filtered at 90th percentile of abundance
        bias_time_abund:    3d plot of lineage bias vs time vs abundance in b and gr cells

    """

    parser = argparse.ArgumentParser(description="Plot input data")
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing long format step7 output', default='Ania_M_all_percent-engraftment_100818_long.csv')
    parser.add_argument('-l', '--lineage-bias', dest='lineage_bias', help='Path to csv containing lineage bias data', default='lineage_bias_from_counts.csv')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='output/Graphs')
    parser.add_argument('-s', '--save', dest='save', help='Set flag if you want to save output graphs', action="store_true")
    parser.add_argument('-g', '--graph', dest='graph_type', help='Type of graph to output', default='default')
    parser.add_argument('-p', '--options', dest='options', help='Graph Options', default='default')

    args = parser.parse_args()
    input_df = pd.read_csv(args.input)
    lineage_bias_df = pd.read_csv(args.lineage_bias)

    analysed_cell_types = ['gr', 'b']

    presence_threshold = 0.01
    present_clones_df = filter_threshold(input_df, presence_threshold, analysed_cell_types)
    all_clones_df = filter_threshold(input_df, 0.0, analysed_cell_types)
    graph_type = args.graph_type


    if args.save:
        print('\n **Saving Plots Enabled** \n')

    if graph_type == 'max_engraftment':
        plot_max_engraftment(present_clones_df, title='All Present Clones')

        percentile = .95
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)
        filtered_df = filter_cell_type_threshold(present_clones_df, thresholds=dominant_thresholds, analyzed_cell_types=['gr', 'b'])
        plot_max_engraftment(filtered_df,
                             title='Filtered by gr > '
                             + str(round(dominant_thresholds['gr'], 2))
                             + ', b > '
                             + str(round(dominant_thresholds['b'], 2))
                             + ' % WBC, Percentile: '
                             + str(round(100*percentile, 2)),
                             percentile=percentile,
                             save=args.save,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Max_Engraftment',
        )


    if graph_type == 'bias_time_abund':
        plot_lineage_bias_abundance_3d(lineage_bias_df)
        plot_lineage_bias_abundance_3d(lineage_bias_df, group='aging_phenotype')
        plot_lineage_bias_abundance_3d(lineage_bias_df, group='no_change')

    if graph_type == 'counts_at_perc':
        percentile = .95
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)

        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')
        line = ((args.options == 'line') | ( args.options == 'default'))
        plot_counts_at_percentile(present_clones_df,
                                  percentile=percentile,
                                  thresholds=dominant_thresholds,
                                  analyzed_cell_types=analysed_cell_types,
                                  save=args.save,
                                  line=line,
                                  save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                  group='aging_phenotype',
                                 )
        plot_counts_at_percentile(present_clones_df,
                                  percentile=percentile,
                                  thresholds=dominant_thresholds,
                                  analyzed_cell_types=analysed_cell_types,
                                  save=args.save,
                                  line=line,
                                  save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                  group='no_change',
                                 )
        plot_counts_at_percentile(present_clones_df,
                                  percentile=percentile,
                                  thresholds=dominant_thresholds,
                                  analyzed_cell_types=analysed_cell_types,
                                  save=args.save,
                                  line=line,
                                  save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                  group='all',
                                 )

    # Venn diagram of present clones
    if graph_type == 'venn':
        venn_barcode_in_time(present_clones_df,
                             analysed_cell_types,
                             save=args.save,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Venn_Presence_At_Time',
                             save_format='png',
                             group='no_change'
                            )
        venn_barcode_in_time(present_clones_df,
                             analysed_cell_types,
                             save=args.save,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Venn_Presence_At_Time',
                             save_format='png',
                             group='aging_phenotype'
                            )
        venn_barcode_in_time(present_clones_df,
                             analysed_cell_types,
                             save=args.save,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Venn_Presence_At_Time',
                             save_format='png',
                             group='all'
                            )
    # heatmap present clones
    if graph_type == 'cluster':
        clustermap_clone_abundance(present_clones_df,
                                   analysed_cell_types,
                                   normalize=True,
                                   save=args.save,
                                   save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Heatmap_Clone_Abundance',
                                   save_format='png',
                                   group='aging_phenotype',
                                  )
        clustermap_clone_abundance(present_clones_df,
                                   analysed_cell_types,
                                   normalize=True,
                                   save=args.save,
                                   save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Heatmap_Clone_Abundance',
                                   save_format='png',
                                   group='no_change',
                                  )

    # Count clones by threshold
    if graph_type == 'clone_count_bar':
        clone_count_thresholds = [0.01]
        plot_clone_count_by_thresholds(present_clones_df,
                                       clone_count_thresholds,
                                       analysed_cell_types,
                                       save=args.save,
                                       save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                       group='all')
        plot_clone_count_by_thresholds(present_clones_df,
                                       clone_count_thresholds,
                                       analysed_cell_types,
                                       save=args.save,
                                       save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                       group='aging_phenotype')
        plot_clone_count_by_thresholds(present_clones_df,
                                       clone_count_thresholds,
                                       analysed_cell_types,
                                       save=args.save,
                                       save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                       group='no_change')

    # Clone counts by threshold as lineplot
    if graph_type == 'clone_count_line':
        clone_count_thresholds = [0.01]
        plot_clone_count_by_thresholds(present_clones_df,
                                       clone_count_thresholds,
                                       analysed_cell_types,
                                       save=args.save,
                                       line=True,
                                       save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time')
        plot_clone_count_by_thresholds(present_clones_df,
                                       clone_count_thresholds,
                                       analysed_cell_types,
                                       save=args.save,
                                       line=True,
                                       save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                       group='aging_phenotype')
        plot_clone_count_by_thresholds(present_clones_df,
                                       clone_count_thresholds,
                                       analysed_cell_types,
                                       save=args.save,
                                       line=True,
                                       save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                       group='no_change')

    # Lineage Bias Line Plots by percentile
    if graph_type == 'top_perc_bias':
        if args.options == 'default':
            percentile = .995
        else:
            percentile = float(args.options)
        print('Percentile set to: ' + str(percentile))
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)

        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')

        filt_lineage_bias_b_df = clones_enriched_at_last_timepoint(input_df=input_df,
                                                                 lineage_bias_df=lineage_bias_df,
                                                                 thresholds=dominant_thresholds,
                                                                 lineage_bias=True,
                                                                 cell_type='gr',
        )
        filt_lineage_bias_gr_df = clones_enriched_at_last_timepoint(input_df=input_df,
                                                                 lineage_bias_df=lineage_bias_df,
                                                                 thresholds=dominant_thresholds,
                                                                 lineage_bias=True,
                                                                 cell_type='b',
        )
        plot_lineage_bias_line(filt_lineage_bias_gr_df,
                               title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['gr'], 2)) + '% WBC abundance in GR at last timepoint',
                               save=args.save,
                               save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/gr',
                               save_format='png',
                               percentile=percentile
                              )
        plot_lineage_bias_line(filt_lineage_bias_b_df,
                               title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['b'], 2)) + '% WBC abundance in B at last timepoint',
                               save=args.save,
                               save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/b',
                               save_format='png',
                               percentile=percentile
                              )

    # Lineage Bias Line Plots by threshold
    if graph_type == 'lineage_bias_line':
        threshold = 1
        filt_lineage_bias_df = clones_enriched_at_last_timepoint(input_df=input_df,
                                                                 lineage_bias_df=lineage_bias_df,
                                                                 thresholds={'any': threshold},
                                                                 lineage_bias=True,
                                                                 cell_type='any')
        plot_lineage_bias_line(filt_lineage_bias_df,
                               title_addon='Filtered by clones with Abundance >' + str(round(threshold, 2)) + '% WBC at last timepoint',
                               save=args.save,
                               save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot',
                               save_format='png',
                              )

    # Abundant clones at specific time
    if graph_type == 'engraftment_time':
        percentile=0.95
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)

        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')

        plot_clone_enriched_at_time(all_clones_df,
                                    [4, 12, 14],
                                    dominant_thresholds,
                                    save=args.save,
                                    save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Dominant_Clone_Abundance_Over_Time',
                                    save_format='png',
                                   )
    
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
