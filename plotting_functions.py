
""" Functions used to help plot data in plot_data.py

"""

from typing import List, Tuple, Dict
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pyvenn import venn
from aggregate_functions import filter_threshold, count_clones, \
    combine_enriched_clones_at_time, count_clones_at_percentile, \
    filter_mice_with_n_timepoints, filter_cell_type_threshold, \
    find_top_percentile_threshold, get_data_from_mice_missing_at_time, \
    get_max_by_mouse_timepoint, sum_abundance_by_change, find_intersect, \
    calculate_thresholds_sum_abundance

COLOR_PALETTES = json.load(open('color_palettes.json', 'r'))

def save_plot(file_name: str, save: bool, save_format: str) -> None:
    if save:
        if os.path.isdir(os.path.dirname(file_name)) or os.path.dirname(file_name) == '':
            print('Saving to: ' + file_name)
            plt.savefig(file_name, format=save_format)
        else:
            print('Directory does not exist for: ' + file_name)
            create_dir = input("Create Directory? (y/n) \n")
            if create_dir.lower() == 'y':
                os.makedirs(os.path.dirname(file_name))
                plt.savefig(file_name, format=save_format)




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
        fname = save_path + os.sep + 'clone_count_t' + str(threshold).replace('.', '-') + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)

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
                                by_mouse: bool = False,
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
        by_mouse {bool} -- wether to set units as mouse_id for lineplot
        save {bool} --  True to save a figure (default: {False})
        save_path {str} -- Path of saved output (default: {'./output'})
        save_format {str} -- Format to save output figure (default: {'png'})

    Returns:
        None -- Run plt.show() to display figures created
    """

    sns.set_palette(sns.color_palette(COLOR_PALETTES['group'][:2]))
    for month in enrichement_months:
        print('\n Month '+ str(month) +'\n')
        enriched_df = combine_enriched_clones_at_time(filtered_df, month, enrichment_thresholds, analyzed_cell_types)
        print('Number of Mice in No Change Group: '
            + str(enriched_df.loc[enriched_df.group == 'no_change'].mouse_id.nunique())
        )
        print('Number of Mice in Aging Phenotype Group: '
            + str(enriched_df.loc[enriched_df.group == 'aging_phenotype'].mouse_id.nunique())
        )
        title_addon = ''
        if month == 12:
            print('EXCLUDING MICE WITH 14 MONTH DATA')
            enriched_df = get_data_from_mice_missing_at_time(enriched_df, exclusion_timepoint=14, timepoint_column='month')
        for cell_type in analyzed_cell_types:
            plt.figure()
            plt.subplot(2, 1, 1)
            print('Plotting clones enriched at month '+str(month)+' Cell Type: ' + cell_type)
            cell_df = enriched_df.loc[enriched_df.cell_type == cell_type]
            if by_mouse:
                title_addon = 'by-mouse_'
                sns.set_palette(sns.color_palette('hls'))
                sns.lineplot(x='month',
                            y='percent_engraftment',
                            hue='mouse_id',
                            style='group',
                            data=cell_df,
                            units='code',
                            legend=False,
                            estimator=None,
                            sort=True,
                            )
            else:
                sns.lineplot(x='month',
                            y='percent_engraftment',
                            hue='group',
                            data=cell_df,
                            legend=None,
                            sort=True,
                            )
            plt.suptitle(cell_type + ' Clones with Abundance > '
                        + str(round(enrichment_thresholds[cell_type], 2))
                        + ' % WBC At Month: ' + str(month))
            plt.xlabel('')
            plt.subplot(2, 1, 2)
            sns.set_palette(sns.color_palette(COLOR_PALETTES['group'][:2]))
            ax = sns.swarmplot(x='month',
                               y='percent_engraftment',
                               hue='group',
                               data=cell_df,
                               dodge=True,
                              )
            fname = save_path \
                    + os.sep \
                    + 'dominant_clones_' + cell_type + '_' + title_addon \
                    + str(round(enrichment_thresholds[cell_type], 2)).replace('.', '-') \
                    + '_' + 'm' + str(month) + '.' + save_format
            save_plot(fname, save, save_format)

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

        fname = save_path + os.sep + 'abundance_heatmap_' + norm_label + cell + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)

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
    print('\n Group: ' + group)
    for cell_type in analysed_cell_types:
        print('\nVenn diagram for: ' + cell_type)
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
            if cell_type == 'b' and section == '1000':
                print('Month 4 Only')
                print(agg_mouse_sections)
            if section == '0001':
                print('Month 14 Only')
                print(agg_mouse_sections)

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

        fname = fname_prefix + '_median.' + save_format
        save_plot(fname, save, save_format)

def plot_lineage_average(lineage_bias_df: pd.DataFrame,
                         title_addon: str = '',
                         percentile: float = 0,
                         threshold: float = 0,
                         abundance: float = 0,
                         month: str = 'last',
                         save: bool = False,
                         save_path: str = './output',
                         save_format: str = 'png'
                        ) -> None:
    fname_prefix = save_path + os.sep + 'lineplot_bias_' + 'm' + str(month)
    if percentile:
        fname_prefix += '_p' + str(round(100*percentile, ndigits=2)).replace('.', '-')
    elif threshold:
        fname_prefix += '_t' + str(round(threshold, ndigits=2)).replace('.', '-')
    elif abundance:
        fname_prefix += '_a' + str(round(abundance, ndigits=2)).replace('.', '-')
        

    plt.figure()
    sns.lineplot(x='month', y='lineage_bias', data=group_names_pretty(lineage_bias_df), hue='group', palette=sns.color_palette(COLOR_PALETTES['group'][:2]))
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in All Mice, Overall Trend')
    plt.title(title_addon)

    fname = fname_prefix + '_average.' + save_format
    save_plot(fname, save, save_format)

def plot_lineage_bias_line(lineage_bias_df: pd.DataFrame,
                           title_addon: str = '',
                           percentile: float = 0,
                           threshold: float = 0,
                           abundance: float = 0,
                           save: bool = False,
                           save_path: str = './output',
                           save_format: str = 'png'
                          ) -> None:
    fname_prefix = save_path + os.sep + 'lineplot_bias'
    if percentile:
        fname_prefix += '_p' + str(round(100*percentile, ndigits=2)).replace('.', '-')
    elif threshold:
        fname_prefix += '_t' + str(round(threshold, ndigits=2)).replace('.', '-')
    elif abundance:
        fname_prefix += '_a' + str(round(abundance, ndigits=2)).replace('.', '-')

    plt.figure()
    sns.lineplot(x='month', y='lineage_bias', data=group_names_pretty(lineage_bias_df), hue='group', palette=sns.color_palette(COLOR_PALETTES['group'][:2]))
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
    save_plot(fname, save, save_format)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'aging_phenotype']
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', units='code', estimator=None) 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in aging_phenotype')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fname = fname_prefix + '_aging-phenotype.' + save_format
    save_plot(fname, save, save_format)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'no_change']
    sns.lineplot(x='month', y='lineage_bias', data=lineage_bias_group_df, hue='mouse_id', units='code', estimator=None) 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in no_change')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fname = fname_prefix + '_no-change.' + save_format
    save_plot(fname, save, save_format)

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

        fname = save_path + os.sep + 'clone_count_p' + str(percentile).replace('.', '-') + '_' + 'Average' + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)

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

            save_plot(fname, save, save_format)
            fname = save_path + os.sep + 'clone_count_p' + str(percentile).replace('.', '-') + '_' + cell_type + '_' + group + '.' + save_format
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

        fname = save_path + os.sep + 'clone_count_p' + str(percentile).replace('.', '-') + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)

def plot_lineage_bias_abundance_3d(lineage_bias_df: pd.DataFrame, analyzed_cell_types: List[str] = ['gr','b'], group: str = 'all'):
    fig = plt.figure()
    fig.suptitle('Group: ' + group)
    if group != 'all':
        lineage_bias_df = lineage_bias_df.loc[lineage_bias_df.group == group]
    ax = fig.add_subplot(121, projection='3d')
    for mouse_id in lineage_bias_df.mouse_id.unique():
        mouse_df = lineage_bias_df.loc[lineage_bias_df.mouse_id == mouse_id]
        ax.scatter(mouse_df.month, mouse_df.lineage_bias, mouse_df[analyzed_cell_types[0]+ '_percent_engraftment'])
        ax.set_xlabel('Month')
        ax.set_ylabel('Lineage Bias Myeloid(+)/Lymphoid(-)')
        ax.set_zlabel('Abundance in '+analyzed_cell_types[0])
    plt.title(analyzed_cell_types[0])
    ax = fig.add_subplot(122, projection='3d')
    for mouse_id in lineage_bias_df.mouse_id.unique():
        mouse_df = lineage_bias_df.loc[lineage_bias_df.mouse_id == mouse_id]
        ax.scatter(mouse_df.month, mouse_df.lineage_bias, mouse_df[analyzed_cell_types[1]+ '_percent_engraftment'])
        ax.set_xlabel('Month')
        ax.set_ylabel('Lineage Bias Myeloid(+)/Lymphoid(-)')
        ax.set_zlabel('Abundance in '+analyzed_cell_types[0])
    ax.set_xlabel('Month')
    ax.set_ylabel('Lineage Bias Myeloid(+)/Lymphoid(-)')
    ax.set_zlabel('Abundance in '+analyzed_cell_types[1])
    plt.title(analyzed_cell_types[1])
    
def plot_max_engraftment_by_group(
        input_df: pd.DataFrame,
        cell_type: str,
        title: str = '',
        percentile: float = 0,
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png'
    ) -> None:


    max_df = get_max_by_mouse_timepoint(input_df)
    max_df = max_df.loc[max_df.cell_type == cell_type]

    plt.figure()
    sns.set_palette(sns.color_palette('hls',2))

    sns.pointplot(x='month', y='percent_engraftment', hue='group', data=max_df)
    plt.suptitle('Max Engraftment of ' + cell_type)
    plt.title(title)

    if percentile:
        fname = save_path + os.sep + 'max_engraftment_p' + str(round(100*percentile, 2)).replace('.', '-') + '_' + cell_type + '.' + save_format
    else:
        fname = save_path + os.sep + 'max_engraftment' + '_' + cell_type + '.' + save_format
    save_plot(fname, save, save_format)

def plot_max_engraftment_by_mouse(input_df: pd.DataFrame, cell_type: str, group: str = 'all', title: str = '', percentile: float = 0, save: bool = False, save_path: str = '', save_format: str = 'png') -> None:
    max_df = get_max_by_mouse_timepoint(input_df)
    max_df = max_df.loc[max_df.cell_type == cell_type]
    if group != 'all':
        max_df = max_df.loc[max_df.group == group]

    plt.figure()
    sns.set_palette(sns.color_palette('hls',2))

    sns.lineplot(x='month', y='percent_engraftment', hue='mouse_id', data=max_df, legend=False)
    plt.suptitle('Max Engraftment of ' + cell_type + ' Group: ' + group)
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Max Percent Clone Abundance')
    if percentile:
        fname = save_path + os.sep + 'max_engraftment_p' + str(round(100*percentile, 2)).replace('.', '-') + '_' + cell_type + '_' + group + '.' + save_format
    else:
        fname = save_path + os.sep + 'max_engraftment' + '_' + cell_type + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_max_engraftment(input_df: pd.DataFrame, title: str = '', percentile: float = 0, save: bool = False, save_path: str = '', save_format: str = 'png') -> None:
    max_df = get_max_by_mouse_timepoint(input_df)

    plt.figure()
    sns.pointplot(x='month', y='percent_engraftment', hue='cell_type', hue_order=['gr','b'], data=max_df)
    plt.suptitle('Max Engraftment of All Mice')
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Max Percent Clone Abundance')
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
    plt.xlabel('Month')
    plt.ylabel('Max Percent Clone Abundance')
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
    plt.xlabel('Month')
    plt.ylabel('Max Percent Clone Abundance')

    if percentile:
        fname = save_path + os.sep + 'max_engraftment' + str(round(100*percentile, 2)).replace('.', '-') + '_' + group + '.' + save_format
    else:
        fname = save_path + os.sep + 'max_engraftment' + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_change_hist(bias_change_df: pd.DataFrame,
                          threshold: float,
                          absolute_value: bool = False,
                          group: str = 'all',
                          save: bool = False,
                          save_path: str = 'output',
                          save_format: str = 'png'
                         ) -> None:
    """ Plot distribution of bias change (hist + rugplot + kde)
    
    Arguments:
        bias_change_df {pd.DataFrame} -- dataframe of bias change information
        threshold {float} -- threshold that was used to filter data
    
    Keyword Arguments:
        absolute_value {bool} -- Whether plot is done on magnitude, or including direction (default: {False})
        group {str} --  Group filtered for (default: {'all'})
        save {bool} --  Wether to save plot (default: {False})
        save_path {str} -- Where to save plot (default: {'output'})
        save_format {str} --  What file format to save plot (default: {'png'})
    """

    plt.figure()
    bins = 20
    if group != 'all':
        bias_change_df = bias_change_df.loc[bias_change_df.group == group]
    if absolute_value:
        sns.distplot(bias_change_df.bias_change.abs(), bins=bins, rug=True)
    else:
        sns.distplot(bias_change_df.bias_change, bins=bins, rug=True)

    plt.title('Distribution of lineage bias change')
    plt.suptitle('Threshold: ' + str(threshold) + ' Group: ' + group)
    plt.xlabel('Magnitude of Lineage Bias Change')
    plt.ylabel('Count of Clones')

    fname = save_path + os.sep + 'bias_change_distribution_t' + str(threshold).replace('.', '-') + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_change_cutoff(bias_change_df: pd.DataFrame,
                            threshold: float,
                            absolute_value: bool = False,
                            group: str = 'all',
                            save: bool = False,
                            save_path: str = 'output',
                            save_format: str = 'png'
                           ) -> None:
    """ Plots KDE of bias change annotated with line to cut "change" vs "non change" clones
    
    Arguments:
        bias_change_df {pd.DataFrame} -- dataframe of bias change information
        threshold {float} -- threshold that was used to filter data
    
    Keyword Arguments:
        absolute_value {bool} -- Whether plot is done on magnitude, or including direction (default: {False})
        group {str} --  Group filtered for (default: {'all'})
        save {bool} --  Wether to save plot (default: {False})
        save_path {str} -- Where to save plot (default: {'output'})
        save_format {str} --  What file format to save plot (default: {'png'})
    """

    plt.figure()

    if group != 'all':
        bias_change_df = bias_change_df.loc[bias_change_df.group == group]

    if absolute_value:
        kde = sns.kdeplot(bias_change_df.bias_change.abs(), shade=True)
    else:
        kde = sns.kdeplot(bias_change_df.bias_change, shade=True)
    
    x, y = kde.get_lines()[0].get_data()
    dy = np.diff(y)/np.diff(x)
    dx = x[1:]
    cutoff_candidates: List = []
    for i, val in enumerate(dy):
        if i != 0:
            if dy[i - 1] <= 0 and dy[i] >= 0:
                cutoff_candidates.append(dx[i])
    plt.vlines(cutoff_candidates, 0, max(y))
    kde.text(cutoff_candidates[0] + .1, max(y)/2, 'Change at: ' + str(round(cutoff_candidates[0],3)))
    plt.title('Kernel Density Estimate of lineage bias change')
    plt.suptitle('Threshold: ' + str(threshold) + ' Group: ' + group)
    plt.xlabel('Magnitude of Lineage Bias Change')
    plt.ylabel('Clone Density at Change')
    kde.legend_.remove()

    fname = save_path + os.sep + 'bias_change_cutoff_t' + str(threshold).replace('.', '-') + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_lineage_bias_violin(lineage_bias_df: pd.DataFrame,
                             title_addon: str = '',
                             percentile: float = 0,
                             group: str = 'all',
                             threshold: float = 0,
                             save: bool = False,
                             save_path: str = './output',
                             save_format: str = 'png'
                            ) -> None:
    fname_prefix = save_path + os.sep + 'violin_bias'
    plt.figure()

    if percentile:
        fname_prefix += '_p' + str(round(100*percentile, ndigits=2)).replace('.', '-')
    elif threshold:
        fname_prefix += '_t' + str(round(threshold, ndigits=2)).replace('.', '-')
    if group != 'all':
        lineage_bias_df = lineage_bias_df.loc[lineage_bias_df.group == group]
        sns.violinplot(x='month', y='lineage_bias', data=lineage_bias_df, inner='stick', cut=0)
    else:
        sns.violinplot(x='month', y='lineage_bias', data=lineage_bias_df, hue='group', palette=sns.color_palette('hls', 2), inner='stick', cut=0)

    plt.xlabel('Month')
    plt.ylabel('Lineage Bias')
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias, Group: ' + group)
    plt.title(title_addon)

    fname = fname_prefix + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_contributions(
        contributions_df: pd.DataFrame,
        cell_type: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    plt.figure()
    plot = sns.lineplot(x='percentile', y='percent_sum_abundance', hue='month_str', data=contributions_df)
    plt.xlabel('Percentile by Clone Abundance')
    plt.ylabel('Percent of Tracked Clone ' + cell_type + ' Population')
    plt.title('Cumulative Abundance at Percentiles for ' + cell_type)

    m4_cont_df = contributions_df.loc[contributions_df.month == 4]

    exp_x, exp_y = find_intersect(m4_cont_df, 50)
    plt.vlines(exp_x, -5, exp_y + 5, linestyles='dashed')
    plt.hlines(exp_y, 0, exp_x + 5, linestyles='dashed')
    plt.text(0, 52, 'Expanded Clones: (' + str(round(exp_x, 2)) + ', ' + str(round(exp_y, 2)) + ')')

    dom_x, dom_y = find_intersect(m4_cont_df, 80)
    plt.vlines(dom_x, -5, dom_y + 5, linestyles=(0, (1, 1)))
    plt.hlines(dom_y, 0, dom_x + 5, linestyles=(0, (1, 1)))
    plt.text(0, 80, 'Dominant Clones: (' + str(round(dom_x, 2)) + ', ' + str(round(dom_y, 2)) + ')')

    fname = save_path + os.sep + 'percentile_abundance_contribution_' + cell_type + '.' + save_format
    save_plot(fname, save, save_format)

def plot_change_contributions(
        changed_marked_df: pd.DataFrame,
        cell_type: str,
        group: str = 'all',
        percent_of_total: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:


    plt.figure()
    changed_sum_df = sum_abundance_by_change(changed_marked_df, percent_of_total=percent_of_total)
    changed_sum_cell_df = changed_sum_df.loc[changed_sum_df.cell_type == cell_type]
    y_units = '(% WBC)'
    palette = sns.color_palette(COLOR_PALETTES['change_status_3'])
    if percent_of_total:
        y_units = '(% of Tracked ' + cell_type.capitalize() +' cells)'
        palette = sns.color_palette(COLOR_PALETTES['change_status_2'])

    if group != 'all':
        changed_sum_df = changed_sum_df.loc[changed_sum_df.group == group]

    sns.barplot(x='month', y='percent_engraftment', hue='change_status', data=changed_sum_cell_df, palette=palette)
    plt.xlabel('Month')
    plt.ylabel('Cumulative Abundance ' + y_units)
    plt.suptitle(cell_type.capitalize() + ' Changed vs Not-Changed Cumulative Abundance')
    plt.title('Group: ' + group)
    plt.gca().legend().set_title('')

    fname = save_path + os.sep + 'contribution_changed_' + cell_type + '_' + group + '.' + save_format
    if percent_of_total:
        fname = save_path + os.sep + 'percent_contribution_changed_' + cell_type + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_change_contributions_by_group(
        changed_marked_df: pd.DataFrame,
        cell_type: str,
        percent_of_total: bool = False,
        line: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:


    plt.figure()
    changed_sum_df = sum_abundance_by_change(changed_marked_df, percent_of_total=percent_of_total)
    changed_sum_df = changed_sum_df.loc[changed_sum_df.changed]
    changed_sum_cell_df = changed_sum_df.loc[changed_sum_df.cell_type == cell_type]

    print('Outlier Mice:')
    print(
        changed_sum_cell_df.loc[(changed_sum_cell_df.percent_engraftment > 60) & (changed_sum_cell_df.month == 14)].mouse_id
    )
    y_units = '(% WBC)'
    palette = sns.color_palette(COLOR_PALETTES['group'][:2])
    if percent_of_total:
        y_units = '(% of Tracked ' + cell_type.capitalize() +' cells)'

    group = 'both-groups'

    if line:
        sns.lineplot(x='month', y='percent_engraftment', hue='group', units='mouse_id', estimator=None, data=changed_sum_cell_df, palette=palette)
    else:
        sns.barplot(x='month', y='percent_engraftment', hue='group', data=changed_sum_cell_df, palette=palette, alpha=0.8)
    plt.xlabel('Month')
    plt.ylabel('Cumulative Abundance ' + y_units)
    plt.suptitle(' Cumulative Abundance of Changed ' + cell_type.capitalize() + ' Cells')
    if percent_of_total:
        plt.title('Relative to total abundance of tracked cells', fontsize=10)
    plt.gca().legend().set_title('')

    addon = 'contribution_changed_'  
    if percent_of_total:
        addon = addon + 'percent_'
    if line:
        addon = addon + 'line_'
    fname = save_path + os.sep + addon + cell_type + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)
    

def plot_weighted_bias_hist(
    lineage_bias_df: pd.DataFrame,
    cell_type: str,
    month: int = 4,
    by_group: bool = True,
    bins: int = 30,
    save: bool = False,
    save_path: str = './output',
    save_format: str = 'png'
    ) -> None:


    plt.figure()

    group = 'all'
    colors = COLOR_PALETTES['group']
    if by_group:
        group = 'by-groups'
        for i,g in enumerate(['aging_phenotype', 'no_change']):
            by_group_df = lineage_bias_df.loc[lineage_bias_df.group == g]
            month_cell_df = by_group_df.loc[(by_group_df.month == month)]
            plt.hist(
                month_cell_df.lineage_bias,
                bins=bins,
                weights=month_cell_df[cell_type+'_percent_engraftment'],
                color=colors[i],
                edgecolor=colors[i],
                label=g.replace('_', ' ').title(),
                alpha=0.7
            )
        plt.legend()
    else:
        month_cell_df = lineage_bias_df.loc[(lineage_bias_df.month == month)]
        plt.hist(month_cell_df.lineage_bias,
            bins=bins,
            weights=month_cell_df[cell_type+'_percent_engraftment'])
    plt.ylabel('Weighted Counts')
    plt.xlabel('Lineage Bias Myeloid (+)/Lymphoid (-)')
    plt.suptitle('Lineage Bias Distribution Weighted by Abundance in ' + cell_type.capitalize())
    plt.title('Month: ' + str(month))

    addon = 'bias_hist_m' + str(month) + '_'
    fname = save_path + os.sep + addon + cell_type + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)
    

def plot_counts_at_abundance(
        input_df: pd.DataFrame,
        abundance_cutoff: float,
        analyzed_cell_types: List[str] = ['gr', 'b'],
        group: str = 'all',
        line: bool = False,
        save: bool = False,
        save_path: str = 'output',
        save_format: str = 'png',
    ) -> None:
                            
    percentiles, thresholds = calculate_thresholds_sum_abundance(
        input_df,
        abundance_cutoff=abundance_cutoff
    )

    if group != 'all':
        input_df = input_df.loc[input_df.group == group]

    filter_df = filter_cell_type_threshold(input_df, thresholds, analyzed_cell_types)
    clone_counts = count_clones(filter_df)
    sns.set_palette(sns.color_palette(COLOR_PALETTES['cell_type']))
    if line:
        _, axis = plt.subplots()
        sns.lineplot(
            x='month',
            y='code',
            hue='cell_type',
            data=clone_counts,
            ax=axis,
        )
        title_string = 'Average/Mouse Clone Counts for Cells Filtered Above Cumulative Abundance Based Threshold'
        plt.suptitle(title_string)
        label = 'Group: ' + group + ', Sum Abundance: ' + str(round(100 - abundance_cutoff, ndigits=2))
        plt.title(label)
        plt.xlabel('Month')
        plt.ylabel('Number of Clones')

        fname = save_path + os.sep + 'clone_count_a' + str(abundance_cutoff).replace('.', '-') + '_' + 'Average' + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)

        for cell_type in clone_counts.cell_type.unique():
            _, axis = plt.subplots()
            clone_counts_cell = clone_counts[clone_counts.cell_type == cell_type]
            sns.lineplot(
                x='month',
                y='code',
                hue='mouse_id',
                data=clone_counts_cell,
                ax=axis,
                legend=False
            )
            if cell_type == 'Total':
                title_string = 'Total Clone Counts for Cells Filtered Above Cumulative Abundance Based Threshold'
            else:
                title_string = 'Clone Counts in ' + cell_type + ' > ' + str(round(thresholds[cell_type], ndigits=2)) + '% WBC'
            plt.suptitle(title_string)
            label = 'Group: ' + group + ', Sum Abundance: ' + str(round(100 - abundance_cutoff, ndigits=2))
            plt.title(label)
            plt.xlabel('Month')
            plt.ylabel('Number of Clones')

            fname = save_path + os.sep + 'clone_count_a' + str(abundance_cutoff).replace('.', '-') + '_' + cell_type + '_' + group + '.' + save_format
            save_plot(fname, save, save_format)
    else:
        _, axis = plt.subplots()
        sns.barplot(
            x='month',
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
        label = 'Group: ' + group + ', Sum Abundance: ' + str(round(100 - abundance_cutoff, ndigits=2))
        plt.title(label)
        plt.xlabel('Month')
        plt.ylabel('Number of Clones')
        fname = save_path + os.sep + 'clone_count_a' + str(abundance_cutoff).replace('.', '-') + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)

def group_names_pretty(input_df: pd.DataFrame) -> pd.DataFrame:
    """ Makes group names formatted 'prettily'
    
    Arguments:
        input_df {pd.DataFrame} -- Data frame with group column
    
    Returns:
        pd.DataFrame -- formatted group column dataframe
    """

    input_df = input_df.assign(group=lambda x: x.group.str.replace('_', ' ').str.title())
    return input_df

def plot_average_abundance(
        input_df: pd.DataFrame,
        cell_type: str,
        thresholds: Dict[str, float],
        by_group: bool = True,
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png'
    ) -> None:
    """ Plot average abundance of cell_type by group

    Arguments:
        input_df {pd.DataFrame} -- abundance dataframe
        cell_type {str} -- 'gr' or 'b'
        thresholds {Dict[str, float]} -- dictionary of threshold {celltype: value}

    Keyword Arguments:
        by_group {bool} -- organize by phenotype group or not (default: {True})
        save {bool} -- save file (default: {False})
        save_path {str} --  where to save file (default: {''})
        save_format {str} -- what format to save file (default: {'png'})

    Returns:
        None -- plt.show() to view graph
    """


    plt.figure()
    sns.set_palette(sns.color_palette(COLOR_PALETTES['group'][:2]))
    cell_df = input_df.loc[input_df.cell_type == cell_type]
    if by_group:
        cell_df = group_names_pretty(cell_df)
        sns.lineplot(x='month', y='percent_engraftment', hue='group', data=cell_df)
    else:
        sns.lineplot(x='month', y='percent_engraftment', hue='mouse_id', data=cell_df)

    title = 'Average Abundance of ' + cell_type.capitalize() \
          + ' with Abundance > ' \
          + str(round(thresholds[cell_type],2)) + '% WBC'
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Clone Abundance (% WBC)')
    fname = save_path + os.sep + 'average_abundance' \
            + '_' + cell_type + '_th' \
            + str(round(thresholds[cell_type],2)).replace('.','-') \
            + '.' + save_format
    save_plot(fname, save, save_format)

def swamplot_abundance_cutoff(
        input_df: pd.DataFrame,
        cell_type: str,
        abundance_cutoff: float,
        color_col: str = 'mouse_id',
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png'
    ) -> None:

    _, thresholds = calculate_thresholds_sum_abundance(
        input_df,
        abundance_cutoff=abundance_cutoff
    )
    filtered_df = filter_cell_type_threshold(
        input_df,
        thresholds,
        analyzed_cell_types=[cell_type]
    )

    plt.figure()
    sns.set_palette(sns.color_palette('hls', 18))
    sns.swarmplot(x='month', y='percent_engraftment', hue='mouse_id', data=filtered_df)

    title = 'Average Abundance of ' + cell_type.capitalize() \
          + ' with Abundance > ' \
          + str(round(thresholds[cell_type],2)) + '% WBC'
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Clone Abundance (% WBC)')
    fname = save_path + os.sep + 'swamplot_abundance' \
            + '_' + cell_type + '_a' \
            + str(round(abundance_cutoff, 2)).replace('.','-') \
            + '.' + save_format
    save_plot(fname, save, save_format)
