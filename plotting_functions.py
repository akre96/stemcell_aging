
""" Functions used to help plot data in plot_data.py

"""

from typing import List, Tuple, Dict, Any
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
    calculate_thresholds_sum_abundance, filter_lineage_bias_anytime, \
    across_gen_bias_change, between_gen_bias_change, calculate_abundance_change, \
    day_to_gen, calculate_bias_change, filter_biased_clones_at_timepoint, \
    bias_clones_to_abundance, filter_stable_initially
from lineage_bias import get_bias_change, calc_bias
from intersection.intersection import intersection

COLOR_PALETTES = json.load(open('color_palettes.json', 'r'))

def y_col_to_title(y_col: str) -> str:
    y_title = y_col.replace('_', ' ').replace(
        'percent engraftment',
        'Abundance (%WBC)'
    ).title()
    return y_title

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
                     by_day: bool = False,
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

    x_var = 'day'
    if not by_day:
        clone_counts['month'] = pd.to_numeric(round(clone_counts['day']/30), downcast='integer')
        x_var = 'month'

    if line:
        for cell_type in clone_counts.cell_type.unique():
            fig, axis = plt.subplots()
            clone_counts_cell = clone_counts[clone_counts.cell_type == cell_type]
            sns.lineplot(x=x_var,
                         y='code',
                         hue='mouse_id',
                         palette=COLOR_PALETTES['mouse_id'],
                         data=clone_counts_cell,
                         ax=axis,
                         legend=False
                        )
            plt.suptitle('Clone Counts in '+ cell_type +' Cells with Abundance > ' + str(threshold) + ' % WBC')
            label = 'Group: ' + group
            plt.title(label)
            plt.xlabel(x_var.title())
            plt.ylabel('Number of Clones')
            if save:
                fname = save_path + os.sep + 'clone_count_t' + str(threshold).replace('.', '-') + '_' + cell_type + '_' + group + '.' + save_format
                print('Saving to: ' + fname)
                plt.savefig(fname, format=save_format)
    else:
        fig, axis = plt.subplots()
        sns.barplot(x=x_var,
                    y='code',
                    hue='cell_type',
                    hue_order=analyzed_cell_types + ['Total'],
                    data=clone_counts,
                    ax=axis,
                    capsize=.08,
                    errwidth=0.5,
                    palette=sns.color_palette(COLOR_PALETTES['cell_type'])
                   )
        plt.suptitle('Clone Counts By Cell Type with Abundance > ' + str(threshold) + ' % WBC')
        label = 'Group: ' + group
        plt.title(label)
        plt.xlabel(x_var.title())
        plt.ylabel('Number of Clones')
        fname = save_path + os.sep + 'clone_count_t' + str(threshold).replace('.', '-') + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)

    return (fig, axis)


def plot_clone_count_by_thresholds(input_df: pd.DataFrame,
                                   thresholds: List[float],
                                   analysed_cell_types: List[str],
                                   by_day: bool = False,
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
                         by_day=by_day,
                         save_path=save_path)

def plot_clone_enriched_at_time(filtered_df: pd.DataFrame,
                                enrichement_months: List[int],
                                enrichment_thresholds: Dict[str, float],
                                analyzed_cell_types: List[str] = ['gr', 'b'],
                                by_clone: bool = False,
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
        by_clone {bool} -- wether to set units as code for lineplot
        save {bool} --  True to save a figure (default: {False})
        save_path {str} -- Path of saved output (default: {'./output'})
        save_format {str} -- Format to save output figure (default: {'png'})

    Returns:
        None -- Run plt.show() to display figures created
    """

    for month in enrichement_months:
        print('\n Month '+ str(month) +'\n')
        enriched_df = combine_enriched_clones_at_time(
            filtered_df,
            enrichment_time=month,
            timepoint_col='month',
            thresholds=enrichment_thresholds,
            analyzed_cell_types=analyzed_cell_types,
            )
        print(
            'Number of Mice in No Change Group: '
            + str(enriched_df.loc[enriched_df.group == 'no_change'].mouse_id.nunique())
        )
        print(
            'Number of Mice in Aging Phenotype Group: '
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
            if by_clone:
                title_addon = 'by-clone_'
                sns.set_palette(sns.color_palette('hls'))
                sns.lineplot(
                    x='month',
                    y='percent_engraftment',
                    hue='mouse_id',
                    style='group',
                    data=cell_df,
                    units='code',
                    legend=False,
                    estimator=None,
                    sort=True,
                    palette=COLOR_PALETTES['mouse_id']
                )
            else:
                sns.lineplot(
                    x='month',
                    y='percent_engraftment',
                    hue='group',
                    data=cell_df,
                    legend=None,
                    sort=True,
                    palette=COLOR_PALETTES['group']
                )
            plt.suptitle(cell_type.title() + ' Clones with Abundance > '
                        + str(round(enrichment_thresholds[cell_type], 2))
                        + ' % WBC At Month: ' + str(month))
            plt.xlabel('')
            plt.ylabel('Abundance (%WBC)')
            plt.subplot(2, 1, 2)
            ax = sns.swarmplot(
                x='month',
                y='percent_engraftment',
                hue='group',
                data=cell_df,
                dodge=True,
                palette=COLOR_PALETTES['group']
            )
            plt.ylabel('Abundance (%WBC)')
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
                         timepoint: str = 'last',
                         timepoint_col: str = 'month',
                         y_col: str = 'lineage_bias',
                         by_clone: bool = False,
                         save: bool = False,
                         save_path: str = './output',
                         save_format: str = 'png'
                        ) -> None:
    fname_prefix = save_path + os.sep + 'lineplot_abundance_time_' + timepoint_col + str(timepoint)
    if percentile:
        fname_prefix += '_p' + str(round(100*percentile, ndigits=2)).replace('.', '-')
    elif threshold:
        fname_prefix += '_t' + str(round(threshold, ndigits=2)).replace('.', '-')
    elif abundance:
        fname_prefix += '_a' + str(round(abundance, ndigits=2)).replace('.', '-')

    if by_clone:
        fname_prefix += '_by-clone'
        for phenotype, group in lineage_bias_df.groupby('group'):
            plt.figure()
            sns.lineplot(
                x=timepoint_col,
                y=y_col,
                data=group_names_pretty(group),
                hue='mouse_id',
                legend=False,
                palette=COLOR_PALETTES['mouse_id'],
                units='code',
                estimator=None
            )
            plt.xlabel(timepoint_col.title())
            y_title = y_col.replace('percent_engraftment', 'Abundance (% WBC)').replace('_', ' ').title()
            plt.ylabel(y_title)
            plt.suptitle(
                'Group: ' \
                + phenotype.replace('_', ' ').title() \
                + ' Mice'
            )
            plt.title(title_addon)
            fname = fname_prefix \
                + '_' + phenotype \
                + '.' + save_format
            save_plot(fname, save, save_format)

    else:
        plt.figure()
        sns.lineplot(
            x=timepoint_col,
            y=y_col,
            data=group_names_pretty(lineage_bias_df),
            hue='group',
            palette=COLOR_PALETTES['group']
        )
        plt.suptitle('All Mice, Overall Trend')
        plt.title(title_addon)
        plt.xlabel(timepoint_col.title())
        y_title = y_col.replace('percent_engraftment', 'Abundance (% WBC)').replace('_', ' ').title()
        plt.ylabel(y_title)

        fname = fname_prefix + '_average.' + save_format
        save_plot(fname, save, save_format)

def plot_lineage_bias_line(
    lineage_bias_df: pd.DataFrame,
    clonal_abundance_df: pd.DataFrame = None,
    title_addon: str = '',
    percentile: float = 0,
    threshold: float = 0,
    abundance: float = 0,
    y_col: str = 'lineage_bias',
    by_day: bool = False,
    timepoint_col: str = None,
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

    x_var = 'month'
    if by_day:
        x_var = 'day'
    if timepoint_col is not None:
        x_var = timepoint_col

    plt.figure()
    sns.lineplot(
        x=x_var,
        y=y_col,
        data=group_names_pretty(lineage_bias_df),
        hue='group',
        palette=COLOR_PALETTES['group']
    )
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in All Mice, Overall Trend')
    plt.title(title_addon)

    fname = fname_prefix + '_all_average.' + save_format
    save_plot(fname, save, save_format)

    plt.figure()
    sns.lineplot(
        x=x_var,
        y=y_col,
        data=lineage_bias_df,
        hue='mouse_id',
        style='group',
        units='code',
        estimator=None,
        palette=COLOR_PALETTES['mouse_id']
    )
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in All Mice by Clone')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fname = fname_prefix + '_all.' + save_format
    save_plot(fname, save, save_format)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'aging_phenotype']
    sns.lineplot(
        x=x_var,
        y=y_col,
        data=lineage_bias_group_df,
        hue='mouse_id',
        units='code',
        estimator=None,
        palette=COLOR_PALETTES['mouse_id']
    ) 
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias in aging_phenotype')
    plt.title(title_addon)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fname = fname_prefix + '_aging-phenotype.' + save_format
    save_plot(fname, save, save_format)

    plt.figure()
    lineage_bias_group_df = lineage_bias_df.loc[lineage_bias_df.group == 'no_change']
    sns.lineplot(
        x=x_var,
        y=y_col,
        data=lineage_bias_group_df,
        hue='mouse_id',
        units='code',
        estimator=None,
        palette=COLOR_PALETTES['mouse_id']
    ) 
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
                              thresholds: Dict[str, float] = None,
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

def plot_lineage_bias_abundance_3d(
        lineage_bias_df: pd.DataFrame,
        analyzed_cell_types: List[str] = ['gr','b'],
        group: str = 'all',
        by_day: bool = False,
    ) -> None:
    fig = plt.figure()
    fig.suptitle('Group: ' + group)
    x_var = 'month'
    if by_day:
        x_var = 'day'
    if group != 'all':
        lineage_bias_df = lineage_bias_df.loc[lineage_bias_df.group == group]

    ax = fig.add_subplot(121, projection='3d')
    for mouse_id in lineage_bias_df.mouse_id.unique():
        mouse_df = lineage_bias_df.loc[lineage_bias_df.mouse_id == mouse_id]
        ax.scatter(mouse_df[x_var], mouse_df.lineage_bias, mouse_df[analyzed_cell_types[0]+ '_percent_engraftment'])
        ax.set_xlabel(x_var.title())
        ax.set_ylabel('Lineage Bias Myeloid(+)/Lymphoid(-)')
        ax.set_zlabel('Abundance in '+analyzed_cell_types[0])
    plt.title(analyzed_cell_types[0])
    ax = fig.add_subplot(122, projection='3d')
    for mouse_id in lineage_bias_df.mouse_id.unique():
        mouse_df = lineage_bias_df.loc[lineage_bias_df.mouse_id == mouse_id]
        ax.scatter(mouse_df[x_var], mouse_df.lineage_bias, mouse_df[analyzed_cell_types[1]+ '_percent_engraftment'])
        ax.set_xlabel(x_var.title())
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


def plot_lineage_bias_violin(lineage_bias_df: pd.DataFrame,
                             timepoint_col: str,
                             title_addon: str = '',
                             percentile: float = 0,
                             group: str = 'all',
                             threshold: float = 0,
                             save: bool = False,
                             save_path: str = './output',
                             save_format: str = 'png',
                            ) -> None:
    """ Creats violin plot of lineage bias over time
    
    Arguments:
        lineage_bias_df {pd.DataFrame} -- lineage bias data frame
    
    Keyword Arguments:
        title_addon {str} -- description for title (default: {''})
        percentile {float} -- percentil to look at (default: {0})
        group {str} -- group to subselect (default: {'all'})
        threshold {float} --  threshold analyzed (default: {0})
        save {bool} -- (default: {False})
        save_path {str} -- (default: {'./output'})
        save_format {str} -- (default: {'png'})
        by_day {bool} -- Whether to plot by day instead of month (default: {False})
    
    Returns:
        None -- [description]
    """

    fname_prefix = save_path + os.sep + 'violin_bias'
    plt.figure()

    x_var = timepoint_col
    if percentile:
        fname_prefix += '_p' + str(round(100*percentile, ndigits=2)).replace('.', '-')
    elif threshold:
        fname_prefix += '_t' + str(round(threshold, ndigits=2)).replace('.', '-')
    if group != 'all':
        lineage_bias_df = lineage_bias_df.loc[lineage_bias_df.group == group]
        sns.violinplot(x=x_var, y='lineage_bias', data=lineage_bias_df, inner='stick', cut=0)
    else:
        sns.violinplot(x=x_var, y='lineage_bias', data=lineage_bias_df, hue='group', palette=COLOR_PALETTES['group'], inner='stick', cut=0)

    plt.xlabel(x_var.title())
    plt.ylabel('Lineage Bias')
    plt.suptitle('Myeloid (+) / Lymphoid (-) Bias, Group: ' + group)
    plt.title(title_addon)

    fname = fname_prefix + '_' + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_contributions(
        contributions_df: pd.DataFrame,
        cell_type: str,
        by_day: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    """ Plot cumulative abundance at percentile, one line per timepoint
    Also adds lines to see intersection where cells contribute 50% and 20% 
    of cumulative abundance
    
    Arguments:
        contributions_df {pd.DataFrame} -- percentile to cumulative abundance df
        cell_type {str} -- cell type to look at
    
    Keyword Arguments:
        by_day {bool} -- by day timepoint instead of month (default: {False})
        save {bool} -- (default: {False})
        save_path {str} -- (default: {'./output'})
        save_format {str} -- (default: {'png'})
    
    Returns:
        None -- plt.show() to view plot
    """


    plt.figure()
    plot = sns.lineplot(
        x='percentile',
        y='percent_sum_abundance',
        hue='time_str',
        data=contributions_df.sort_values(by='day'),
        palette=sns.color_palette(COLOR_PALETTES['time_point'], n_colors=contributions_df.time_str.nunique())
    )
    plt.xlabel('Percentile by Clone Abundance')
    plt.ylabel('Percent of Tracked Clone ' + cell_type + ' Population')
    plt.title('Cumulative Abundance at Percentiles for ' + cell_type)

    if by_day:
        m4_cont_df = contributions_df.loc[contributions_df.day == contributions_df.day.min()]
        print('min day: ' + str(contributions_df.day.min()))
    else:
        m4_cont_df = contributions_df.loc[contributions_df.month == 4]

    exp_x, exp_y = find_intersect(m4_cont_df, 50)
    plt.vlines(exp_x, -5, exp_y + 5, linestyles='dashed')
    plt.hlines(exp_y, 0, exp_x + 5, linestyles='dashed')
    plt.text(25, 52, 'Expanded Clones: (' + str(round(exp_x, 2)) + ', ' + str(round(exp_y, 2)) + ')')

    dom_x, dom_y = find_intersect(m4_cont_df, 80)
    plt.vlines(dom_x, -5, dom_y + 5, linestyles=(0, (1, 1)))
    plt.hlines(dom_y, 0, dom_x + 5, linestyles=(0, (1, 1)))
    plt.text(30, 80, 'Dominant Clones: (' + str(round(dom_x, 2)) + ', ' + str(round(dom_y, 2)) + ')')

    fname = save_path + os.sep + 'percentile_abundance_contribution_' + cell_type + '.' + save_format
    save_plot(fname, save, save_format)

def plot_change_contributions(
        changed_marked_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    """ Plot contribution of changed cells
    
    Arguments:
        changed_marked_df {pd.DataFrame} -- bias_change with clones marked as changed
    
    Keyword Arguments:
        group {str} -- group to filter for (default: {'all'})
        percent_of_total {bool} -- plot as percent of tracked cell (default: {False})
        save {bool} -- (default: {False})
        save_path {str} -- (default: {'./output'})
        save_format {str} -- (default: {'png'})
    
    Returns:
        None -- plt.show() to view plot
    """


    if timepoint_col == 'gen':
        changed_marked_df = changed_marked_df[changed_marked_df.gen != 8.5]
    percent_of_total = True
    changed_sum_df = sum_abundance_by_change(
        changed_marked_df,
        timepoint_col=timepoint_col,
        percent_of_total=percent_of_total
    )

    for cell_type, cell_df in changed_sum_df.groupby('cell_type'):
        last_time_per_mouse_df = pd.DataFrame(columns=changed_sum_df.columns)
        first_time_per_mouse_df = pd.DataFrame(columns=changed_sum_df.columns)
        for label, g_df in cell_df.groupby(['mouse_id', 'change_status', 'change_type']):
            if label[1] == 'Unchanged':
                continue
            sort_df = g_df.sort_values(by=timepoint_col)
            last_time_per_mouse_df = last_time_per_mouse_df.append(sort_df.iloc[-1], ignore_index=True)
            first_time_per_mouse_df = first_time_per_mouse_df.append(sort_df.iloc[0])

        last_time_per_mouse_df = last_time_per_mouse_df.sort_values(by='percent_engraftment', ascending=False)
        first_time_per_mouse_df = first_time_per_mouse_df.sort_values(by='percent_engraftment', ascending=False)
        last_time_per_mouse_df = last_time_per_mouse_df.assign(total=100)
        first_time_per_mouse_df = first_time_per_mouse_df.assign(total=100)

        sns.set(style="whitegrid")

        for group, g_df in last_time_per_mouse_df.groupby('group'):
            for m, m_df in g_df.groupby('mouse_id'):
                lymph = m_df[m_df.change_type == 'Lymphoid']
                myl = m_df[m_df.change_type == 'Myeloid']
                g_df[(g_df.mouse_id == m) & (g_df.change_type == 'Myeloid')].percent_engraftment = myl.percent_engraftment + lymph.percent_engraftment

            plt.figure()
            sns.barplot(
                x='total',
                y='mouse_id',
                data=g_df,
                color=COLOR_PALETTES['change_status']['Unchanged'],
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Myeloid'],
                color=COLOR_PALETTES['change_type']['Myeloid'],
                label='Myeloid'
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Lymphoid'],
                color=COLOR_PALETTES['change_type']['Lymphoid'],
                label='Lymphoid'
            )
            plt.xlabel('Contribution of Changed Cells')
            plt.ylabel('')
            plt.suptitle(cell_type.title() + ' Changed vs Not-Changed Cumulative Abundance')
            plt.title('Group: ' + group.replace('_', ' ').title())
            plt.gca().legend(title='Change Direction', loc='lower right')
            sns.despine(left=True, bottom=True)

            fname = save_path + os.sep + 'percent_contribution_changed_' + cell_type + '_' + group + '.' + save_format
            save_plot(fname, save, save_format)

        for group, g_df in first_time_per_mouse_df.groupby('group'):
            for m, m_df in g_df.groupby('mouse_id'):
                lymph = m_df[m_df.change_type == 'Lymphoid']
                myl = m_df[m_df.change_type == 'Myeloid']
                g_df[(g_df.mouse_id == m) & (g_df.change_type == 'Myeloid')].percent_engraftment = myl.percent_engraftment + lymph.percent_engraftment

            plt.figure()
            sns.barplot(
                x='total',
                y='mouse_id',
                data=g_df,
                color=COLOR_PALETTES['change_status']['Unchanged'],
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Myeloid'],
                color=COLOR_PALETTES['change_type']['Myeloid'],
                label='Myeloid'
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Lymphoid'],
                color=COLOR_PALETTES['change_type']['Lymphoid'],
                label='Lymphoid'
            )
            plt.xlabel('Contribution of Changed Cells')
            plt.ylabel('')
            plt.suptitle(cell_type.title() + ' Changed vs Not-Changed Cumulative Abundance - First Timepoint')
            plt.title('Group: ' + group.replace('_', ' ').title())
            plt.gca().legend(title='Change Direction', loc='lower right')
            sns.despine(left=True, bottom=True)

            fname = save_path + os.sep + 'percent_contribution_changed_init' + cell_type + '_' + group + '.' + save_format
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
    """ Plot contributions of 'changed' cells, one line per group
    
    Arguments:
        changed_marked_df {pd.DataFrame} -- bias_change with clones marked as changed
        cell_type {str} -- cell type to analyze
    
    Keyword Arguments:
        percent_of_total {bool} -- look as percent of total (default: {False})
        line {bool} --  plot as line instead of bar (default: {False})
        save {bool} -- (default: {False})
        save_path {str} -- (default: {'./output'})
        save_format {str} -- (default: {'png'})

    Returns:
        None -- plt.show() to view plot
    """



    plt.figure()
    changed_sum_df = sum_abundance_by_change(changed_marked_df, percent_of_total=percent_of_total)
    changed_sum_df = changed_sum_df.loc[changed_sum_df.changed]
    changed_sum_cell_df = changed_sum_df.loc[changed_sum_df.cell_type == cell_type]

    print('Outlier Mice:')
    print(
        changed_sum_cell_df.loc[(changed_sum_cell_df.percent_engraftment > 60) & (changed_sum_cell_df.month == 14)].mouse_id
    )
    y_units = '(% WBC)'
    if percent_of_total:
        y_units = '(% of Tracked ' + cell_type.capitalize() +' cells)'

    group = 'both-groups'

    if line:
        sns.lineplot(
            x='month',
            y='percent_engraftment',
            hue='group',
            units='mouse_id',
            estimator=None,
            data=changed_sum_cell_df,
            palette=COLOR_PALETTES['group']
            )
    else:
        sns.barplot(
            x='month',
            y='percent_engraftment',
            hue='group',
            data=changed_sum_cell_df,
            palette=COLOR_PALETTES['group'],
            alpha=0.8
        )
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
    """ Histogram of lineage bias at time point, weighted by abundance
    
    Arguments:
        lineage_bias_df {pd.DataFrame} -- lineage bias data frame
        cell_type {str} -- cell type to weight abundance of
    
    Keyword Arguments:
        month {int} -- time point to analyze (default: {4})
        by_group {bool} -- wether to split by phenotype group (default: {True})
        bins {int} -- bins for histogram (default: {30})
        save {bool} --  (default: {False})
        save_path {str} --  (default: {'./output'})
        save_format {str} --  (default: {'png'})
    
    Returns:
        None -- plt.show() to view plot
    """



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
        timepoint_col: str,
        analyzed_cell_types: List[str] = ['gr', 'b'],
        group: str = 'all',
        line: bool = False,
        save: bool = False,
        save_path: str = 'output',
        save_format: str = 'png',
    ) -> None:
    """ Plot clone counts at cumulative abundance based thresholds
    
    Arguments:
        input_df {pd.DataFrame} -- abundance data frame
        abundance_cutoff {float} -- cumulative abundance of bottom percentil cutoff
    
    Keyword Arguments:
        analyzed_cell_types {List[str]} -- (default: {['gr', 'b']})
        group {str} -- 'no_change' or 'aging_phenotype (default: {'all'})
        line {bool} --  line or bar plot (default: {False})
        save {bool} --  (default: {False})
        save_path {str} --  (default: {'output'})
        save_format {str} --  (default: {'png'})
    
    Returns:
        None -- plt.show() to view plot
    """


    _, thresholds = calculate_thresholds_sum_abundance(
        input_df,
        timepoint_col=timepoint_col,
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
                palette=COLOR_PALETTES['mouse_id'],
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
    cell_df = input_df.loc[input_df.cell_type == cell_type]
    if by_group:
        cell_df = group_names_pretty(cell_df)
        sns.lineplot(
            x='month',
            y='percent_engraftment',
            hue='group',
            palette=COLOR_PALETTES['group'],
            data=cell_df
        )
    else:
        sns.lineplot(
            x='month',
            y='percent_engraftment',
            hue='mouse_id',
            palette=COLOR_PALETTES['mouse_id'],
            data=cell_df
        )

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
        thresholds: Dict[str, float],
        n_timepoints: int,
        by_day: bool = False,
        timepoint_col: str = None,
        color_col: str = 'mouse_id',
        group: str = 'all',
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png'
    ) -> None:
    """ Swarm plot of abundance of mice with all 4 month timepoints

    Arguments:
        input_df {pd.DataFrame} -- abundance dataframe
        cell_type {str} -- cell type to plot
        abundance_cutoff {float} -- cumulative abundance cutoff to calculate threshold at
        thresholds {Dict[str, float]} -- thresholds to filter input by
        n_timepoints {int} -- number of timepoints required for a clone to have

    Keyword Arguments:
        color_col {str} -- column to set hue to (default: {'mouse_id'})
        group {str} -- filter by group (default: {'all'})
        by_day {bool} -- use day as timepoint column
        save {bool} -- save plot or not (default: {False})
        save_path {str} -- path to save file (default: {''})
        save_format {str} -- plot save format (default: {'png'})

    Returns:
        None -- plt.show() to view plot
    """


    if group != 'all':
        input_df = input_df[input_df.group == group]
    
    time_col = 'month'
    if by_day:
        time_col = 'day'
    if timepoint_col is not None:
        time_col = timepoint_col

    print('Minimum number of timepoints: ' + str(n_timepoints))
    all_timepoint_df = filter_mice_with_n_timepoints(
        input_df,
        n_timepoints=n_timepoints,
        time_col=time_col,
    )
    filtered_df = filter_cell_type_threshold(
        all_timepoint_df,
        thresholds,
        analyzed_cell_types=[cell_type]
    )

    plt.figure()
    pal = sns.color_palette('hls', filtered_df.mouse_id.nunique())
    if color_col == 'mouse_id':
        pal = COLOR_PALETTES['mouse_id']
    sns.swarmplot(
        x=time_col,
        y='percent_engraftment',
        hue=color_col,
        data=filtered_df,
        palette=pal
        )

    title = 'Abundance of ' + cell_type.capitalize() \
          + ' with Abundance > ' \
          + str(round(thresholds[cell_type],2)) + '% WBC'
    plt.suptitle(title)
    plt.title('Group: ' + group.replace('_', ' ').title())
    plt.xlabel(time_col.title())
    plt.ylabel('Clone Abundance (% WBC)')
    fname = save_path + os.sep + 'swamplot_abundance' \
            + '_' + cell_type + '_a' \
            + str(round(abundance_cutoff, 2)).replace('.','-') \
            + '_' + group \
            + '.' + save_format
    save_plot(fname, save, save_format)


def plot_bias_change_between_gen(
        lineage_bias_df: pd.DataFrame,
        abundance_cutoff: float,
        thresholds: Dict[str, float],
        magnitude: bool = False,
        by_clone: bool = False,
        group: str = 'all',
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png',
        style: str = None,
        legend: Any = None,
    ) -> None:
    """ Plots lineage bias change between generations for serial transplant data

    Arguments:
        lineage_bias_df {pd.DataFrame} -- lineage bias dataframe
        abundance_cutoff {float} -- cumulative abundance cutoff to calculate threshold at
        thresholds {Dict[str, float]} -- thresholds to filter input by

    Keyword Arguments:
        group {str} -- filter by group (default: {'all'})
        magnitude {bool} -- use absolute value of change
        by_clone {bool} -- plot individual clones
        save {bool} -- save plot or not (default: {False})
        save_path {str} -- path to save file (default: {''})
        save_format {str} -- plot save format (default: {'png'})

    Returns:
        None -- plt.show() to view plot
    """

    if group != 'all':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

    
    plt.figure()
    filtered_df = filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds
        )
    between_gen_bias_change_df = between_gen_bias_change(
        filtered_df,
        absolute=magnitude
        )
    if style is None:
       style = 'group' 
    if by_clone:
        sns.lineplot(
            x='gen_change',
            y='bias_change',
            data=group_names_pretty(between_gen_bias_change_df),
            hue='mouse_id',
            estimator=None,
            style=style,
            units='code',
            palette=COLOR_PALETTES['mouse_id'],
            legend=legend
            )
    else:
        palette = COLOR_PALETTES['group']
        sns.lineplot(
            x='gen_change',
            y='bias_change',
            data=group_names_pretty(between_gen_bias_change_df),
            hue='group',
            palette=palette,
            )

    if magnitude:
        plt.ylabel('Magnitude Bias Change')
    else:
        plt.ylabel('Bias Change')

    fname_addon = ''
    if magnitude:
        fname_addon = '_magnitude'
    if by_clone:
        fname_addon = fname_addon + '_by-clone'

    title = 'Bias Change of clones ' \
          + ' with Abundance Gr > ' \
          + str(round(thresholds['gr'], 2)) + '% WBC' \
          + ' & B > ' \
          + str(round(thresholds['b'], 2)) + '% WBC' \
          + ' at Any Timepoint'
    plt.suptitle(title)
    if group != 'all':
        plt.title('Group: ' + group.replace('_', ' ').title())
    plt.xlabel('Generation Change')
    fname = save_path + os.sep + 'bias_change_generations' \
            + fname_addon + '_a' \
            + str(round(abundance_cutoff, 2)).replace('.', '-') \
            + '_' + group \
            + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_change_across_gens(
        lineage_bias_df: pd.DataFrame,
        abundance_cutoff: float,
        thresholds: Dict[str, float],
        magnitude: bool = False,
        by_clone: bool = False,
        group: str = 'all',
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png'
    ) -> None:
    """ Plots lineage bias change across generations for serial transplant data
        goes one to two, one to three, one to four, etc.
    Arguments:
        lineage_bias_df {pd.DataFrame} -- lineage bias dataframe
        abundance_cutoff {float} -- cumulative abundance cutoff to calculate threshold at
        thresholds {Dict[str, float]} -- thresholds to filter input by

    Keyword Arguments:
        group {str} -- filter by group (default: {'all'})
        magnitude {bool} -- use absolute value of change
        save {bool} -- save plot or not (default: {False})
        save_path {str} -- path to save file (default: {''})
        save_format {str} -- plot save format (default: {'png'})

    Returns:
        None -- plt.show() to view plot
    """

    if group != 'all':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

    
    plt.figure()

    filtered_df = filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds
        )
    across_gen_bias_change_df = across_gen_bias_change(
        filtered_df,
        absolute=magnitude
        )
    title = 'Bias Change of clones ' \
        + ' with Abundance Gr > ' \
        + str(round(thresholds['gr'], 2)) + '% WBC' \
        + ' & B > ' \
        + str(round(thresholds['b'], 2)) + '% WBC' \
        + ' at Any Timepoint'

    if by_clone:
        plt.subplot(2, 1, 1)
        ax = sns.swarmplot(
            x='gen_change',
            y='bias_change',
            data=group_names_pretty(across_gen_bias_change_df),
            hue='mouse_id',
            palette=COLOR_PALETTES['mouse_id']
            )
        ax.legend_.remove()

        if magnitude:
            plt.ylabel('Magnitude Bias Change')
        else:
            plt.ylabel('Bias Change')

        plt.xlabel('Generation Change')
        if group != 'all':
            plt.title('Group: ' + group.replace('_', ' ').title())

        plt.subplot(2, 1, 2)
        sns.lineplot(
            x='gen_change',
            y='bias_change',
            data=group_names_pretty(across_gen_bias_change_df),
            hue='mouse_id',
            palette=COLOR_PALETTES['mouse_id'],
            units='code',
            estimator=None,
            legend=None
            )
        if magnitude:
            plt.ylabel('Magnitude Bias Change')
        else:
            plt.ylabel('Bias Change')
        
        plt.xlabel('Generation Change')
    else:
        palette = COLOR_PALETTES['group']
        sns.lineplot(
            x='gen_change',
            y='bias_change',
            data=group_names_pretty(across_gen_bias_change_df),
            hue='group',
            palette=palette,
        )
        if magnitude:
            plt.ylabel('Magnitude Bias Change')
        else:
            plt.ylabel('Bias Change')
        
        plt.xlabel('Generation Change')

    plt.suptitle(title)

    fname_addon = ''
    if magnitude:
        fname_addon = '_magnitude'
    if by_clone:
        fname_addon = fname_addon + '_by-clone'

    fname = save_path + os.sep + 'bias_change_across_gens' \
            + fname_addon + '_a' \
            + str(round(abundance_cutoff, 2)).replace('.', '-') \
            + '_' + group \
            + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_change_time_kdes(
        lineage_bias_df: pd.DataFrame,
        first_timepoint: int,
        absolute_value: bool = True,
        group: str = 'all',
        cumulative: bool = True,
        timepoint_col: str = 'day',
        save: bool = False,
        save_path: str = 'output',
        save_format: str = 'png',
        cache_dir: str = '',
        cached_change: pd.DataFrame = None,
    ) -> None:

    time_change = 'between'
    if cumulative:
        time_change = 'across'

    if cached_change is not None:
        bias_change_df = cached_change
        if group != 'all':
            bias_change_df = bias_change_df[bias_change_df.group == group]
    else:
        # If not, calculate and save cached
        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df.assign(gen=lambda x: day_to_gen(x.day))
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen <= 8]

        if group != 'all':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

        bias_change_df = calculate_bias_change(
            lineage_bias_df,
            timepoint_col=timepoint_col,
            cumulative=cumulative,
            first_timepoint=first_timepoint
        )
        addon = time_change + '_' + group
        bias_change_df.to_csv(cache_dir + os.sep + addon + '_bias_change_df.csv', index=False)
    
    plt.figure()
    if group != 'all':
        bias_change_df = bias_change_df.loc[bias_change_df.group == group]

    x_label = 'Lineage Bias Change'
    magnitude = ''
    if absolute_value:
        bias_change_df.bias_change = bias_change_df.bias_change.abs()
        x_label = 'Magnitude of ' + x_label
        magnitude = '_magnitude'

    label_changes = bias_change_df[['label_change', 't2']].drop_duplicates().sort_values(by=['t2'])
    colors = sns.color_palette('coolwarm', n_colors=len(label_changes))
    for i in range(len(label_changes)):
        kde = sns.kdeplot(bias_change_df[bias_change_df.t2 == label_changes.iloc[i].t2].bias_change, label=label_changes.iloc[i].label_change, color=colors[i], linewidth=3)


    plt.title('Kernel Density Estimate of lineage bias change ' + time_change + ' ' + timepoint_col + 's')
    plt.suptitle(' Group: ' + group)
    plt.xlabel(x_label)
    plt.ylabel('Clone Density at Change')

    fname_addon = ''
    if magnitude:
        fname_addon = '_magnitude'
    if cumulative:
        fname_addon = fname_addon + '_across'
        title_add = ' across time points'
    else:
        fname_addon = fname_addon + '_between'
        title_add = ' between time points'

    fname = save_path + os.sep + 'bias_change_kde_' + group + fname_addon + '.' + save_format
    save_plot(fname, save, save_format)

def plot_abundance_change(
        abundance_df: pd.DataFrame,
        abundance_cutoff: float = 0,
        thresholds: Dict[str, float] = {'gr': 0, 'b': 0},
        cumulative: bool = False,
        magnitude: bool = False,
        filter_end_time: bool = False,
        by_clone: bool = False,
        by_day: bool = False,
        by_gen: bool = False,
        first_timepoint: int = 1,
        group: str = 'all',
        analyzed_cell_types: List[str] = ['gr', 'b'],
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png',
        cached_change: pd.DataFrame = None,
        cache_dir: str = ''
    ) -> None:


    if by_day:
        timepoint_col = 'day'
    elif by_gen:
        timepoint_col = 'gen'
        abundance_df = abundance_df.assign(gen=lambda x: day_to_gen(x.day))
    else:
        timepoint_col = 'month'

    if cached_change is not None:
        abundance_change_df = cached_change
        if group != 'all':
            abundance_df = abundance_df[abundance_df.group == group]
    else:
        # If not, calculate and save cached
        if group != 'all':
            abundance_df = abundance_df[abundance_df.group == group]
        abundance_change_df = calculate_abundance_change(
            abundance_df,
            timepoint_col=timepoint_col,
            cumulative=cumulative,
            first_timepoint=first_timepoint
        )
        addon = 'between'
        if cumulative:
            addon = 'across'
        abundance_change_df.to_csv(cache_dir + os.sep + addon + '_abundance_change_df.csv', index=False)

    if magnitude:
        abundance_change_df.abundance_change = abundance_change_df.abundance_change.abs()
    
    for cell_type in analyzed_cell_types:
        
        plt.figure()
        cell_change_df = abundance_change_df[abundance_change_df.cell_type == cell_type]
        if abundance_cutoff != 0:
            filt_abund = filter_cell_type_threshold(
                abundance_df,
                thresholds=thresholds,
                analyzed_cell_types=[cell_type],
                )
            if not filt_abund[filt_abund.cell_type != cell_type].empty:
                ValueError('More cell types than expected')
            if filter_end_time:
                cell_change_df[timepoint_col] = cell_change_df['t2']
            else:
                cell_change_df[timepoint_col] = cell_change_df['t1']
            
            cell_change_df = cell_change_df.merge(
                filt_abund[['code', 'mouse_id', 'cell_type', timepoint_col]],
                on=['code', 'mouse_id', 'cell_type', timepoint_col],
                how='inner',
                validate='m:m'
            )


        ordered_labels = cell_change_df[['label_change', 't2']].sort_values(['t2'])['label_change'].unique()
        plt.subplot(2, 1, 1)
        palette = COLOR_PALETTES['group']
        sns.pointplot(
            x='label_change',
            y='abundance_change',
            data=group_names_pretty(cell_change_df),
            order=ordered_labels,
            hue='group',
            palette=palette,
            )
        plt.xlabel('')
        if magnitude:
            plt.ylabel('Magnitude Abundance Change')
        else:
            plt.ylabel('Abundance Change')
        plt.subplot(2, 1, 2)
        sns.lineplot(
            x='t2',
            y='abundance_change',
            data=cell_change_df,
            hue='mouse_id',
            estimator=None,
            markers=True,
            style='group',
            units='code',
            palette=COLOR_PALETTES['mouse_id'],
            legend=None
            )

        if magnitude:
            plt.ylabel('Magnitude Abundance Change')
        else:
            plt.ylabel('Abundance Change')

        fname_addon = ''
        if magnitude:
            fname_addon = '_magnitude'
        if by_clone:
            fname_addon = fname_addon + '_by-clone'
        if cumulative:
            fname_addon = fname_addon + '_across'
            title_add = ' across time points'
        else:
            fname_addon = fname_addon + '_between'
            title_add = ' between time points'

        if filter_end_time:
            filter_time = 'after'
        else:
            filter_time = 'before'
        title = 'Abundance Change in ' + cell_type.title()  \
                + ' with Abundance > ' + str(round(thresholds[cell_type],2)) \
                + '% (WBC) ' + filter_time.title() + ' Change'
        plt.suptitle(title)
        if group != 'all':
            plt.title('Group: ' + group.replace('_', ' ').title() + title_add)
        plt.xlabel('Time (' + timepoint_col.title() + 's)' + title_add)
        fname = save_path + os.sep + 'abundance_change' \
                + fname_addon + '_a' \
                + str(round(abundance_cutoff, 2)).replace('.', '-') \
                + '_' + filter_time \
                + '_' + group \
                + '_' + cell_type \
                + '.' + save_format
        save_plot(fname, save, save_format)

def plot_bias_change_cutoff(
        lineage_bias_df: pd.DataFrame,
        thresholds: Dict[str, float],
        abundance_cutoff: float = 0.0,
        absolute_value: bool = False,
        group: str = 'all',
        min_time_difference: int = 0,
        save: bool = False,
        save_path: str = 'output',
        save_format: str = 'png',
        cached_change: pd.DataFrame = None,
        cache_dir: str = ''
    ) -> None:
    """ Plots KDE of bias change annotated with line to cut "change" vs "non change" clones

    Arguments:
        lineage_bias_df {pd.DataFrame} -- dataframe of lineage bias information
        thresholds {Dict[str,float]} -- thresholds used to filter
        abundance_cutoff {float} -- abundance cutoff used to generate thresholds

    Keyword Arguments:
        absolute_value {bool} -- Whether plot is done on magnitude, or including direction (default: {False})
        group {str} --  Group filtered for (default: {'all'})
        min_time_difference {int} -- Minimum days of seperation for bias change (280ish for 10 months)
        save {bool} --  Wether to save plot (default: {False})
        save_path {str} -- Where to save plot (default: {'output'})
        save_format {str} --  What file format to save plot (default: {'png'})
    """


    if cached_change is not None:
        bias_change_df = cached_change
    else:
        filt_lineage_bias_df = filter_lineage_bias_anytime(
            lineage_bias_df,
            thresholds=thresholds
        )
        bias_change_df = get_bias_change(filt_lineage_bias_df)
        bias_change_df.to_csv(cache_dir + os.sep + 'bias_change_df_a'+str(round(abundance_cutoff, 2)) + '.csv', index=False)
        

    plt.figure()
    bias_change_df = bias_change_df[bias_change_df.time_change >= min_time_difference]
    if group != 'all':
        bias_change_df = bias_change_df.loc[bias_change_df.group == group]

    if absolute_value:
        kde = sns.kdeplot(
            bias_change_df.bias_change.abs(),
            kernel='gau',
            shade=True,
            color='silver',
            alpha=.3
        )
    else:
        kde = sns.kdeplot(
            bias_change_df.bias_change,
            kernel='gau',
            shade=True,
            color='silver',
            alpha=.3
        )

    # C0 KDE of all clones
    x, y = kde.get_lines()[0].get_data()
    y_peak = y.argmax() + 1

    # C1 KDE of "unchanged" clones
    y1 = np.zeros(x.shape)
    y_vals = np.append(y[:y_peak], y[:y_peak][::-1])
    y1[:y_vals.shape[0]] = y_vals

    # C2 KDE of "changed" clones
    y2 = y - y1

    x_c, y_c = intersection(x, y1, x, y2)
    plt.plot(x, y1, c='deepskyblue')
    plt.plot(x , y2, c='firebrick')
    plt.scatter(x_c, y_c, c='k')
    plt.vlines(x_c[0], 0, max(y))
    kde.text(x_c[0] + .1, max(y)/1.1, 'Change at: ' + str(round(x_c[0], 3)))

    plt.title('Kernel Density Estimate of lineage bias change')
    plt.suptitle('Abundance Cutoff: ' + str(round(abundance_cutoff, 2)) + ' Group: ' + group)
    plt.xlabel('Magnitude of Lineage Bias Change')
    plt.ylabel('Clone Density at Change')
    kde.legend_.remove()

    fname = save_path + os.sep + 'bias_change_cutoff_a' + str(abundance_cutoff).replace('.', '-') + '_' + group + '_mtd' + str(min_time_difference) + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_change_rest(
        lineage_bias_df: pd.DataFrame,
        cumulative: bool = False,
        by_clone: bool = False,
        by_day: bool = False,
        by_gen: bool = False,
        first_timepoint: int = 1,
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png',
        cached_change: pd.DataFrame = None,
        cache_dir: str = ''
    ) -> None:


    if by_day:
        timepoint_col = 'day'
    elif by_gen:
        timepoint_col = 'gen'
        lineage_bias_df = lineage_bias_df.assign(gen=lambda x: day_to_gen(x.day))
    else:
        timepoint_col = 'month'

    if cached_change is not None:
        bias_change_df = cached_change

    else:
        # If not, calculate and save cached
        bias_change_df = calculate_bias_change(
            lineage_bias_df,
            timepoint_col=timepoint_col,
            cumulative=cumulative,
            first_timepoint=first_timepoint
        )
        addon = 'between'
        if cumulative:
            addon = 'across'
        bias_change_df.to_csv(cache_dir + os.sep + addon + '_bias_change_df.csv', index=False)

    
    bias_change_df = bias_change_df.sort_values(by=['t2'])
    for group, group_df in bias_change_df.groupby('group'):
        plt.figure()

        ordered_labels = bias_change_df[['label_change', 't2']].sort_values(['t2'])['label_change'].unique()
        group_names = ['aging_phenotype', 'no_change']
        plt.subplot(2, 1, 1)
        sns.lineplot(
            x='label_change',
            y='bias_change',
            data=group_df,
            hue='mouse_id',
            style='code',
            palette=COLOR_PALETTES['mouse_id']
            )
        plt.ylabel('Bias Change')

        plt.subplot(2, 1, 2)
        group_df.bias_change = group_df.bias_change.abs()
        sns.lineplot(
            x='label_change',
            y='bias_change',
            data=group_df,
            hue='mouse_id',
            style='code',
            legend=False,
            palette=COLOR_PALETTES['mouse_id']
            )
        plt.ylabel('Magnitude Bias Change')

        fname_addon = ''
        if cumulative:
            fname_addon = fname_addon + '_across'
            title_add = ' across time points'
        else:
            fname_addon = fname_addon + '_between'
            title_add = ' between time points'

        title = 'Bias Change in rest of clones ' + group.replace('_'," ").title()
        plt.suptitle(title)
        plt.xlabel('Time (' + timepoint_col.title() + 's)' + title_add)
        fname = save_path + os.sep + 'rest_bias_change' \
                + fname_addon \
                + '_' + group \
                + '.' + save_format
        save_plot(fname, save, save_format)

def plot_rest_vs_tracked(
        lineage_bias_df: pd.DataFrame,
        rest_of_clones_bias_df: pd.DataFrame,
        cell_count_df: pd.DataFrame,
        y_col: str,
        abundance_cutoff: float,
        thresholds: Dict[str, float],
        timepoint_col = 'day',
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png',
        cached_change: pd.DataFrame = None,
        cache_dir: str = ''
    ) -> None:
    
    lineage_bias_df = filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds
    )
    lineage_bias_df['code'] = 'Tracked'
    lineage_bias_df = lineage_bias_df.append(rest_of_clones_bias_df, ignore_index=True)

    #total_df = pd.DataFrame(columns=lineage_bias_df.columns)
    #for mt, group in cell_count_df.groupby(['mouse_id', timepoint_col]):
        #mt_row = pd.DataFrame(columns=lineage_bias_df.columns)
        #mouse_id = mt[0]
        #time = mt[1]
        #gr = group[group.cell_type == 'gr'].cell_count
        #b = group[group.cell_type == 'b'].cell_count
        #wbc = group[group.cell_type == 'wbc'].cell_count
        #bias = calc_bias(gr.values[0], b.values[0])
        #mt_row.gr_percent_engraftment = [gr.values[0]/wbc.values[0]]
        #mt_row.b_percent_engraftment = [b.values[0]/wbc.values[0]]
        #mt_row.lineage_bias = [bias]
        #mt_row.mouse_id = [mouse_id]
        #mt_row.code = ['Total']
        #mt_row[timepoint_col] = [time]
        #total_df = total_df.append(mt_row)

    #lineage_bias_df = lineage_bias_df.append(total_df, ignore_index=True)

    for mouse_id, mouse_df in lineage_bias_df.groupby('mouse_id'):
        plt.figure()
        sns.lineplot(
            x=timepoint_col,
            y=y_col,
            hue='code',
            data=mouse_df[mouse_df.code == 'Tracked'],
            markers=True,
            palette=COLOR_PALETTES['code']
        )
        y_title = y_col_to_title(y_col)
        plt.ylabel('Tracked ' + y_title)
        ax2 = plt.twinx()
        sns.lineplot(
            x=timepoint_col,
            y=y_col,
            hue='code',
            data=mouse_df[mouse_df.code != 'Tracked'],
            markers=True,
            palette=COLOR_PALETTES['code']
        )
        group = str(mouse_df.iloc[0].group)
        plt.title('Group: ' + group.replace('_', ' ').title())
        plt.ylabel(y_title)
        plt.suptitle(mouse_id  \
            + ' ' + y_title \
            + ' Tracked Clone Abundance Cutoff (anytime): ' \
            + str(round(abundance_cutoff, 2))
        )
        plt.xlabel(timepoint_col.title())

        fname = save_path + os.sep  \
                + group + os.sep \
                + 'rest_vs_tracked_' + mouse_id \
                + '_' + y_col \
                + '_a' \
                + str(round(abundance_cutoff, 2)) \
                + '.' + save_format
        save_plot(fname, save, save_format)

def plot_extreme_bias_abundance(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png',
    ) -> None:

    for cell_type in ['gr', 'b']:
        fname_suffix = '_' + cell_type \
            + '_extreme_bias_abundance.' \
            +  save_format

        extreme_bin = .2500
        extreme_myeloid_index = (1-lineage_bias_df.lineage_bias) <= extreme_bin
        plt.figure()
        extreme_myeloid_df = lineage_bias_df[extreme_myeloid_index]
        myeloid_df = pd.DataFrame(extreme_myeloid_df.groupby([timepoint_col, 'group', 'mouse_id'])['gr_percent_engraftment','b_percent_engraftment'].sum()).reset_index()

        # Myeloid Biased Abundance
        sns.lineplot(
            x=timepoint_col,
            y=cell_type+'_percent_engraftment',
            hue='group',
            style='group',
            data=myeloid_df,
            palette=COLOR_PALETTES['group'],
            markers=True,
        )
        plt.ylabel('Sum ' + cell_type.title() + ' Abundance (%WBC)')
        plt.title(cell_type.title() + ' Abundance of Extremely Myeloid Biased Clones')

        bias_type = 'myeloid'
        fname = save_path + os.sep + bias_type + fname_suffix
        save_plot(fname, save, save_format)

        # Lymhoid Biased Abundance
        plt.figure()
        extreme_lymphoid_index = (lineage_bias_df.lineage_bias + 1) <= extreme_bin
        extreme_lymphoid_df = lineage_bias_df[extreme_lymphoid_index]
        lymphoid_df = pd.DataFrame(extreme_lymphoid_df.groupby([timepoint_col, 'group', 'mouse_id'])['gr_percent_engraftment','b_percent_engraftment'].sum()).reset_index()
        sns.lineplot(
            x=timepoint_col,
            y=cell_type+'_percent_engraftment',
            hue='group',
            style='group',
            data=lymphoid_df,
            palette=COLOR_PALETTES['group'],
            markers=True,
        )
        plt.ylabel('Sum ' + cell_type.title() + ' Abundance (%WBC)')
        plt.title(cell_type.title() + ' Abundance of Extremely Lymphoid Biased Clones')

        bias_type = 'lymphoid'
        fname = save_path + os.sep + bias_type + fname_suffix
        save_plot(fname, save, save_format)

def plot_extreme_bias_time(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: int,
        y_col: str,
        bias_cutoff: float,
        invert_selection: bool = False,
        by_clone: bool = False,
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png',
    ) -> None:
    fname_addon = ''
    biased_at_time_df = filter_biased_clones_at_timepoint(
        lineage_bias_df,
        bias_cutoff,
        timepoint,
        timepoint_col,
        within_cutoff=invert_selection
    )
    filtered_clones = biased_at_time_df.code.unique()
    
    # Find abundance data if not looking at lineage bias
    if y_col != 'lineage_bias':
        biased_at_time_df = bias_clones_to_abundance(
            biased_at_time_df,
            clonal_abundance_df,
            y_col
        )
        

    y_title = y_col_to_title(y_col)
    if invert_selection:
        plot_title = y_title + ' of Clones less biased than +/- ' \
            + str(round(bias_cutoff, 2)) + ' at ' \
            + timepoint_col.title() + ' ' + str(timepoint)
    else:
        plot_title = y_title + ' of Clones more biased than ' \
            + str(round(bias_cutoff, 2)) + ' at ' \
            + timepoint_col.title() + ' ' + str(timepoint)
    if by_clone:
        for group, group_df in biased_at_time_df.groupby('group'):
            plt.figure()
            fname_addon += '_by-clone'
            sns.lineplot(
                x=timepoint_col,
                y=y_col,
                data=group_df,
                hue='mouse_id',
                units='code',
                estimator=None,
                legend=None,
                palette=COLOR_PALETTES['mouse_id']
            )
            plt.ylabel(y_title)
            plt.xlabel(timepoint_col.title())
            plt.suptitle(plot_title)
            plt.title('Group: ' + group.replace('_', ' ').title())

            if invert_selection:
                fname = save_path + os.sep + 'within_bias' \
                    + str(round(bias_cutoff, 2)).replace('.','-') \
                    + '_' + timepoint_col[0] \
                    + str(timepoint) \
                    + '_' + group \
                    + fname_addon \
                    + '_' + y_col + '.' + save_format
            else:
                fname = save_path + os.sep + 'extreme_bias_' \
                    + str(round(bias_cutoff, 2)).replace('.','-') \
                    + '_' + timepoint_col[0] \
                    + str(timepoint) \
                    + '_' + group \
                    + fname_addon \
                    + '_' + y_col + '.' + save_format
            save_plot(fname, save, save_format)
    else:
        plt.figure()
        sns.lineplot(
            x=timepoint_col,
            y=y_col,
            data=biased_at_time_df,
            hue='group',
            palette=COLOR_PALETTES['group'],
        )
        plt.ylabel(y_title)
        plt.xlabel(timepoint_col.title())
        plt.suptitle(plot_title)

        if invert_selection:
            fname = save_path + os.sep + 'within_bias' \
                + str(round(bias_cutoff, 2)).replace('.','-') \
                + '_' + timepoint_col[0] \
                + str(timepoint) \
                + fname_addon \
                + '_' + y_col + '.' + save_format
        else:
            fname = save_path + os.sep + 'extreme_bias_' \
                + str(round(bias_cutoff, 2)).replace('.','-') \
                + '_' + timepoint_col[0] \
                + str(timepoint) \
                + fname_addon \
                + '_' + y_col + '.' + save_format
        save_plot(fname, save, save_format)
    
def plot_bias_dist_at_time(
        lineage_bias_df: pd.DataFrame,
        abundance_thresholds: List[float],
        timepoint_col: str,
        group: str = 'all',
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png',
    ) -> None:

    for t, tgroup in lineage_bias_df.groupby([timepoint_col]):
        pal = sns.color_palette('coolwarm', len(abundance_thresholds))
        i=0
        plt.figure()
        plt.title(
            ' Lineage Bias Distribution'\
            + ' at ' + str(t) + ' ' + timepoint_col.title()
        )
        for a in abundance_thresholds:
            index = (
                (tgroup['gr' + '_percent_engraftment'] >= a) \
                | (tgroup['b' + '_percent_engraftment'] >= a)
            )
            data = tgroup[index].lineage_bias
            sns.kdeplot(data, color=pal[i], label=a)
            i+=1
        
        fname = save_path + os.sep \
            + 'Lineage_Bias_Dist_at_Time_Thresholds_' \
            + timepoint_col[0] + str(t) \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_stable_clones(
        lineage_bias_df: pd.DataFrame,
        bias_change_cutoff: float,
        t1: int,
        timepoint_col: str,
        clonal_abundance_df: pd.DataFrame = None,
        thresholds: Dict[str,float] = {'gr': 0, 'b': 0},
        y_col: str = 'lineage_bias',
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    stable_clone_df = filter_stable_initially(
        lineage_bias_df,
        t1=t1,
        timepoint_col=timepoint_col,
        bias_change_cutoff=bias_change_cutoff,
    )
    fname_prefix = save_path + os.sep + 'stable_clones'
    y_title = y_col_to_title(y_col)
    x_title = timepoint_col.title()

    if y_col != 'lineage_bias':
        stable_clone_df = bias_clones_to_abundance(
            stable_clone_df,
            clonal_abundance_df,
            y_col
        )

    if thresholds != {'gr': 0, 'b': 0}:
        print('\n !! WARNING: Threshold filtering not implemented !! \n')
    
    plt.figure()
    sns.lineplot(
        x=timepoint_col,
        y=y_col,
        data=group_names_pretty(stable_clone_df),
        hue='group',
        palette=COLOR_PALETTES['group']
    )
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    title = 'Bias Change < ' +str(bias_change_cutoff) \
        + ' between ' + timepoint_col.title() + ' ' + str(t1) \
        + ' And Next Time Point per clone'
    plt.title(title)
    

    fname = fname_prefix + '_' + y_col + '_' \
        + timepoint_col[0] + str(t1) \
        + '_average.' + save_format
    save_plot(fname, save, save_format)
    for gname, group_df in stable_clone_df.groupby('group'):
        plt.figure()
        sns.lineplot(
            x=timepoint_col,
            y=y_col,
            data=group_df,
            hue='mouse_id',
            palette=COLOR_PALETTES['mouse_id'],
            units='code',
            estimator=None,
            legend=False,
        )
        plt.suptitle('Group: ' + gname.replace('_', ' ').title())
        plt.title(title)
        plt.ylabel(y_title)
        plt.xlabel(x_title)
        fname = fname_prefix + '_' + y_col + '_' \
            + timepoint_col[0] + str(t1) \
            + '_' + gname + '_' + 'by-clone_' \
            + '.' + save_format
        save_plot(fname, save, save_format)


def plot_bias_dist_mean_abund(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        cutoffs: List[float],
        y_col: str = 'sum_abundance',
        by_group: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    df_cols = ['mouse_id', 'code', 'group', 'average_'+y_col, 'bias_change']
    bias_dist_df = pd.DataFrame(columns=df_cols)
    for name, group in lineage_bias_df.groupby(['code', 'mouse_id', 'group']):
        if len(group) < 2:
            continue
        bias_change_row = pd.DataFrame(columns=df_cols)
        sorted_group = group.sort_values(by=timepoint_col)
        t1 = sorted_group.iloc[0]
        t2 = sorted_group.iloc[-1]
        if y_col == 'sum_abundance':
            avg_val = (sorted_group['gr_percent_engraftment'] + sorted_group['b_percent_engraftment']).mean()
        else:
            avg_val = sorted_group[y_col].mean()
        bias_change = t2.lineage_bias - t1.lineage_bias
        bias_change_row['code'] = [name[0]]
        bias_change_row['mouse_id'] = [name[1]]
        bias_change_row['group'] = [name[2]]
        bias_change_row['average_'+y_col] = [avg_val]
        bias_change_row['bias_change'] = [bias_change]
        bias_dist_df = bias_dist_df.append(bias_change_row, ignore_index=True)
    
    if by_group:
        for gname, g_df in bias_dist_df.groupby('group'):
            plt.figure()
            pal = sns.color_palette('coolwarm', len(cutoffs))
            i=0
            for cutoff in cutoffs:
                c = pal[i]
                i += 1
                filt_df = g_df[g_df['average_'+y_col] >= cutoff]
                sns.distplot(
                    filt_df.bias_change,
                    rug=True,
                    hist=False,
                    color=c,
                    label=str(cutoff)
                )

            plt.suptitle(y_col_to_title('average_'+y_col))
            plt.title(gname.replace('_',' ').title())
            plt.xlabel('Overall Change in Bias Per Clone')
            fname = save_path + os.sep +'bias_change_dist_' \
                + y_col \
                + '_' + gname \
                + '.' + save_format
            save_plot(fname, save, save_format)
    else:
        plt.figure()
        pal = sns.color_palette('coolwarm', len(cutoffs))
        i=0
        for cutoff in cutoffs:
            c = pal[i]
            i += 1
            filt_df = bias_dist_df[bias_dist_df['average_'+y_col] >= cutoff]
            sns.distplot(
                filt_df.bias_change,
                rug=True,
                hist=False,
                color=c,
                label=str(cutoff)
            )

        plt.title(y_col_to_title('average_'+y_col))
        plt.xlabel('Overall Change in Bias Per Clone')
        fname = save_path + os.sep +'bias_change_dist_' \
            + y_col + '.' + save_format
        save_plot(fname, save, save_format)

def plot_abund_swarm_box(
        clonal_abundance_df: pd.DataFrame,
        thresholds: Dict[str, float],
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    clonal_abundance_df = group_names_pretty(clonal_abundance_df)
    for cell_type, cell_df in clonal_abundance_df.groupby('cell_type'):
        plt.figure()
        str_th = str(round(thresholds[cell_type], 2))
        plt.title(
            cell_type.title() \
                + ' Clones With Abundance > ' \
                + str_th
        )
        ax = sns.boxplot(
            x='group',
            y='percent_engraftment',
            color='white',
            data=cell_df,
            whis=np.inf,
            dodge=False,
        )
        sns.swarmplot(
            x='group',
            hue='mouse_id',
            palette=COLOR_PALETTES['mouse_id'],
            y='percent_engraftment',
            data=cell_df,
            color=".2",
            ax=ax
        )
        plt.ylabel('Abundance (% WBC)')
        plt.xlabel('Group')
        ax.legend().remove()
        fname = save_path + os.sep + 'abund_first_timepoint_ ' \
            + cell_type + '_a' + str_th.replace('.', '-') \
            + '.' + save_format
        save_plot(fname, save, save_format)