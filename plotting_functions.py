
""" Functions used to help plot data in plot_data.py

"""

from typing import List, Tuple, Dict, Any
from collections import OrderedDict
from itertools import combinations
import os
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from colour import Color, hex2rgb
from colorama import init, Fore, Back, Style
from pyvenn import venn
from aggregate_functions import filter_threshold, count_clones, \
    combine_enriched_clones_at_time, count_clones_at_percentile, \
    filter_mice_with_n_timepoints, filter_cell_type_threshold, \
    find_top_percentile_threshold, get_data_from_mice_missing_at_time, \
    get_max_by_mouse_timepoint, sum_abundance_by_change, find_intersect, \
    calculate_thresholds_sum_abundance, filter_lineage_bias_anytime, \
    across_gen_bias_change, between_gen_bias_change, calculate_abundance_change, \
    day_to_gen, calculate_bias_change, filter_biased_clones_at_timepoint, \
    bias_clones_to_abundance, filter_stable_initially, \
    calculate_first_last_bias_change_with_avg_data, add_first_last_to_lineage_bias, \
    add_average_abundance_to_lineage_bias, abundant_clone_survival,\
    not_survived_bias_by_time_change, mark_changed, filter_bias_change_timepoint, \
    find_enriched_clones_at_time, create_lineage_bias_survival_df, calculate_bias_change_cutoff, \
    find_last_clones, add_bias_category, find_last_clones_in_mouse, \
    filter_lineage_bias_thresholds, filter_lymphoid_exhausted_at_time, \
    calculate_survival_perc, filter_first_last_by_mouse, MAP_LINEAGE_BIAS_CATEGORY_SHORT, \
    MAP_LINEAGE_BIAS_CATEGORY, get_clones_at_timepoint, abundance_to_long_by_cell_type, \
    add_exhaustion_labels, mark_outliers
from data_types import timepoint_type, y_col_type
from lineage_bias import get_bias_change, calc_bias
from intersection.intersection import intersection

init(autoreset=True)
COLOR_PALETTES = json.load(open('color_palettes.json', 'r'))
LINE_STYLES = json.load(open('line_styles.json', 'r'))
NEW_GROUP_NAME_MAP = {
    'aging_phenotype': 'E-MOLD',
    'no_change': 'D-MOLD'
}
def y_col_to_title(y_col: str) -> str:
    y_title = y_col.replace('_', ' ').title().replace(
        'Percent Engraftment',
        'Abundance (%WBC)'
    )
    return y_title

def save_plot(file_name: str, save: bool, save_format: str) -> None:
    if save:
        if os.path.isdir(os.path.dirname(file_name)) or os.path.dirname(file_name) == '':
            print(Fore.GREEN + 'Saving to: ' + file_name)
            plt.savefig(
                file_name,
                format=save_format,
                bbox_inches='tight',
            )
        else:
            print('Directory does not exist for: ' + file_name)
            create_dir = input("Create Directory? (y/n) \n")
            if create_dir.lower() == 'y':
                os.makedirs(os.path.dirname(file_name))
                plt.savefig(
                    file_name,
                    format=save_format,
                    bbox_inches='tight',
                )

def print_p_value(context: str, p_value: float, show_ns: bool = False):
    """ Print P-Value, styke by significance
    
    Arguments:
        context {str} -- What to print just before P-Value
        p_value {float}
    """
    if p_value < 0.001:
        print(
            Fore.WHITE + Back.CYAN + Style.BRIGHT
            + context
            + ' P-Value: ' + str(p_value)
        )
    elif p_value < 0.01:
        print(
            Fore.CYAN + Style.BRIGHT
            + context
            + ' P-Value: ' + str(p_value)
        )
    elif p_value < 0.05:
        print(
            Fore.CYAN 
            + context
            + ' P-Value: ' + str(p_value)
        )
    elif show_ns:
        print(
            Fore.WHITE
            + context
            + ' P-Value: ' + str(p_value)
        )

def get_myeloid_to_lymphoid_colors(cats: List[str]) -> List[str]:
    myeloid_color = Color(COLOR_PALETTES['change_type']['Myeloid'])
    myeloid_colors = list(Color('white').range_to(
        myeloid_color,
        int(round(len(cats)/2)) + 1
    ))
    lymphoid_color = Color(COLOR_PALETTES['change_type']['Lymphoid'])
    lymphoid_colors = list(lymphoid_color.range_to(
        Color('white'),
        int(round(len(cats)/2)) + 1
    ))
    colors = lymphoid_colors[:-1] \
        + [Color(COLOR_PALETTES['change_type']['Unchanged'])] \
        + myeloid_colors[1:]
    colors = [x.hex_l for x in colors]
    return colors

def plot_clone_count(clone_counts: pd.DataFrame,
                     thresholds: Dict[str, float],
                     analyzed_cell_types: List[str],
                     abundance_cutoff: float,
                     timepoint_col: str,
                     group: str = 'all',
                     line: bool = False,
                     save: bool = False,
                     save_path: str = './output',
                     save_format: str = 'png',
                    ) -> None:
    """ Plots clone counts, based on stats from count_clones function

    Arguments:
        clone_counts {pd.DataFrame} -- dictionary of statistics from data, mean and sem required
        time_points {List[int]} -- list of time points to plot
        threshold {float} -- threshold value, used in title of plot
    plt.show()
        timepoint_col {str} -- column to look for time values in
        analysed_cell_types {List[str]} -- list of cell types analyzed

    """

    x_var = timepoint_col
    n_timepoints = clone_counts[timepoint_col].nunique()

    if line:
        for cell_type, c_df in clone_counts.groupby('cell_type'):
            if cell_type == 'Total':
                thresh = abundance_cutoff
            else:
                thresh = round(thresholds[cell_type], 2)
            plt.figure()
            sns.lineplot(x=x_var,
                         y='code',
                         hue='mouse_id',
                         palette=COLOR_PALETTES['mouse_id'],
                         data=c_df,
                         legend=False
                        )
            plt.suptitle('Clone Counts in '+ cell_type +' Cells with Abundance > ' + str(thresh) + ' % WBC')
            label = 'Group: ' + group
            plt.title(label)
            plt.xlabel(x_var.title())
            plt.ylabel('Number of Clones')
            if save:
                fname = save_path + os.sep + 'clone_count_t' + str(thresh).replace('.', '-') + '_' + cell_type + '_' + group + '.' + save_format
                print('Saving to: ' + fname)
                plt.savefig(fname, format=save_format)
    else:
        plt.figure(figsize=(n_timepoints*1.5, 5))
        sns.barplot(x=x_var,
                    y='code',
                    hue='cell_type',
                    hue_order=analyzed_cell_types + ['Total'],
                    data=clone_counts,
                    capsize=.08,
                    errwidth=0.5,
                    palette=COLOR_PALETTES['cell_type']
                   )
        plt.suptitle('Clone Counts By Cell Type with Abundance Cutoff ' + str(100 - abundance_cutoff))
        label = 'Group: ' + group
        plt.title(label)
        plt.xlabel(x_var.title())
        plt.ylabel('Number of Clones')
        fname = save_path + os.sep \
            + 'clone_count_a' \
            + str(abundance_cutoff).replace('.', '-') + '_' + group + '.' + save_format
        save_plot(fname, save, save_format)



def plot_clone_count_by_thresholds(input_df: pd.DataFrame,
                                   thresholds: Dict[str, float],
                                   analysed_cell_types: List[str],
                                   timepoint_col: str,
                                   abundance_cutoff: float = 0,
                                   by_day: bool = False,
                                   group: str = 'all',
                                   line: bool = False,
                                   save: bool = False,
                                   save_path: str = './output/'
                                  ) -> None:
    """Wrapper of plot_clone_counts to plot multiple for desired threshold values

    Arguments:
        input_df {pd.DataFrame} -- long formatted data from step7 output
        thresholds {Dict[str, float]} -- list of thresholds to plot
        analysed_cell_types {List[str]} -- cell types to consider in analysis
        timepoint_col {str} -- column to look for time values in

    Returns:
        None -- plots created, run plt.show() to observe
    """

    # Filter by group if specified
    if group != 'all':
        input_df = input_df.loc[input_df.group == group]

    # Plot at thresholds
    threshold_df = filter_cell_type_threshold(
        input_df,
        thresholds, 
        analysed_cell_types)
    clone_counts = count_clones(threshold_df, timepoint_col)

    plot_clone_count(clone_counts,
                        thresholds,
                        analysed_cell_types,
                        abundance_cutoff=abundance_cutoff,
                        group=group,
                        save=save,
                        line=line,
                        timepoint_col=timepoint_col,
                        save_path=save_path)

def plot_clone_enriched_at_time(filtered_df: pd.DataFrame,
                                enrichement_months: List[Any],
                                enrichment_thresholds: Dict[str, float],
                                timepoint_col: str,
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
            timepoint_col=timepoint_col,
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
                    x=timepoint_col,
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
                    x=timepoint_col,
                    y='percent_engraftment',
                    hue='group',
                    data=group_names_pretty(cell_df),
                    legend=None,
                    sort=True,
                    palette=COLOR_PALETTES['group']
                )
            plt.suptitle(cell_type.title() + ' Clones with Abundance > '
                        + str(round(enrichment_thresholds[cell_type], 2))
                        + ' % WBC At ' + timepoint_col.title() 
                        + ': ' + str(month))
            plt.xlabel('')
            plt.ylabel('Abundance (%WBC)')
            plt.subplot(2, 1, 2)
            ax = sns.swarmplot(
                x=timepoint_col,
                y='percent_engraftment',
                hue='group',
                data=group_names_pretty(cell_df),
                dodge=True,
                palette=COLOR_PALETTES['group']
            )
            plt.ylabel('Abundance (%WBC)')
            plt.xlabel(timepoint_col.title())
            fname = save_path \
                    + os.sep \
                    + 'dominant_clones_' + cell_type + '_' + title_addon \
                    + str(round(enrichment_thresholds[cell_type], 2)).replace('.', '-') \
                    + '_' + timepoint_col[0] + str(month) + '.' + save_format
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
                         clonal_abundance_df: pd.DataFrame,
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

    log_scale = False
    if y_col != 'lineage_bias':
        lineage_bias_df = bias_clones_to_abundance(
            lineage_bias_df,
            clonal_abundance_df,
            y_col
        )
        log_scale = True

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 2,
            'axes.linewidth': 3,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )

    if by_clone:
        fname_prefix += '_by-clone'
        phenotype = 'all'
        ymax = lineage_bias_df[y_col].max()
        ymin = lineage_bias_df[y_col].min()
        plt.figure(figsize=(6, 4))

        ax = sns.lineplot(
            x=timepoint_col,
            y=y_col,
            data=group_names_pretty(lineage_bias_df),
            hue='mouse_id',
            legend=False,
            palette=COLOR_PALETTES['mouse_id'],
            units='code',
            estimator=None
        )
        if log_scale:
            ax.set(yscale='log')

        plt.xlabel(timepoint_col.title())
        plt.xticks(ticks=lineage_bias_df[timepoint_col].unique())
        y_title = y_col.replace('percent_engraftment', 'Abundance (% WBC)').replace('_', ' ').title()
        plt.ylabel(y_title)
        plt.title(title_addon)
        fname = fname_prefix \
            + '_' + phenotype \
            + '.' + save_format
        save_plot(fname, save, save_format)

        for phenotype, group in lineage_bias_df.groupby('group'):
            plt.figure(figsize=(6, 4))
            ax = sns.lineplot(
                x=timepoint_col,
                y=y_col,
                data=group_names_pretty(group),
                hue='mouse_id',
                style='group',
                legend=False,
                palette=COLOR_PALETTES['mouse_id'],
                units='code',
                estimator=None
            )
            plt.xticks(ticks=lineage_bias_df[timepoint_col].unique())
            if log_scale:
                ax.set(yscale='log')
            plt.xlabel(timepoint_col.title())
            y_title = y_col.replace('percent_engraftment', 'Abundance (% WBC)').replace('_', ' ').title()
            plt.ylabel(y_title)
            plt.suptitle(
                'Group: ' \
                + phenotype.replace('_', ' ').title() \
                + ' Mice'
            )
            plt.title(title_addon)
            plt.ylim(ymin, ymax)
            fname = fname_prefix \
                + '_' + phenotype \
                + '.' + save_format
            save_plot(fname, save, save_format)

    else:
        plt.figure()
        ax = sns.lineplot(
            x=timepoint_col,
            y=y_col,
            data=group_names_pretty(lineage_bias_df),
            hue='group',
            palette=COLOR_PALETTES['group']
        )
        plt.xticks(ticks=lineage_bias_df[timepoint_col].unique())
        if log_scale:
            ax.set(yscale='log')
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

    if y_col != 'lineage_bias':
        lineage_bias_df = bias_clones_to_abundance(
            lineage_bias_df,
            clonal_abundance_df,
            y_col
        )
        
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
        first_last_df = filter_first_last_by_mouse(cell_df, timepoint_col)
        last_time_per_mouse_df = first_last_df[first_last_df.mouse_time_desc == 'Last']
        first_time_per_mouse_df = first_last_df[first_last_df.mouse_time_desc == 'First']

        last_time_per_mouse_df = last_time_per_mouse_df.sort_values(by='percent_engraftment', ascending=False)
        first_time_per_mouse_df = first_time_per_mouse_df.sort_values(by='percent_engraftment', ascending=False)
        last_time_per_mouse_df = last_time_per_mouse_df.assign(total=100)
        first_time_per_mouse_df = first_time_per_mouse_df.assign(total=100)

        sns.set(
            style="whitegrid",
            font_scale=2.0,
            rc={
                'xtick.labelsize': 22,
                'ytick.labelsize': 22,
                'axes.titlesize': 'xx-small',
                'figure.titlesize' : 'x-small'
            }
        )

        print('\nPlotting ' + cell_type.title())
        for group, g_df in last_time_per_mouse_df.groupby('group'):
            print(g_df[['mouse_id', 'percent_engraftment', timepoint_col, 'mouse_time_desc', 'change_status']])
            for m, m_df in g_df.groupby('mouse_id'):
                lymph = m_df[m_df.change_type == 'Lymphoid']
                myl = m_df[m_df.change_type == 'Myeloid']
                pd.options.mode.chained_assignment = None
                g_df[(g_df.mouse_id == m) & (g_df.change_type == 'Myeloid')].percent_engraftment = myl.percent_engraftment + lymph.percent_engraftment
                pd.options.mode.chained_assignment = 'warn'

            print(y_col_to_title(group) + ' Mice: ' + str(g_df.mouse_id.nunique()))
            plt.figure()
            ax = sns.barplot(
                x='total',
                y='mouse_id',
                data=g_df,
                color=COLOR_PALETTES['change_type']['Unchanged'],
                label="Unchanged"
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Myeloid'],
                color=COLOR_PALETTES['change_type']['Myeloid'],
                label='Myeloid',
                ax=ax,
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Lymphoid'],
                color=COLOR_PALETTES['change_type']['Lymphoid'],
                label='Lymphoid',
                ax=ax,
            )
            plt.xlabel('Contribution of Changed Cells')
            plt.ylabel('')
            plt.suptitle(cell_type.title() + ' Cumulative Abundance of Clones by Bias Change')
            plt.title('Group: ' + group.replace('_', ' ').title() + ' - Last Time Point')
            plt.gca().legend(title='Change Direction', loc='lower right').remove()
            sns.despine(left=True, bottom=True)

            fname = save_path + os.sep + 'percent_contribution_changed_' + cell_type + '_' + group + '.' + save_format
            save_plot(fname, save, save_format)

        for group, g_df in first_time_per_mouse_df.groupby('group'):
            for m, m_df in g_df.groupby('mouse_id'):
                lymph = m_df[m_df.change_type == 'Lymphoid']
                myl = m_df[m_df.change_type == 'Myeloid']
                pd.options.mode.chained_assignment = None
                g_df[(g_df.mouse_id == m) & (g_df.change_type == 'Myeloid')].percent_engraftment = myl.percent_engraftment + lymph.percent_engraftment
                pd.options.mode.chained_assignment = 'warn'

            plt.figure()
            ax = sns.barplot(
                x='total',
                y='mouse_id',
                data=g_df,
                color=COLOR_PALETTES['change_type']['Unchanged'],
                label='Unchanged'
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Myeloid'],
                color=COLOR_PALETTES['change_type']['Myeloid'],
                ax=ax,
                label='Myeloid'
            )
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                data=g_df[g_df.change_type == 'Lymphoid'],
                color=COLOR_PALETTES['change_type']['Lymphoid'],
                ax=ax,
                label='Lymphoid'
            )
            plt.xlabel('Contribution of Changed Cells')
            plt.ylabel('')
            plt.suptitle(cell_type.title() + ' Cumulative Abundance of Clones by Bias Change')
            plt.title('Group: ' + group.replace('_', ' ').title() + ' - First Time Point')
            plt.gca().legend(title='Change Direction', loc='lower right').remove()
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
        timepoint_col: str,
        by_group: bool = False,
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png'
    ) -> None:
    """ Swarm plot of abundance of mouse_clones

    Arguments:
        input_df {pd.DataFrame} -- abundance dataframe
        cell_type {str} -- cell type to plot
        abundance_cutoff {float} -- cumulative abundance cutoff to calculate threshold at
        thresholds {Dict[str, float]} -- thresholds to filter input by
        n_timepoints {int} -- number of timepoints required for a clone to have

    Keyword Arguments:
        by_group -- plot by group or not
        save {bool} -- save plot or not (default: {False})
        save_path {str} -- path to save file (default: {''})
        save_format {str} -- plot save format (default: {'png'})

    Returns:
        None -- plt.show() to view plot
    """

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 4,
            'axes.linewidth': 3,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )

    time_col = timepoint_col

    filtered_df = filter_cell_type_threshold(
        input_df,
        thresholds,
        analyzed_cell_types=[cell_type]
    )
    if abundance_cutoff == 0:
        print('\n ~~ Cutoff set to 0, due to number of clones plotting will take some time ~~ \n')

    print('Plotting for ' + cell_type.title() + ' Cells')
    if by_group:
        for group, g_df in filtered_df.groupby('group'):
            unique_time = g_df[timepoint_col].unique()
            print(
                Fore.CYAN + Style.BRIGHT 
                + '\nPerforming Wilcoxon test on ' + y_col_to_title(group)
                + ' ' + cell_type.title() + ' Clone Abundance Across Time'
            )
            for (t1, t2) in combinations(unique_time, 2):
                t1_df = g_df[g_df[timepoint_col] == t1]
                t2_df = g_df[g_df[timepoint_col] == t2]
                merged = t1_df.merge(
                    t2_df[['mouse_id', 'code', 'percent_engraftment']],
                    on=['mouse_id', 'code'],
                    how='inner',
                    validate='1:1',
                    suffixes=['_1', '_2']
                )
                _, p_value = stats.wilcoxon(
                    merged['percent_engraftment_1'],
                    merged['percent_engraftment_2'],
                )
                context: str = timepoint_col.title() + ' ' + str(t1) \
                + ' vs ' + str(t2) + ', ' + str(merged.code.nunique()) \
                + ' ' + cell_type.title() + ' Clones'
                print_p_value(context, p_value)
                
            plt.figure(figsize=(7, 5))
            plt.yscale('log')
            pal = COLOR_PALETTES['mouse_id']
            ax = sns.swarmplot(
                x=time_col,
                y='percent_engraftment',
                hue='mouse_id',
                data=g_df,
                palette=pal,
                zorder=0
                )
            sns.boxplot(
                x=time_col,
                y='percent_engraftment',
                data=g_df,
                boxprops={'facecolor': 'None'},
                ax=ax,
                fliersize=0,
            )

            
            title = cell_type.capitalize() \
                + ' > ' \
                + str(round(thresholds[cell_type],2)) + '% WBC'
            if abundance_cutoff != 0:
                plt.suptitle(title)
            plt.title('Group: ' + group.replace('_', ' ').title())
            plt.xlabel(time_col.title())
            plt.ylabel('Clone Abundance (% WBC)')
            plt.legend().remove()
            fname = save_path + os.sep + 'swamplot_abundance' \
                    + '_' + cell_type + '_a' \
                    + str(round(abundance_cutoff, 2)).replace('.','-') \
                    + '_' + group \
                    + '.' + save_format
            save_plot(fname, save, save_format)
    else:
        plt.figure(figsize=(7,5))
        plt.yscale('log')
        pal = COLOR_PALETTES['group']
        ax = sns.swarmplot(
            x=time_col,
            y='percent_engraftment',
            hue='group',
            data=filtered_df,
            palette=pal
            )
        sns.boxplot(
            x=time_col,
            y='percent_engraftment',
            data=filtered_df,
            boxprops={'facecolor': 'None'},
            ax=ax,
            fliersize=0,
        )

        title = cell_type.capitalize() \
            + ' > ' \
            + str(round(thresholds[cell_type],2)) + '% WBC'
        plt.suptitle(title)
        plt.xlabel(time_col.title())
        plt.ylabel('Clone Abundance (% WBC)')
        fname = save_path + os.sep + 'swamplot_abundance' \
                + '_' + cell_type + '_a' \
                + str(round(abundance_cutoff, 2)).replace('.','-') \
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
            estimator=np.median,
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
        timepoint_col: str,
        timepoint: float = None,
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
        bias_change_df = get_bias_change(filt_lineage_bias_df, timepoint_col)
        bias_change_df.to_csv(cache_dir + os.sep + 'bias_change_df_a'+str(round(abundance_cutoff, 2)) + '.csv', index=False)
        
    timepoint_text = ''
    if timepoint is not None:
        bias_change_df = filter_bias_change_timepoint(
            bias_change_df,
            timepoint
        )
        timepoint_text = ' - Clones Must have First or Last Time at: ' +str(timepoint)
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
    x, y, y1, y2, x_c, y_c = calculate_bias_change_cutoff(
        bias_change_df,
        min_time_difference=min_time_difference,
        kde=kde,
        timepoint=timepoint,
    )
    plt.plot(x, y1, c=COLOR_PALETTES['change_status']['Unchanged'])
    plt.plot(x, y2, c=COLOR_PALETTES['change_status']['Changed'])
    plt.scatter(x_c, y_c, c='k')
    plt.vlines(x_c[0], 0, max(y))
    kde.text(x_c[0] + .1, max(y)/1.1, 'Change at: ' + str(round(x_c[0], 3)))

    if group != 'all':
        plt.title(' Group: ' + y_col_to_title(group) + timepoint_text)
    plt.xlabel('Magnitude of Lineage Bias Change')
    plt.ylabel('Relative Clone Count')
    kde.legend_.remove()

    fname = save_path + os.sep \
        + 'bias_change_cutoff_a' + str(abundance_cutoff).replace('.', '-') \
        + '_' + group \
        + '_mtd' + str(min_time_difference) \
        + '_' + timepoint_col[0] + str(timepoint) \
        + '.' + save_format
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

    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 1.5,
            'axes.linewidth': 4,
            'axes.labelsize': 25,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 28,
            'ytick.labelsize': 22,
        }

        )
    
    # Find abundance data if not looking at lineage bias
    if y_col != 'lineage_bias':
        biased_at_time_df = bias_clones_to_abundance(
            biased_at_time_df,
            clonal_abundance_df,
            y_col
        )
        

    y_title = y_col_to_title(y_col)
    if invert_selection:
        plot_title = 'Bias within +/- ' \
            + str(round(bias_cutoff, 2)) + ' at ' \
            + timepoint_col.title() + ' ' + str(timepoint)
        if round(bias_cutoff, 2) == 0.71:
            plot_title = 'Balanced' \
                + ' At ' \
                + timepoint_col.title() + ' ' + str(timepoint)
    else:
        plot_title = 'Bias beyond ' \
            + str(round(bias_cutoff, 2)) + ' at ' \
            + timepoint_col.title() + ' ' + str(timepoint)
        
    if by_clone:
        for group, group_df in biased_at_time_df.groupby('group'):
            plt.figure(figsize=(7,6))
            fname_addon += '_by-clone'
            ax = sns.lineplot(
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
            ax.set_xticks(group_df[timepoint_col].unique())

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
                    + str(round(bias_cutoff, 2)).replace('.', '-') \
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
    
    bias_dist_df = calculate_first_last_bias_change_with_avg_data(
        lineage_bias_df,
        y_col,
        timepoint_col,
    )
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
                rug=e,
                hist=True,
                kde=False,
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

def plot_bias_dist_mean_abund_group_vs(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        cutoff: float,
        mtd: int,
        timepoint: int,
        y_col: str = 'sum_abundance',
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    sns.set_context(
        'paper',
        font_scale=2,
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 22,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'figure.titlesize': 'small',
        }

        )
    _, _, _, _, cutoffs, _ = calculate_bias_change_cutoff(
        get_bias_change(
            lineage_bias_df,
            timepoint_col,
        ),
        mtd,
        timepoint=timepoint
    )
    bias_change_cutoff = cutoffs[0]
    bias_dist_df = calculate_first_last_bias_change_with_avg_data(
        lineage_bias_df,
        y_col,
        timepoint_col
    )
    
    plt.figure(figsize=(7,6))
    for gname, g_df in bias_dist_df.groupby('group'):
        c = COLOR_PALETTES['group'][gname]
        filt_df = g_df[g_df['average_'+y_col] >= cutoff]
        sns.distplot(
            filt_df.bias_change,
            rug=False,
            kde=False,
            hist=True,
            hist_kws={
                'histtype': 'step',
                'linewidth': 3,
                'alpha': .8,
                'color': c
            },
            color=c,
            label=NEW_GROUP_NAME_MAP[gname],

        )

        plt.title(
            y_col_to_title('average_'+y_col) \
            + ' > ' + str(cutoff)
        )
        plt.xlabel('Change In Lineage Bias')
        plt.ylabel('Number of Clones')
        plt.xlim((-2,2))
        # Shrink current axis's height by 10% on the bottom
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            ncol=2)

    min_max_change = [bias_change_cutoff, -1 * bias_change_cutoff]
    ymin, ymax = plt.ylim()
    plt.vlines(min_max_change, 0, ymax, linestyles='dashed')
    fname = save_path + os.sep +'bias_change_dist_vs_group_' \
        + y_col \
        + '_' + str(cutoff).replace('.','-') \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_change_mean_scatter(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        y_col: str = 'sum_abundance',
        by_group: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    bias_dist_df = calculate_first_last_bias_change_with_avg_data(
        lineage_bias_df,
        y_col,
        timepoint_col
    )
    if by_group:
        for gname, g_df in bias_dist_df.groupby('group'):
            fig, ax = plt.subplots()
            ax.set(xscale='log')
            for mouse, m_df in g_df.groupby('mouse_id'):
                ax.scatter(
                    m_df['average_'+y_col],
                    m_df['bias_change'],
                    c=COLOR_PALETTES['mouse_id'][mouse],
                    s=16
                )
            plt.hlines(0, 0, g_df['average_'+y_col].max(), linestyles='dashed', color='lightgrey')

            plt.title('Group: ' + gname.replace('_', ' ').title())
            plt.suptitle(y_col_to_title('average_'+y_col))
            plt.xlabel(y_col_to_title('average_'+y_col) + ' (% WBC)')
            plt.ylabel('Overall Change in Bias Per Clone')
            fname = save_path + os.sep + y_col + '_vs_bias_change' \
                + '_' + gname \
                + '_logX' \
                + '.' + save_format
            save_plot(fname, save, save_format)
    else:
        fig, ax = plt.subplots()

        plt.xlim(bias_dist_df['average_'+y_col].min(), bias_dist_df['average_'+y_col].max())
        ax.set(xscale='log')
        for group, g_df in bias_dist_df.groupby('group'):
            ax.scatter(
                g_df['average_'+y_col],
                g_df['bias_change'],
                c=COLOR_PALETTES['group'][group],
                s=16
            )
        plt.hlines(0, 0, bias_dist_df['average_'+y_col].max(), linestyles='dashed', color='lightgrey')

        plt.title(
            y_col_to_title('average_'+y_col)
        )
        plt.xlabel(y_col_to_title('average_'+y_col) + ' (% WBC)')
        plt.ylabel('Overall Change in Bias Per Clone')
        fname = save_path + os.sep + y_col + '_vs_bias_change' \
            + '_logX' \
            + '.' + save_format
        save_plot(fname, save, save_format)

def moving_average(row):
    mean = row.mean()
    std = row.std()
    filt = (row - mean / std).abs() < 5
    #print(filt)
    filt_row = row[filt]
    return filt_row.mean()

def plot_dist_bias_over_time(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        by_group: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    num_points = 2000
    rolling_div = 10
    x_points = np.linspace(-1.8, 1.8, num_points)
    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]
    num_time_points = lineage_bias_df[timepoint_col].nunique()
    if by_group:
        for group, g_df in lineage_bias_df.groupby('group'):
            plt.figure(figsize=(7,6))
            plt.ylabel('')
            plt.title('Group: ' + group.replace('_', ' ').title())
            palette = COLOR_PALETTES[timepoint_col]
            pdfs = {}
            for t, t_df in g_df.groupby(timepoint_col):
                for mouse_id, m_df in t_df.groupby('mouse_id'):
                    try:
                        kde = stats.gaussian_kde(m_df.lineage_bias)
                    except np.linalg.LinAlgError:
                        print(Fore.YELLOW + 'Adding noise to: ' + mouse_id)
                        print('Unique Bias Before Noise:')
                        print(m_df.lineage_bias.unique())
                        print('Unique Bias After Noise:')
                        noisey = np.random.normal(m_df.lineage_bias, 0.001)
                        print(np.unique(noisey).tolist())
                        kde = stats.gaussian_kde(noisey)
                    m_kde = kde.pdf(x_points)
                    pdfs[mouse_id] = m_kde.tolist()
                pdfs = pd.DataFrame.from_dict(pdfs)
                pdfs['average_pdf']= pdfs.mean(axis=1)
                pdfs['rolling_average_pdf']= pdfs.average_pdf.rolling(round(num_points/rolling_div)).mean()
                plt.plot(
                    x_points,
                    pdfs.rolling_average_pdf,
                    color=palette[str(int(t))],
                    label=str(int(t)),
                    alpha=0.8,
                )

            plt.xlabel('Lineage Bias')
            # Shrink current axis's height by 10% on the bottom
            ax = plt.gca()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=num_time_points, title=timepoint_col.title())

            file_name = save_path + os.sep \
                + 'bias_dist_time_' \
                + group \
                + '.' + save_format
            save_plot(file_name, save, save_format)
    else:
        plt.figure()
        plt.suptitle(
            'Distribution of Lineage Bias at Each ' + timepoint_col.title()
        )
        plt.ylabel('')
        pal = sns.color_palette(
            palette=COLOR_PALETTES['time_point'],
            n_colors=lineage_bias_df[timepoint_col].nunique()
        )
        palette = COLOR_PALETTES[timepoint_col]
        pdfs = {}
        for t, t_df in lineage_bias_df.groupby(timepoint_col):
            for mouse_id, m_df in t_df.groupby('mouse_id'):
                if m_df.group.isna().any():
                    print(Fore.YELLOW + 'Skippping mouse: ' + mouse_id + ', no group')
                    continue
                else:
                    try:
                        kde = stats.gaussian_kde(m_df.lineage_bias)
                    except np.linalg.LinAlgError:
                        print(Fore.YELLOW + 'Adding noise to: ' + mouse_id)
                        print('Unique Bias Before Noise:')
                        print(m_df.lineage_bias.unique())
                        print('Unique Bias After Noise:')
                        noisey = np.random.normal(m_df.lineage_bias, 0.001)
                        print(np.unique(noisey).tolist())
                        kde = stats.gaussian_kde(noisey)

                m_kde = kde.pdf(x_points)
                pdfs[mouse_id] = m_kde.tolist()
            pdfs = pd.DataFrame.from_dict(pdfs)
            pdfs['average_pdf']= pdfs.mean(axis=1)
            pdfs['rolling_average_pdf']= pdfs.average_pdf.rolling(round(num_points/rolling_div)).mean()
            plt.plot(
                x_points,
                pdfs.rolling_average_pdf,
                color=palette[str(int(t))],
                label=str(int(t)),
            )
        plt.xlabel('Lineage Bias')
        plt.legend(title=timepoint_col.title())
        file_name = save_path + os.sep \
            + 'bias_dist_time.' + save_format
        save_plot(file_name, save, save_format)

def plot_dist_bias_at_time(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: int,
        thresholds: Dict[str, float],
        save: bool = False,
        abundance_cutoff: float = 0,
        save_path: str = './output',
        save_format: str = 'png',
        by_mouse: bool = False,
    ) -> None:
    i: int = 0
    if timepoint == 'last':
        t_df = find_last_clones_in_mouse(lineage_bias_df, timepoint_col)
    else:
        t_df = lineage_bias_df[lineage_bias_df[timepoint_col].isin([timepoint])]
    print(t_df)
    filt_df = filter_lineage_bias_anytime(
        t_df,
        thresholds
    )
    if by_mouse:
        for group, g_df in filt_df.groupby('group'):
            plt.figure()
            plt.suptitle('Abundance Cutoff: ' + str(abundance_cutoff))
            plt.title(
                'Distribution of Lineage Bias at ' \
                    + timepoint_col.title() \
                    + ' ' + str(timepoint)
            )
            plt.ylabel('')
            for mouse_id, m_df in g_df.groupby('mouse_id'):
                sns.distplot(
                    m_df.lineage_bias,
                    label=mouse_id,
                    color=COLOR_PALETTES['mouse_id'][mouse_id],
                    hist=False,
                    kde_kws={
                        "linewidth": 2,
                        "alpha": 1,
                    }
                )
            _, y_max = plt.ylim()
            if y_max > 2:
                plt.ylim(0, 2)
            plt.xlabel('Lineage Bias')
            plt.suptitle('Group: ' + group.replace('_', ' ').title())
            file_name = save_path + os.sep \
                + 'bias_dist_at_' + timepoint_col[0] \
                + str(timepoint) + '_by-mouse' \
                + group \
                + '_a' + str(abundance_cutoff).replace('.','-') \
                + '.' + save_format
            save_plot(file_name, save, save_format)
    else:
        plt.figure()
        plt.suptitle('Abundance Cutoff: ' + str(abundance_cutoff))
        plt.title(
            'Distribution of Lineage Bias at ' \
                + timepoint_col.title() \
                + ' ' + str(timepoint)
        )
        plt.ylabel('')
        for group, g_df in filt_df.groupby('group'):
            sns.distplot(
                g_df.lineage_bias,
                label=group.replace('_', ' ').title(),
                hist=False,
                color=COLOR_PALETTES['group'][group],
                kde_kws={
                    "linewidth": 2,
                    "alpha": 1,
                }
            )
        plt.xlabel('Lineage Bias')
        file_name = save_path + os.sep \
            + 'bias_dist_at_' + timepoint_col[0] \
            + str(timepoint) \
            + '_a' + str(abundance_cutoff).replace('.','-') \
            + '.' + save_format
        save_plot(file_name, save, save_format)

def plot_bias_first_last(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        cutoff: float,
        filter_col: str = 'sum_abundance',
        by_group: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ) -> None:
    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

    with_avg_abund = add_average_abundance_to_lineage_bias(
        lineage_bias_df,
    )

    with_time_desc = add_first_last_to_lineage_bias(
        with_avg_abund,
        timepoint_col
    )    
    with_time_desc_filt = with_time_desc[with_time_desc.time_description.isin(['First', 'Last'])]
    filt_df = with_time_desc_filt[with_time_desc_filt['average_'+filter_col] >= cutoff]
    if by_group:
        for group, g_df in filt_df.groupby('group'):
            plt.figure()
            ax = sns.violinplot(
                x='time_description',
                y='lineage_bias',
                color='white',
                data=g_df,
                dodge=False,
                cut=0,
            )
            sns.swarmplot(
                x='time_description',
                y='lineage_bias',
                hue='mouse_id',
                palette=COLOR_PALETTES['mouse_id'],
                data=g_df,
                ax=ax,
            )
            plt.legend().remove()
            plt.ylabel('Lineage Bias')
            plt.xlabel('Time Point')
            plt.suptitle('Lineage Bias of Clones: ' + group.replace('_', ' ').title())
            plt.title(
                'Filtered by Clones With ' \
                + y_col_to_title('average_'+filter_col) \
                + ' > ' + str(cutoff)
                )

            file_name = save_path + os.sep \
                + 'first-last-lineage-bias' \
                + '_' + filter_col \
                + '_t' + str(cutoff).replace('.', '-') \
                + '_' + group \
                + '.' + save_format
            save_plot(file_name, save, save_format)
    else:
        plt.figure()
        ax = sns.violinplot(
            x='time_description',
            y='lineage_bias',
            color='white',
            data=filt_df,
            dodge=False,
            cut=0,
        )
        sns.swarmplot(
            x='time_description',
            y='lineage_bias',
            hue='group',
            palette=COLOR_PALETTES['group'],
            data=filt_df,
            ax=ax,
        )
        plt.legend().remove()
        plt.ylabel('Lineage Bias')
        plt.xlabel('Time Point')
        plt.suptitle('Lineage Bias of Clones First And Last Time Point')
        plt.title(
            'Filtered by Clones With ' \
            + y_col_to_title('average_'+filter_col) \
            + ' > ' + str(cutoff)
            )

        file_name = save_path + os.sep \
            + 'first-last-lineage-bias' \
            + '_' + filter_col \
            + '_t' + str(cutoff).replace('.', '-') \
            + '.' + save_format
        save_plot(file_name, save, save_format)

def plot_abundant_clone_survival(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        cell_type: str,
        cumulative: bool,
        by_mouse: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    if timepoint_col == 'gen':
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df[timepoint_col] != 8.5]
    transplant_survival_df = abundant_clone_survival(
        clonal_abundance_df,
        timepoint_col,
        thresholds,
        [cell_type],
        cumulative
    )
    time_change = 'between'
    if cumulative:
        time_change = 'across'
    
    fname_prefix = save_path + os.sep \
        + 'survival_' + time_change \
        + '_' + timepoint_col + '_' \
        + cell_type + '_a' \
        + str(round(thresholds[cell_type],2)).replace('.','-')
    fname_suffix = '.' + save_format


    if by_mouse:
        for group, g_df in transplant_survival_df.groupby('group'):
            plt.figure()
            plt.suptitle(
                'Survival of Clones with ' + cell_type.title()
                + ' Abundance > ' + str(round(thresholds[cell_type], 2))
                + ' Before Change'
            )
            plt.title('Group: ' + group.replace('_', ' ').title())
            sns.barplot(
                x='time_change',
                y='percent_survival',
                data=g_df,
                hue='mouse_id',
                palette=COLOR_PALETTES['mouse_id']
            )
            plt.xlabel(timepoint_col.title() + ' Change')
            plt.ylabel('Percent Survival')
            plt.legend().remove()
            file_name = fname_prefix \
                + '_' + group + fname_suffix
            save_plot(file_name, save, save_format)
    else:
        plt.figure()
        plt.title(
            'Survival of Clones with ' + cell_type.title()
            + ' Abundance > ' + str(round(thresholds[cell_type], 2))
            + ' Before Change'
        )
        sns.barplot(
            x='time_change',
            y='percent_survival',
            data=transplant_survival_df,
            hue='group',
            palette=COLOR_PALETTES['group']
        )
        plt.xlabel(timepoint_col.title() + ' Change')
        plt.ylabel('Percent Survival')
        file_name = fname_prefix + fname_suffix
        save_plot(file_name, save, save_format)

def plot_not_survived_count_mouse(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    not_survived_df = not_survived_bias_by_time_change(
        lineage_bias_df,
        timepoint_col,
        ignore_bias_cat=True
    )
    sns.set(style='whitegrid')
    for group, g_df in not_survived_df.groupby('group'):
        plt.subplots(figsize=(6,5))
        sns.lineplot(
            x='time_survived',
            y='count',
            hue='mouse_id',
            markers=True,
            dashes=False,
            palette=COLOR_PALETTES['mouse_id'],
            data=g_df
        )
        plt.suptitle(
            'Survival Of Clones Over Time'
        )
        plt.title('Group: ' + group.replace('_', ' ').title())
        plt.xlabel(
            timepoint_col.title()
            + '(s) Survived'
        )
        plt.ylabel('Unique Clones Per Mouse')
        plt.legend().remove()
        fname = save_path + os.sep \
            + 'clone_count_survival' \
            + '_' + group \
            + '_' + 'by_mouse' \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_not_survived_by_bias(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        group: str = 'all',
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    if group != 'all':
            lineage_bias_df = lineage_bias_df[lineage_bias_df['group'] == group]

    fig, ax = plt.subplots(figsize=(8,7))
    not_survived_df = not_survived_bias_by_time_change(
        lineage_bias_df,
        timepoint_col
    )
    width = 0.35
    not_survived_df['bias_category'] = not_survived_df.bias_category.apply(
        lambda x: MAP_LINEAGE_BIAS_CATEGORY[x]
    )
    cats = [
        'LC',
        'LB',
        'BL',
        'B',
        'BM',
        'MB',
        'MC'
    ]
    cats = [MAP_LINEAGE_BIAS_CATEGORY[x] for x in cats]
    means = pd.DataFrame(not_survived_df.groupby(['bias_category', 'time_survived'])['count'].mean()).reset_index()
    means = means.pivot(
        index='time_survived',
        columns='bias_category',
        values='count'
    )
    sem = pd.DataFrame(not_survived_df.groupby(['bias_category', 'time_survived'])['count'].sem()).reset_index()
    sem = sem.pivot(
        index='time_survived',
        columns='bias_category',
        values='count'
    )

    myeloid_color = Color(COLOR_PALETTES['change_type']['Myeloid'])
    myeloid_colors = list(Color('white').range_to(
        myeloid_color,
        int(round(len(cats)/2)) + 2
    ))
    lymphoid_color = Color(COLOR_PALETTES['change_type']['Lymphoid'])
    lymphoid_colors = list(lymphoid_color.range_to(
        Color('white'),
        int(round(len(cats)/2)) + 1
    ))
    colors = lymphoid_colors[:-2] \
        + [Color(COLOR_PALETTES['change_type']['Unchanged'])] \
        + myeloid_colors[2:]
    colors = [x.hex_l for x in colors]

    x = means.index.values
    for i in range(len(cats)):
        bias_cat = cats[i]
        m_df = means[bias_cat]
        e_df = sem[bias_cat]
        if i != 0:
            last_df = means[cats[:i]].sum(axis=1)
            plt.bar(
                x,
                m_df,
                width,
                yerr=e_df,
                color=colors[i],
                label=bias_cat,
                bottom=last_df
            )
        else:
            plt.bar(
                x,
                m_df,
                width,
                yerr=e_df,
                color=colors[i],
                label=bias_cat
            )
    plt.xlabel(
        timepoint_col.title()
        + '(s) Survived'
    )
    plt.ylabel('Unique Clones Per Mouse')
    if group == 'all':
        plt.legend()
    plt.suptitle(
        'Survival Of Clones Over Time'
    )
    plt.title('Group: ' + group.replace('_', ' ').title())
    fname = save_path + os.sep \
        + 'clone_count_survival' \
        + '_' + group \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_not_survived_abundance(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    survival_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )
    
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 1.5,
            'axes.linewidth': 4,
            'axes.labelsize': 25,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 28,
            'ytick.labelsize': 22,
        }

        )
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Rank Sum Test of Exhausted vs. Survived'
    )
    for group, g_df in survival_df.groupby('group'):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\n  - Group: ' + group.replace('_', ' ').title()
        )
        fig, ax = plt.subplots(figsize=(7,5))
        # T-Test on interesting result

        for time_change, t_df in g_df.groupby('time_change'):
            t_s = t_df[t_df['survived'] == 'Survived']
            t_e = t_df[t_df['survived'] == 'Exhausted']
            stat, p_value = stats.ranksums(
                t_e.accum_abundance,
                t_s.accum_abundance,
            )
            context: str = timepoint_col.title() + ' ' + str(int(time_change))
            print_p_value(context, p_value)
            

        ax = sns.boxplot(
            x=timepoint_col,
            y='accum_abundance',
            hue='survived',
            data=g_df,
            hue_order=['Exhausted', 'Survived']
        )
        ax.set(yscale='log')
        plt.xlabel(
           'End Point (' + timepoint_col.title() + ')'
        )
        plt.legend().remove()
        plt.ylabel('Average Abundance')
        plt.suptitle(
            'Exhausted Clone Abundance'
        )
        plt.title('Group: ' + group.replace('_', ' ').title())
        fname = save_path + os.sep \
            + 'abundance_not_survived' \
            + '_' + group \
            + '.' + save_format
        save_plot(fname, save, save_format)
    group = 'all'
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\n  - Group: ' + group.replace('_', ' ').title()
    )
    fig, ax = plt.subplots(figsize=(10,8))
    # T-Test on interesting result
    for time_change, t_df in survival_df.groupby('time_change'):
        t_s = t_df[t_df['survived'] == 'Survived']
        t_e = t_df[t_df['survived'] == 'Exhausted']
        stat, p_value = stats.ranksums(
            t_e.accum_abundance,
            t_s.accum_abundance,
        )
        context: str = timepoint_col.title() + ' ' + str(int(time_change))
        print_p_value(context, p_value)
        

    ax = sns.boxplot(
        x=timepoint_col,
        y='accum_abundance',
        hue='survived',
        data=survival_df,
        hue_order=['Exhausted', 'Survived']
    )
    ax.set(yscale='log')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
        fancybox=True, shadow=True, ncol=2)

    plt.xlabel(
        timepoint_col.title()
        + '(s) Survived'
    )
    plt.ylabel('Average Abundance Until Last Time Point')
    plt.suptitle(
        'Abundance Of Not Surviving Clones Over Time'
    )
    plt.title('Group: ' + group.replace('_', ' ').title())
    plt.xlabel(
        timepoint_col.title()
    )
    plt.legend().remove()
    plt.ylabel('Average Abundance')
    plt.suptitle(
        'Exhausted Clone Abundance'
    )
    fname = save_path + os.sep \
        + 'abundance_not_survived' \
        + '_' + group \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_not_survived_count_box(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )
    labeled_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )
    not_survived_df = labeled_df[labeled_df.time_change == labeled_df.total_time_change]
    min_timepoint = lineage_bias_df[timepoint_col].min()
    not_survived_df = not_survived_df.assign(
        last_timepoint= lambda x: (x.time_change + min_timepoint).astype('int32').astype(str)
    )
    all_clone_num = lineage_bias_df.groupby(['mouse_id']).code.nunique().mean()

    # Plot All Clones
    count_df = pd.DataFrame(not_survived_df.groupby(
        ['mouse_id', 'group', 'last_timepoint', 'time_change']).code.nunique()
        ).reset_index().sort_values(by='time_change')
    fig = plt.figure(figsize=(8,6))
    ax = sns.boxplot(
        x='last_timepoint',
        y='code',
        order=count_df.last_timepoint.unique(),
        data=count_df,
        palette=COLOR_PALETTES[timepoint_col]
    )
    _, max_codes = plt.ylim()
    ax2 = ax.twinx()
    ax.set_ylim(0, max_codes)
    ax.set_ylabel('# Of Exhausted Clones')
    ax2.set_ylim(0, 100 * max_codes/all_clone_num)
    ax2.set_ylabel('% Average Unique Clones Per Mouse')

    ax.set_xlabel('End Point (' + timepoint_col.title() + ')')
    plt.title(
        'Count of Exhausted Clones Over Time'
    )
    fname = save_path + os.sep \
        + 'not_survived_count' \
        + '.' + save_format
    save_plot(fname, save, save_format)

    # Plot Clones By Group
    for group, g_df in not_survived_df.groupby('group'):
        count_df = pd.DataFrame(
            g_df.groupby(['mouse_id', 'group', 'last_timepoint', 'time_change']).code.nunique()
            ).reset_index().sort_values(by='time_change')
        fig = plt.figure(figsize=(10,7))
        ax = sns.boxplot(
            x='last_timepoint',
            y='code',
            order=count_df.last_timepoint.unique(),
            data=count_df.sort_values(by='time_change'),
            palette=COLOR_PALETTES[timepoint_col]
        )
        _, max_codes = plt.ylim()
        ax2 = ax.twinx()
        ax.set_ylim(0, max_codes)
        ax.set_ylabel('# Of Exhausted Clones')
        ax2.set_ylim(0, 100 * max_codes/all_clone_num)
        ax2.set_ylabel('% Average Unique Clones Per Mouse')

        ax.set_xlabel('End Point (' + timepoint_col.title() + ')')
        plt.suptitle(
            'Count of Exhausted Clones Over Time'
        )
        plt.title('Group: ' + group.replace('_', ' ').title())
        fname = save_path + os.sep \
            + 'not_survived_count' \
            + '_' + group \
            + '.' + save_format
        save_plot(fname, save, save_format)


def plot_hsc_abund_bias_at_last(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_group: bool,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:

    if clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].empty:
        raise ValueError('No HSC Cells in Clonal Abundance Data')
    cats_order = [
        'LC',
        'LB',
        'B',
        'MB',
        'MC'
    ]
    cats_order = [MAP_LINEAGE_BIAS_CATEGORY_SHORT[x] for x in cats_order]
    colors = get_myeloid_to_lymphoid_colors(cats_order)
    palette = dict(zip(cats_order, colors))

    last_clones = find_last_clones(
        lineage_bias_df,
        timepoint_col
    )
    labeled_last_clones = add_bias_category(
        last_clones
    )
    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']
    myeloid_hsc_abundance_df = hsc_data.merge(
        labeled_last_clones[['code','mouse_id','bias_category_short']],
        on=['code','mouse_id'],
        how='inner',
        validate='m:m'
    )

    unique_cats = list(OrderedDict.fromkeys(cats_order))
    if by_group:
        plt.figure(figsize=(12,8))
        ax = sns.barplot(
            y='percent_engraftment',
            x='bias_category_short',
            hue='group',
            hue_order=['Aging Phenotype', 'No Change'],
            order=unique_cats,
            data=group_names_pretty(myeloid_hsc_abundance_df),
            palette=COLOR_PALETTES['group'],
            saturation=1,
        )
        ax.set(yscale='log')
        plt.xlabel('')
        plt.ylabel('HSC Relative Abundance (% HSCs)')
        plt.title(
            'HSC Abundance by Bias at Last Time Point'
            )

        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test of HSC Abundance Between Phenotypic Groups At Bias Types\n'
        )
        for cat, c_df in myeloid_hsc_abundance_df.groupby('bias_category_short'):
            aging_vals = c_df[c_df['group'] == 'aging_phenotype']
            non_vals = c_df[c_df['group'] == 'no_change']
            stat, p_value = stats.ttest_ind(
                aging_vals.percent_engraftment,
                non_vals.percent_engraftment,
            )
            context: str = cat
            print_p_value(context, p_value)

        fname = save_path + os.sep \
            + 'abund_hsc_biased_at_last_by-group' \
            + '.' + save_format
        save_plot(fname, save, save_format)
    else:
        plt.figure(figsize=(12,8))
        ax = sns.barplot(
            y='percent_engraftment',
            x='bias_category_short',
            order=unique_cats,
            data=group_names_pretty(myeloid_hsc_abundance_df),
            palette=palette,
        )
        ax.set(yscale='log')
        plt.xlabel('')
        plt.ylabel('HSC Abundance (% HSCs)')
        plt.title(
            'HSC Abundance by Bias at Last Time Point'
            )

        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test of HSC Abundance Between Bias Types\n'
        )
        for (a, b) in combinations(unique_cats,2):
            a_vals = myeloid_hsc_abundance_df[myeloid_hsc_abundance_df['bias_category_short'] == a]
            b_vals = myeloid_hsc_abundance_df[myeloid_hsc_abundance_df['bias_category_short'] == b]
            stat, p_value = stats.ttest_ind(
                a_vals.percent_engraftment,
                b_vals.percent_engraftment,
            )
            context: str = a + ' vs ' + b
            print_p_value(context, p_value)

        fname = save_path + os.sep \
            + 'abund_hsc_biased_at_last' \
            + '.' + save_format
        save_plot(fname, save, save_format)


def plot_change_marked(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        y_col: str,
        by_clone: bool,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:
    bias_change_df = get_bias_change(
        lineage_bias_df,
        timepoint_col
    )
    plot_clones_df = lineage_bias_df
    if y_col != 'lineage_bias':
        plot_clones_df = bias_clones_to_abundance(
            lineage_bias_df,
            clonal_abundance_df,
            y_col
        )
    changed_marked_df = mark_changed(
        plot_clones_df,
        bias_change_df,
        min_time_difference=mtd
    )
    fname_prefix = save_path + os.sep + 'change-status_mtd' + str(mtd) + '_' + y_col + '_'
    fname_suffix = '.' + save_format
    if by_clone:
        for (group, change_status), g_df in changed_marked_df.groupby(['group', 'change_status']):
            plt.figure()
            ax = sns.lineplot(
                x=timepoint_col,
                y=y_col,
                data=g_df.sort_values(by='change_type'),
                hue='mouse_id',
                #style='change_type',
                units='code',
                estimator=None,
                palette=COLOR_PALETTES['mouse_id']
            )
            plt.title(
                'Group: ' + group.replace('_', ' ').title()
                + ', ' + change_status
                )
            plt.ylabel(y_col_to_title(y_col))
            plt.xlabel(timepoint_col.title())

            # Remove legend items for mouse_id
            #handles, labels = ax.get_legend_handles_labels()
            #short_handles = handles[labels.index('change_type')+1:]
            #short_labels= labels[labels.index('change_type')+1:]
            #plt.legend(handles=short_handles, labels=short_labels)
            plt.legend().remove()
            fname = fname_prefix + group \
                + '_' + change_status + '_by-clone' + fname_suffix
            save_plot(fname, save, save_format)
    else:
        group = 'all'
        for change_status, c_df in changed_marked_df.groupby(['change_status']):
            plt.figure()
            ax = sns.lineplot(
                x=timepoint_col,
                y=y_col,
                data=c_df.sort_values(by='change_type'),
                hue='group',
                style='change_type',
                estimator=np.median,
                palette=COLOR_PALETTES['group']
            )
            plt.title(
                    change_status.title()
                )
            plt.ylabel(y_col_to_title('median_'+y_col))
            plt.xlabel(timepoint_col.title())
            # Remove legend items for phenotype
            handles, labels = ax.get_legend_handles_labels()
            short_handles = handles[labels.index('change_type')+1:]
            short_labels= labels[labels.index('change_type')+1:]
            plt.legend(handles=short_handles, labels=short_labels)

            # Save
            fname = fname_prefix + group \
                + '_' + change_status + fname_suffix
            save_plot(fname, save, save_format)

def plot_stable_abund_time_clones(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        bias_change_cutoff: float,
        abund_timepoint: Any,
        t1: int,
        timepoint_col: str,
        thresholds: Dict[str,float],
        cell_type: str,
        y_col: str,
        save: bool,
        save_path: str ,
        save_format: str
    ) -> None:

    stable_clone_df = filter_stable_initially(
        lineage_bias_df,
        t1=t1,
        timepoint_col=timepoint_col,
        bias_change_cutoff=bias_change_cutoff,
    )
    abundant_at_timepoint = find_enriched_clones_at_time(
        input_df=clonal_abundance_df,
        enrichment_time=abund_timepoint,
        enrichment_threshold=thresholds[cell_type],
        cell_type=cell_type,
        timepoint_col=timepoint_col,
        lineage_bias=False,
    )
    fname_prefix = save_path + os.sep + 'stable_clones_' \
        + cell_type + '_abund_' + timepoint_col[0] + str(abund_timepoint)
    y_title = y_col_to_title(y_col)
    x_title = timepoint_col.title()

    if y_col != 'lineage_bias':
        stable_clone_df = bias_clones_to_abundance(
            stable_clone_df,
            abundant_at_timepoint,
            y_col
        )
    else:
        stable_clone_df = stable_clone_df.merge(
            abundant_at_timepoint[['mouse_id', 'code']],
            how='inner',
            on=['mouse_id', 'code'],
            validate="m:m"
        )

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
    title = 'Initial Bias Change < ' +str(bias_change_cutoff) \
        + ' ' + cell_type.title() + ' Abundance at ' + timepoint_col.title() + ' ' \
        + str(abund_timepoint).title() + ' > ' + str(round(thresholds[cell_type], 2))
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
def plot_perc_survival_bias(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        by_clone: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    fname_prefix = save_path + os.sep
    if by_clone:
        y_col = 'exhausted_count'
        y_label = 'Number Per Mouse of Exhausted Clones'
        fname_prefix += 'count_survive_bias_'
    else:
        y_col = 'exhausted_perc'
        y_label = 'Percent of Exhausted Clones Within Category'
        fname_prefix += 'perc_survive_bias_'

    survival_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )


    cats = [
        'LC',
        'LB',
        'B',
        'MB',
        'MC'
    ]
    cats = [MAP_LINEAGE_BIAS_CATEGORY[x] for x in cats]
    colors = get_myeloid_to_lymphoid_colors(cats)
    palette = dict(zip(cats, colors))
    survival_perc = calculate_survival_perc(
        survival_df,
        lineage_bias_df,
        timepoint_col,
    )
    survival_perc['bias_category'] = survival_perc.bias_category.apply(
        lambda x: MAP_LINEAGE_BIAS_CATEGORY[x]
    )

    for group, g_df in survival_perc.groupby('group'):
        plt.figure(figsize=(7,5))
        sns.barplot(
            y=y_col,
            x='last_time',
            hue='bias_category',
            hue_order=cats,
            palette=palette,
            data=g_df,
            capsize=.05,
            errwidth=.9,
            saturation=1
        )
        plt.legend(title='').remove()
        plt.ylabel(y_label)
        plt.xlabel('Last Time Point of Exhausted Clones')
        plt.title('Group: ' + y_col_to_title(group))
        fname = fname_prefix + group + '.' + save_format
        save_plot(fname, save, save_format)

    group = 'all'
    plt.figure(figsize=(10,9))
    sns.barplot(
        y=y_col,
        x='last_time',
        hue='bias_category',
        hue_order=cats,
        palette=palette,
        data=survival_perc,
        capsize=.05,
        saturation=1,
        errwidth=.9,
    )
    plt.legend(title='')
    plt.title('Group: ' + y_col_to_title(group))
    plt.ylabel(y_label)
    plt.xlabel('Last Time Point of Exhausted Clones')
    fname = fname_prefix + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_dist_by_change(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        group: str,
        timepoint: Any = None,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 4,
            'axes.linewidth': 3,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )
    bias_change_df = get_bias_change(
        lineage_bias_df,
        timepoint_col
    )
    change_marked_df = mark_changed(
        lineage_bias_df,
        bias_change_df,
        min_time_difference=mtd,
        timepoint=timepoint
    )
    if group != 'all':
        change_marked_df = change_marked_df[change_marked_df.group == group]
    
    fname_prefix = save_path + os.sep \
        + 'lineage_bias_by_change_' \
        + 't' + str(timepoint) \
        + '_mtd' + str(mtd)
    for time, t_df in change_marked_df.groupby(timepoint_col):
        plt.figure()
        plt.suptitle('Group: ' + y_col_to_title(group))
        plt.title(
            'Distrubution of Lineage Bias of Clones at '
            + timepoint_col.title() + ' ' + str(time))
        for status, c_df in t_df.groupby('change_status'):
            if status == 'Unchanged':
                continue
            sns.distplot(
                c_df['lineage_bias'],
                color=COLOR_PALETTES['change_status'][status],
                label=status,
                bins=20,
                hist=True,
                hist_kws={
                    "histtype": "step",
                    "linewidth": 2,
                    "alpha": 1,
                },
                rug=False,
                kde=False,
            )
        plt.xlabel('Lineage Bias Distribution')
        plt.ylabel('Clone Count')
        plt.legend(title='Lineage Bias Change Type').remove()
        fname = fname_prefix + '_at_' + timepoint_col[0] + str(time) \
            + '_' + group \
            + '.' + save_format
        save_plot(fname, save, save_format)
    time='last'
    plt.figure()
    plt.suptitle('Group: ' + y_col_to_title(group))
    plt.title(
        'Distrubution of Lineage Bias of Clones at '
        + timepoint_col.title() + ' ' + str(time))
    t_df = filter_first_last_by_mouse(
        change_marked_df,
        timepoint_col
    )
    t_df = t_df[t_df.mouse_time_desc == 'Last']
    for status, c_df in t_df.groupby('change_status'):
        if status == 'Unchanged':
            continue
        sns.distplot(
            c_df['lineage_bias'],
            label=status,
            hist=True,
            bins=20,
            hist_kws={
                "histtype": "step",
                "linewidth": 2,
                "alpha": .9,
            },
            rug=False,
            kde=False,
            color=COLOR_PALETTES['change_status'][status],
        )
    plt.xlabel('Lineage Bias Distribution')
    plt.ylabel('Clone Count')
    plt.legend(title='Lineage Bias Change Type').remove()
    fname = fname_prefix + '_at_' + timepoint_col[0] + str(time) \
        + '_' + group \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_abundance_by_change(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        timepoint: Any = None,
        sum: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    bias_change_df = get_bias_change(
        lineage_bias_df,
        timepoint_col
    )
    change_marked_df = mark_changed(
        clonal_abundance_df,
        bias_change_df,
        min_time_difference=mtd,
        timepoint=timepoint
    )
    fname_prefix = save_path + os.sep \
        + 'abundance_by_bias_change' \
        + 't' + str(timepoint) \
        + '_mtd' + str(mtd)
    desc_add = ''
    if sum:
        change_marked_df = pd.DataFrame(
            change_marked_df.groupby([
                'mouse_id',
                'group',
                'cell_type',
                'change_type',
                timepoint_col,
            ]).percent_engraftment.sum()
        ).reset_index()
        desc_add = 'sum'
    
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Independent T-Test Between Change Status '+ desc_add.title() + ' Abundance at each time point'
    )
    for (group, cell_type), c_df in change_marked_df.groupby(['group', 'cell_type']):
        plt.figure(figsize=(7,5))
        plt.title(
            cell_type.title() + ' '
            + desc_add.title() 
            + ' Abundance by Lineage Bias Change'
        )
        plt.suptitle('Group: ' + y_col_to_title(group))
        ax = sns.boxplot(
            x=timepoint_col,
            y='percent_engraftment',
            data=c_df,
            hue='change_type',
            hue_order=['Unchanged', 'Lymphoid', 'Myeloid'],
            palette=COLOR_PALETTES['change_type'],
        )
        ax.set(yscale='log')
        plt.xlabel(timepoint_col.title())
        plt.ylabel(y_col_to_title(cell_type+'_percent_engraftment'))
        plt.legend().remove()
        fname = fname_prefix + '_' + cell_type \
            + '_' + group \
            + '_' + desc_add \
            + '.' + save_format
        save_plot(fname, save, save_format)
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\n  - Group: ' + group.replace('_', ' ').title()
            + '  Cell Type: ' + cell_type.title()
        )
        for timepoint, t_df in c_df.groupby(timepoint_col):
            for (a, b) in combinations(['Unchanged', 'Myeloid', 'Lymphoid'],2):
                a_vals = t_df[t_df['change_type'] == a]
                b_vals = t_df[t_df['change_type'] == b]
                stat, p_value = stats.ttest_ind(
                    a_vals['percent_engraftment'],
                    b_vals['percent_engraftment'],
                )
                context: str = str(timepoint) + ' - ' + a + ' vs ' + b
                print_p_value(context, p_value)

    group = 'all'
    for cell_type, c_df in change_marked_df.groupby(['cell_type']):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\n  - Group: ' + group.replace('_', ' ').title()
            + '  Cell Type: ' + cell_type.title()
        )
        for timepoint, t_df in c_df.groupby(timepoint_col):
            for (a, b) in combinations(['Unchanged', 'Myeloid', 'Lymphoid'],2):
                a_vals = t_df[t_df['change_type'] == a]
                b_vals = t_df[t_df['change_type'] == b]
                stat, p_value = stats.ttest_ind(
                    a_vals['percent_engraftment'],
                    b_vals['percent_engraftment'],
                )
                context: str = str(timepoint) + ' - ' + a + ' vs ' + b
                print_p_value(context, p_value)
        plt.figure(figsize=(10,9))
        plt.title(
            cell_type.title() + ' '
            + desc_add.title()
            + ' Abundance by Lineage Bias Change'
        )
        ax = sns.boxplot(
            x=timepoint_col,
            y='percent_engraftment',
            data=c_df,
            hue='change_type',
            hue_order=['Unchanged', 'Lymphoid', 'Myeloid'],
            palette=COLOR_PALETTES['change_type'],
        )
        ax.set(yscale='log')
        plt.xlabel(timepoint_col.title())
        plt.ylabel(y_col_to_title(cell_type+'_percent_engraftment'))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            fancybox=True, shadow=True, ncol=3, title='')
        fname = fname_prefix + '_' + cell_type \
            + '_' + desc_add \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_bias_dist_contribution_over_time(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        cell_type: str,
        by_group: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

    n_timepoints = lineage_bias_df[timepoint_col].nunique()
    if by_group:
        for group, g_df in lineage_bias_df.groupby('group'):
            plt.figure(figsize=(7,5))
            plt.suptitle(
                'Blood Contribution by Lineage Bias at Each ' + timepoint_col.title()
            )
            plt.ylabel(cell_type.title() + ' Contribution (% WBC)')
            plt.title('Group: ' + group.replace('_', ' ').title())
            palette = COLOR_PALETTES[timepoint_col]
            for t, t_df in g_df.groupby(timepoint_col):
                sns.distplot(
                    t_df.lineage_bias,
                    label=str(int(t)),
                    bins=15,
                    hist=True,
                    kde=False,
                    hist_kws={
                        "histtype": "step",
                        "linewidth": 2,
                        "alpha": 1,
                        "color": palette[str(int(t))],
                        "weights": t_df[cell_type + '_percent_engraftment']
                    }
                )

            plt.xlabel('Lineage Bias')
            plt.legend().remove()
            file_name = save_path + os.sep \
                + 'bias_dist_time_' \
                + cell_type + '_' \
                + group \
                + '.' + save_format
            save_plot(file_name, save, save_format)
    else:
        plt.figure(figsize=(10,9))
        ax = plt.subplot(111)
        plt.title(
            'Blood Contribution by Lineage Bias at Each ' + timepoint_col.title()
        )
        palette = COLOR_PALETTES[timepoint_col]
        plt.ylabel(cell_type.title() + ' Contribution (% WBC)')
        for t, t_df in lineage_bias_df.groupby(timepoint_col):
            sns.distplot(
                t_df.lineage_bias,
                label=str(int(t)),
                bins=15,
                hist=True,
                kde=False,
                hist_kws={
                    "histtype": "step",
                    "linewidth": 2,
                    "alpha": 1,
                    "color": palette[str(int(t))],
                    "weights": t_df[cell_type + '_percent_engraftment']
                }
            )
        plt.xlabel('Lineage Bias')
        plt.legend(title=timepoint_col.title())
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
                fancybox=True, shadow=False, ncol=n_timepoints)
        file_name = save_path + os.sep \
            + cell_type + '_' \
            + 'bias_dist_time.' + save_format
        save_plot(file_name, save, save_format)

def plot_n_most_abundant(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        n: int, 
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 4,
            'axes.linewidth': 3,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.major.size': 4,
            'ytick.minor.size': 4,
            'xtick.labelsize': 28,
            'ytick.labelsize': 28,
            'figure.titlesize': 'medium',
        }
        )

    most_abund_contrib = pd.DataFrame(
        clonal_abundance_df.groupby(
            ['mouse_id', 'group', timepoint_col, 'cell_type']
        )['percent_engraftment'].apply(lambda x: x.nlargest(n).sum())
    ).reset_index()

    for cell_type, c_df in most_abund_contrib.groupby(['cell_type']):
        plt.figure(figsize=(7,5))
        ax = sns.swarmplot(
            x=timepoint_col,
            y='percent_engraftment',
            hue='group',
            palette=COLOR_PALETTES['group'],
            dodge=True,
            size=8,
            data=c_df,
        )
        sns.boxplot(
            x=timepoint_col,
            y='percent_engraftment',
            hue='group',
            palette=COLOR_PALETTES['group'],
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            showcaps=False,
            ax=ax,
            data=c_df,
        )
        ax.set(yscale='log')
        plt.title(
            cell_type.title()
            + ' Top ' + str(n) + ' Clones'
        )
        plt.ylabel('Cumulative Abundance (% WBC)')
        plt.xlabel(timepoint_col.title())
        plt.legend().remove()
        file_name = save_path + os.sep \
            + 'top' + str(n) \
            + '_sum-abundance' \
            + '_' + cell_type \
            + '.' + save_format

        save_plot(file_name, save, save_format)
        
def plot_clone_count_swarm(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        analyzed_cell_types: List[str],
        line: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )
    threshold_df = filter_cell_type_threshold(
        clonal_abundance_df,
        thresholds, 
        analyzed_cell_types
    )
        
    clone_counts = count_clones(threshold_df, timepoint_col)

    for cell_type, c_df in clone_counts.groupby(['cell_type']):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming T-Test vs Groups for ' + cell_type.title()
        )
        times = c_df[timepoint_col].unique()
        for timepoint, t_df in c_df.groupby(timepoint_col):
            emold = t_df[t_df.group == 'aging_phenotype'].code
            dmold = t_df[t_df.group == 'no_change'].code

            stat, p_value = stats.ttest_ind(
                emold,
                dmold
            )
            context: str = timepoint_col.title() + ' ' + str(int(timepoint))
            print_p_value(context, p_value)

        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming T-Test vs Time Points for ' + cell_type.title()
        )
        for (t1, t2) in combinations(times, 2):
            t1_df = c_df[c_df[timepoint_col] == t1].code
            t2_df = c_df[c_df[timepoint_col] == t2].code

            stat, p_value = stats.ttest_ind(
                t1_df,
                t2_df, 
            )
            context: str = timepoint_col.title() + ' ' + str(t1) \
                + ' vs ' + str(t2)
            print_p_value(context, p_value)


        plt.figure(figsize=(6,5))
        if line:
            ax = sns.lineplot(
                x=timepoint_col,
                y='code',
                hue='mouse_id',
                palette=COLOR_PALETTES['mouse_id'],
                data=c_df,
            )
            sns.scatterplot(
                x=timepoint_col,
                y='code',
                hue='mouse_id',
                palette=COLOR_PALETTES['mouse_id'],
                data=c_df,
                ax=ax
            )
            desc = 'line'
        else:
            ax = sns.swarmplot(
                x=timepoint_col,
                y='code',
                hue='group',
                hue_order=['aging_phenotype', 'no_change'],
                size=7,
                palette=COLOR_PALETTES['group'],
                dodge=True,
                data=c_df,
                zorder=0,
            )
            sns.boxplot(
                x=timepoint_col,
                y='code',
                hue='group',
                hue_order=['aging_phenotype', 'no_change'],
                palette=COLOR_PALETTES['group'],
                showbox=False,
                whiskerprops={
                    "alpha": 0
                },
                showcaps=False,
                ax=ax,
                data=c_df,
                fliersize=0,
            )
            desc = 'swarm'
        plt.title(
            cell_type.title() 
            + ' Clone Counts with Abundance Cutoff ' 
            + str(100 - abundance_cutoff)
        )
        plt.xlabel(timepoint_col.title())
        plt.ylabel('Number of Clones')
        plt.legend().remove()
        fname = save_path + os.sep \
            + desc + '_' \
            + 'clone_count_' + cell_type \
            + '_a' + str(abundance_cutoff).replace('.', '-') \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_swarm_violin_first_last_bias(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[float, str],
        abundance_cutoff: float,
        by_group: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ) -> None:
    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]


    with_time_desc = add_first_last_to_lineage_bias(
        lineage_bias_df,
        timepoint_col
    )    
    with_time_desc_filt = with_time_desc[with_time_desc.time_description.isin(['First', 'Last'])]
    filt_df = filter_lineage_bias_thresholds(
        with_time_desc_filt,
        thresholds
    )
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 2.5,
            'axes.linewidth': 4,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 28,
            'ytick.labelsize': 28,
            'figure.titlesize': 'medium',
        }

        )
    if by_group:
        for group, g_df in filt_df.groupby('group'):
            plt.figure(figsize=(7,5))
            ax = sns.violinplot(
                x='time_description',
                y='lineage_bias',
                order=['First', 'Last'],
                color='white',
                data=g_df,
                dodge=False,
                cut=0,
            )
            sns.swarmplot(
                x='time_description',
                y='lineage_bias',
                hue='mouse_id',
                order=['First', 'Last'],
                palette=COLOR_PALETTES['mouse_id'],
                data=g_df,
                size=7,
                ax=ax,
            )
            #plt.legend().remove()
            plt.ylabel('Lineage Bias')
            plt.xlabel('Time Point')
            plt.suptitle('Lineage Bias of Clones: ' + group.replace('_', ' ').title())
            plt.title(
                'Filtered by Clones With Cumulative Abundance '
                + ' > ' + str(100 - abundance_cutoff)
                + ' in Any Cell Type'
                )
            plt.legend().remove()
            file_name = save_path + os.sep \
                + 'first-last-lineage-bias' \
                + '_a' + str(abundance_cutoff).replace('.', '-') \
                + '_' + group \
                + '.' + save_format
            save_plot(file_name, save, save_format)
    else:
        plt.figure(figsize=(10,9))
        ax = sns.violinplot(
            x='time_description',
            y='lineage_bias',
            color='white',
            order=['First', 'Last'],
            data=filt_df,
            dodge=False,
            cut=0,
        )
        sns.swarmplot(
            x='time_description',
            y='lineage_bias',
            order=['First', 'Last'],
            hue='group',
            palette=COLOR_PALETTES['group'],
            data=filt_df,
            ax=ax,
        )
        plt.legend().remove()
        plt.ylabel('Lineage Bias')
        plt.xlabel('Time Point')
        plt.title(
            'Filtered by Clones With Cumulative Abundance '
            + ' > ' + str(100 - abundance_cutoff)
            + ' in Any Cell Type'
            )
        plt.suptitle('Lineage Bias of Clones First And Last Time Point')
        file_name = save_path + os.sep \
            + 'first-last-lineage-bias' \
            + '_a' + str(abundance_cutoff).replace('.', '-') \
            + '.' + save_format
        save_plot(file_name, save, save_format)

def plot_not_survived_abundance_at_time(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    survival_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Independent T-Test of Exhausted vs. Survived'
    )
    for cell_type in ['gr', 'b']:
        for group, g_df in survival_df.groupby('group'):
            print(
                Fore.CYAN + Style.BRIGHT 
                + '\n  - Group: ' + group.replace('_', ' ').title()
                + '  Cell Type: ' + cell_type.title()
            )
            y_col = cell_type + '_percent_engraftment'
            fig, ax = plt.subplots(figsize=(7,5))
            # T-Test on interesting result

            for time_change, t_df in g_df.groupby('time_change'):
                t_s = t_df[t_df['survived'] == 'Survived']
                t_e = t_df[t_df['survived'] == 'Exhausted']
                stat, p_value = stats.ttest_ind(
                    t_e[y_col],
                    t_s[y_col],
                )
                context: str = timepoint_col.title() + ' ' + str(int(time_change))
                print_p_value(context, p_value)
                

            ax = sns.boxplot(
                x='time_change',
                y=y_col,
                hue='survived',
                data=g_df,
                hue_order=['Exhausted', 'Survived']
            )
            ax.set(yscale='log')
            plt.xlabel(
                timepoint_col.title()
                + '(s) Survived'
            )
            plt.ylabel(y_col_to_title(y_col))
            plt.suptitle(
                cell_type.title() + ' Abundance Of Not Surviving Clones'
            )
            plt.title('Group: ' + group.replace('_', ' ').title())
            fname = save_path + os.sep \
                + 'abundance_not_survived' \
                + '_' + group \
                + '_' + cell_type \
                + '.' + save_format
            save_plot(fname, save, save_format)

def plot_exhausted_lymphoid_at_time(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: int,
        y_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ) -> None:
    
    filt_survival_df = filter_lymphoid_exhausted_at_time(
        lineage_bias_df,
        timepoint_col,
        timepoint,
    )
    filt_survival_all_df = lineage_bias_df.merge(
        filt_survival_df[['mouse_id','code','group']].drop_duplicates(),
        on=['mouse_id', 'code', 'group'],
        how='inner',
        validate='m:1'
    )
    fname_prefix = save_path + os.sep + 'exhausted_lymph_at_' \
        + timepoint_col[0] + str(timepoint) + '_'
    y_title = y_col_to_title(y_col)
    x_title = timepoint_col.title()

    if y_col != 'lineage_bias':
        filt_survival_all_df = bias_clones_to_abundance(
            filt_survival_all_df,
            clonal_abundance_df,
            y_col
        )

    plt.figure(figsize=(7, 5))
    sns.lineplot(
        x=timepoint_col,
        y=y_col,
        data=group_names_pretty(filt_survival_all_df),
        hue='group',
        palette=COLOR_PALETTES['group']
    )
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    title = 'Lymphoid Clones Exhausted at ' \
        + str(timepoint_col.title()) + ' ' + str(timepoint)
    plt.title(title)
    

    fname = fname_prefix + '_' + y_col + '_' \
        + '_average.' + save_format
    save_plot(fname, save, save_format)
    for gname, group_df in filt_survival_all_df.groupby('group'):
        plt.figure(figsize=(7, 5))
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
            + '_' + gname + '_' + 'by-clone_' \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_contribution_by_bias_cat(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        cell_type: str,
        by_sum: bool,
        by_group: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ) -> None:
    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

    cats = [
        'LC',
        'LB',
        'B',
        'MB',
        'MC'
    ]
    cats = [MAP_LINEAGE_BIAS_CATEGORY[x] for x in cats]
    colors = get_myeloid_to_lymphoid_colors(cats)
    palette = dict(zip(cats, colors))

    with_bias_cats = add_bias_category(
        lineage_bias_df,
    )    
    first_last_by_mouse_df = filter_first_last_by_mouse(
        with_bias_cats,
        timepoint_col,
    )
    y_col = cell_type + '_percent_engraftment'
    if by_sum:
        group_df = pd.DataFrame(
            first_last_by_mouse_df.groupby(
                ['mouse_id', 'group', 'mouse_time_desc', 'bias_category_short']
            )[y_col].sum()
        ).reset_index()
        math_desc = 'sum'
    else:
        group_df = pd.DataFrame(
            first_last_by_mouse_df.groupby(
                ['mouse_id', 'group', 'mouse_time_desc', 'bias_category_short']
            )[y_col].mean()
        ).reset_index()
        math_desc = 'mean'

    if by_group:
        for group, g_df in group_df.groupby('group'):
            plt.figure(figsize=(7,5))
            ax = sns.barplot(
                x='mouse_time_desc',
                y=y_col,
                order=['First', 'Last'],
                hue='bias_category_short',
                hue_order=cats,
                data=g_df,
                palette=palette,
                saturation=1,
            )
            ax.set(yscale='log')
            plt.legend().remove()
            plt.ylabel(y_col_to_title(y_col))
            plt.xlabel('Time Point')
            plt.suptitle('Group: ' + group.replace('_', ' ').title())
            plt.title(
                math_desc.title() + ' ' + y_col_to_title(y_col)
                )

            file_name = save_path + os.sep \
                + 'first-last_' \
                + math_desc + '_' \
                + y_col \
                + '_' + group \
                + '.' + save_format
            save_plot(file_name, save, save_format)
    else:
        plt.figure(figsize=(10,9))
        ax = sns.barplot(
            x='mouse_time_desc',
            y=y_col,
            order=['First', 'Last'],
            hue='bias_category_short',
            hue_order=cats,
            data=group_df,
            palette=palette,
            saturation=1,
        )
        ax.set(yscale='log')

        plt.ylabel(y_col_to_title(y_col))
        plt.xlabel('Time Point')
        plt.title(
            math_desc.title() + ' ' + y_col_to_title(y_col)
            )
        plt.legend(title='')
        file_name = save_path + os.sep \
            + 'first-last_' \
            + math_desc + '_' \
            + y_col \
            + '.' + save_format
        save_plot(file_name, save, save_format)
        
def plot_clone_count_bar_first_last(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        analyzed_cell_types: List[str],
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    threshold_df = filter_cell_type_threshold(
        clonal_abundance_df,
        thresholds, 
        analyzed_cell_types
    )
        
    clone_counts = count_clones(threshold_df, timepoint_col)
    clone_counts = filter_first_last_by_mouse(
        clone_counts,
        timepoint_col
    )
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 25,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 28,
            'ytick.labelsize': 22,
        }

        )

    for cell_type, c_df in clone_counts.groupby(['cell_type']):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test on ' 
            + ' ' + cell_type.title() + ' Clone Clone Counts Between Groups'
        )
        for t1, t_df in c_df.groupby('mouse_time_desc'):
            _, p_value = stats.ttest_ind(
                t_df[t_df.group == 'aging_phenotype']['code'],
                t_df[t_df.group == 'no_change']['code'],
            )
            context: str = timepoint_col.title() + ' ' + str(t1) 
            print_p_value(context, p_value, show_ns=True)

        plt.figure(figsize=(6,5))
        ax = sns.barplot(
            x='mouse_time_desc',
            order=['First', 'Last'],
            y='code',
            hue='group',
            hue_order=['aging_phenotype', 'no_change'],
            palette=COLOR_PALETTES['group'],
            saturation=1,
            data=c_df,
        )
        if abundance_cutoff == 50:
            plt.title(
                cell_type.title() 
                + ' Expanded Clones'
            )
        else:
            plt.title(
                cell_type.title() 
                + ' Clone Counts with Abundance Cutoff ' 
                + str(100 - abundance_cutoff)
            )
        plt.xlabel(timepoint_col.title())
        plt.ylabel('Number of Clones')
        plt.legend().remove()
        fname = save_path + os.sep \
            + 'clone_count_' + cell_type \
            + '_a' + str(abundance_cutoff).replace('.', '-') \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_clone_count_swarm_vs_cell_type(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        analyzed_cell_types: List[str],
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    if timepoint_col == 'gen':
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df[timepoint_col] != 8.5]
        clonal_abundance_df[timepoint_col] = clonal_abundance_df[timepoint_col].astype(int)
    threshold_df = filter_cell_type_threshold(
        clonal_abundance_df,
        thresholds, 
        analyzed_cell_types
    )
        
    clone_counts = count_clones(threshold_df, timepoint_col)

    plt.figure(figsize=(8,5))
    ax = sns.boxplot(
        x=timepoint_col,
        y='code',
        hue='cell_type',
        hue_order=analyzed_cell_types + ['Total'],
        palette=COLOR_PALETTES['cell_type'],
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        data=clone_counts,
    )
    sns.swarmplot(
        x=timepoint_col,
        y='code',
        hue='cell_type',
        size=7,
        hue_order=analyzed_cell_types + ['Total'],
        palette=COLOR_PALETTES['cell_type'],
        dodge=True,
        ax=ax,
        data=clone_counts,
    )
    if abundance_cutoff < 0.011:
        plt.title(
            'Present Clone Count' 
        )
    else:
        plt.title(
            'Clone Counts with Abundance Cutoff ' 
            + str(100 - abundance_cutoff)
        )
    plt.xlabel(timepoint_col.title())
    plt.ylabel('Number of Clones')
    plt.legend().remove()
    fname = save_path + os.sep \
        + 'clone_count_cell_vs' \
        + '_a' + str(abundance_cutoff).replace('.', '-') \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_perc_survival_bias_heatmap(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        by_clone: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )
    fname_prefix = save_path + os.sep
    if by_clone:
        y_col = 'exhausted_count'
        y_label = 'Number Per Mouse of Exhausted Clones'
        fname_prefix += 'count_survive_bias_heatmap_'
    else:
        y_col = 'exhausted_perc'
        y_label = 'Percent of Exhausted Clones Within Category'
        fname_prefix += 'perc_survive_bias_heatmap_'

    survival_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )


    survival_perc = calculate_survival_perc(
        survival_df,
        lineage_bias_df,
        timepoint_col,
    )
    cats = [
        'LC',
        'LB',
        'B',
        'MB',
        'MC'
    ]
    times = survival_perc['last_time'].unique()
    #cats = [MAP_LINEAGE_BIAS_CATEGORY[x] for x in cats]
    #survival_perc['bias_category'] = survival_perc.bias_category.apply(
        #lambda x: MAP_LINEAGE_BIAS_CATEGORY[x]
    #)
    for group, g_df in survival_perc.groupby('group'):
        pivot_df = g_df.pivot_table(
            columns='last_time',
            index='bias_category',
            values=y_col,
            aggfunc=np.mean,
            fill_value=0
        )
        plt.figure(figsize=(10,8))
        sns.heatmap(
            pivot_df[times].reindex(cats),
            cmap='cividis',
        )
        plt.title('Group: ' + y_col_to_title(group))
        plt.xlabel('End Point (' + timepoint_col.title() + ')')
        plt.ylabel('Lineage Bias Category')
        plt.suptitle(y_label)
        fname = fname_prefix + group + '.' + save_format
        save_plot(fname, save, save_format)

    group = 'all'
    pivot_df = survival_perc.pivot_table(
        columns='last_time',
        index='bias_category',
        values=y_col,
        aggfunc=np.mean,
        fill_value=0
    )
    plt.figure(figsize=(10,9))
    sns.heatmap(
        pivot_df[times].reindex(cats),
        cmap='cividis'
    )
    plt.title('Group: ' + y_col_to_title(group))
    plt.xlabel('End Point (' + timepoint_col.title() + ')')
    plt.ylabel('Lineage Bias Category')
    plt.suptitle(y_label)
    fname = fname_prefix + group + '.' + save_format
    save_plot(fname, save, save_format)

def plot_hsc_pie_mouse(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:

    clonal_abundance_df = filter_first_last_by_mouse(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.mouse_time_desc == 'Last']

    for (mouse, group), m_df in clonal_abundance_df.groupby(['mouse_id', 'group']):
        m_hsc_df = m_df[m_df.cell_type == 'hsc']
        m_gr_df = m_df[m_df.cell_type == 'gr'].rename(columns={'percent_engraftment': 'gr_percent_engraftment'})
        m_b_df = m_df[m_df.cell_type == 'b'].rename(columns={'percent_engraftment': 'b_percent_engraftment'})
        m_hsc_ct_df = m_hsc_df.merge(
            m_gr_df[['code', timepoint_col, 'gr_percent_engraftment']],
            how='left',
            on=['code', timepoint_col],
            validate='1:1',
        ).merge(
            m_b_df[['code', timepoint_col, 'b_percent_engraftment']],
            how='left',
            on=['code', timepoint_col],
            validate='1:1',
        )

        total_hsc_abundance = m_hsc_ct_df.percent_engraftment.sum()
        m_hsc_ct_df['relative_hsc_abundance'] = m_hsc_ct_df.percent_engraftment/total_hsc_abundance
        m_hsc_ct_df = m_hsc_ct_df.sort_values(by='relative_hsc_abundance').fillna(value=0)

        gr_norm = mpl.colors.Normalize(
            vmin=m_hsc_ct_df.gr_percent_engraftment.min(),
            vmax=m_hsc_ct_df.gr_percent_engraftment.max(),
        )
        gr_mapper = cm.ScalarMappable(norm=gr_norm, cmap=cm.viridis)
        gr_colors = [gr_mapper.to_rgba(x) for x in m_hsc_ct_df.gr_percent_engraftment]
        b_norm = mpl.colors.Normalize(
            vmin=m_hsc_ct_df.b_percent_engraftment.min(),
            vmax=m_hsc_ct_df.b_percent_engraftment.max(),
        )
        b_mapper = cm.ScalarMappable(norm=b_norm, cmap=cm.viridis)
        b_colors = [b_mapper.to_rgba(x) for x in m_hsc_ct_df.b_percent_engraftment]

        fig1, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Gr')
        ax1.pie(
            m_hsc_ct_df['relative_hsc_abundance'], 
            startangle=90,
            colors=gr_colors,
        )
        fig1.colorbar(
            gr_mapper,
            ax=ax1,
            orientation='horizontal',
            fraction=0.1,
        )

        ax2.set_title('B')
        ax2.pie(
            m_hsc_ct_df['relative_hsc_abundance'], 
            startangle=90,
            colors=b_colors,
        )
        fig1.colorbar(
            b_mapper,
            ax=ax2, 
            orientation='horizontal',
            fraction=0.1,
        )

        center_circle1 = plt.Circle(
            (0,0),
            0.7,
            fc='white',
        )
        center_circle2 = plt.Circle(
            (0,0),
            0.7,
            fc='white',
        )
        ax1.add_artist(center_circle1)
        ax1.axis('equal')
        ax2.add_artist(center_circle2)
        ax2.axis('equal')
        plt.suptitle(mouse)
        plt.tight_layout()
        fname = save_path + os.sep \
            + group + '_' + mouse \
            + '_hsc_abund_pie.' \
            + save_format
        save_plot(fname, save, save_format)

def plot_dist_bias_at_time_vs_group(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: timepoint_type,
        cell_type: str,
        bins: int,
        change_type: str = None,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    y_col = cell_type + '_percent_engraftment'
    if cell_type == 'sum':
        y_col = 'sum_abundance'
        lineage_bias_df[y_col] = lineage_bias_df['b_percent_engraftment'] + lineage_bias_df['gr_percent_engraftment']
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )
    bin_edge_count = bins + 1
    dodge_amount = 0.02
    bin_edges = np.linspace(-1, 1, bin_edge_count)
    center_points = (bin_edges[1:] + bin_edges[:-1]) / 2
    dodged_center_points = center_points + dodge_amount

    desc_addon = ''
    if change_type is not None:
        if change_type.lower() == 'changed':
            desc_addon = 'changed_clones_'
            bias_change_df = get_bias_change(
                lineage_bias_df,
                timepoint_col,
            )
            lineage_bias_df = mark_changed(
                lineage_bias_df,
                bias_change_df,
                min_time_difference=1,
            )
            lineage_bias_df = lineage_bias_df[lineage_bias_df['change_type'] != 'Unchanged']
        elif change_type.lower() == 'unchanged':
            desc_addon = 'unchanged_clones_'
            bias_change_df = get_bias_change(
                lineage_bias_df,
                timepoint_col,
            )
            lineage_bias_df = mark_changed(
                lineage_bias_df,
                bias_change_df,
                min_time_difference=1,
            )
            lineage_bias_df = lineage_bias_df[lineage_bias_df['change_type'] == 'Unchanged']

    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

    lineage_bias_at_time_df = get_clones_at_timepoint(
        lineage_bias_df,
        timepoint_col,
        timepoint
    )


    hist_df = pd.DataFrame()
    for (mouse_id, group), m_df in lineage_bias_at_time_df.groupby(['mouse_id', 'group']):
        m_hist, t_bins = np.histogram(m_df.lineage_bias, bins=bin_edges, weights=m_df[y_col])

        if not np.array_equal(t_bins, bin_edges):
            raise ValueError('Bins from numpy histogram, and input not same')

        # Slight x-axis shift to no_change group allowign for visibility of lines/error bars
        if group == 'no_change':
            m_row = pd.DataFrame.from_dict(
                {
                    'count': m_hist,
                    'lineage_bias': dodged_center_points,
                    'mouse_id': [mouse_id] * (bin_edge_count - 1),
                    'group': [group] * (bin_edge_count - 1),
                }
            )
        else:
            m_row = pd.DataFrame.from_dict(
                {
                    'count': m_hist,
                    'lineage_bias': center_points,
                    'mouse_id': [mouse_id] * (bin_edge_count - 1),
                    'group': [group] * (bin_edge_count - 1),
                }
            )
        hist_df = hist_df.append(m_row, ignore_index=True)

    plt.figure(figsize=(7,6))
    sns.lineplot(
        data=hist_df,
        x='lineage_bias',
        y='count',
        hue='group',
        err_style='bars',
        palette=COLOR_PALETTES['group']
    )
    plt.ylabel(cell_type.title() + ' Abundance (%WBC)')
    plt.xlabel('Lineage Bias')
    plt.suptitle(desc_addon.replace('_', ' ').title())
    plt.title('Lineage Bias at ' + str(timepoint).title() + ' ' + timepoint_col.title())
    plt.legend().remove()

    fname = save_path + os.sep \
        + 'lineage_bias_pointplot_vs_group_' \
        + desc_addon \
        + timepoint_col + '_' + str(timepoint) \
        + '_' + str(bins) + '-bins' \
        + '_' + cell_type \
        + '.' + save_format
    save_plot(fname, save, save_format)

def hsc_to_ct_compare(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        invert: bool,
        by_mouse: bool,
        cell_type: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 6,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )
    clonal_abundance_df = find_last_clones_in_mouse(
        clonal_abundance_df,
        timepoint_col,
    )
    cell_type_wide_df = abundance_to_long_by_cell_type(clonal_abundance_df, timepoint_col)
    x_col = 'hsc_percent_engraftment'
    y_col = cell_type+'_percent_engraftment'
    desc_addon = ''

    # Invert means normalize to hsc abundance
    if invert:
        cell_type_wide_df[y_col] = cell_type_wide_df[y_col]/cell_type_wide_df[x_col] 
        desc_addon='_invert'

    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    # Scatter plot each mouse data
    for [mouse_id, group], m_df in cell_type_wide_df.groupby(['mouse_id', 'group']):
        if by_mouse:
            no_na_df = m_df[['mouse_id', 'code', x_col, y_col]].dropna(axis='index')
            no_na_zero_df = no_na_df[
                (no_na_df[x_col] > 0) & \
                (no_na_df[y_col] > 0)
            ]
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x=np.log(no_na_zero_df[x_col]),
                    y=np.log(no_na_zero_df[y_col]),
                )
                r_squared = r_value**2
                print(Fore.CYAN + Style.BRIGHT + mouse_id + ' Log(' + cell_type + ') vs Log(hsc):')
                print(Fore.CYAN + " r-squared:"  + str(r_squared))
                print_p_value('', p_value)
            except ValueError:
                continue
            fig, ax = plt.subplots(figsize=(7,5))

        plt.loglog(
            m_df[x_col],
            m_df[y_col],
            'o',
            markeredgecolor='white',
            markeredgewidth=.5,
            color=COLOR_PALETTES['mouse_id'][mouse_id]
        )
        if by_mouse:
            y_min, y_max = plt.ylim()
            x_min, x_max = plt.xlim()
            reg_y = [np.exp(np.log(x) * slope + intercept) for x in no_na_zero_df['hsc_percent_engraftment']]
            plt.plot(no_na_zero_df[x_col], reg_y, color=COLOR_PALETTES['mouse_id'][mouse_id])
            # Plot addons: titles, vlines, hlines
            plt.vlines(thresholds['hsc'], y_min, y_max)
            plt.hlines(thresholds[cell_type], x_min, x_max)
            plt.ylabel(cell_type.title() + ' Cell Abundance (%WBC)')
            plt.xlabel('HSC Abundance')

            plt.title(
                cell_type.title() + ' vs HSC: '
            )
            plt.suptitle(
                'P-Value: ' + str(p_value)
                + ' R-Squared: ' + str(round(r_squared, 4))
            )
            fname = save_path + os.sep \
                + group + '_' \
                + mouse_id + '_' \
                + 'hsc_abundance_relation_' \
                + cell_type + '_' \
                + 'a' + str(abundance_cutoff).replace('.','-') \
                + desc_addon \
                + '.' + save_format
            save_plot(fname, save, save_format)

    if not by_mouse:
        y_min, y_max = plt.ylim()
        x_min, x_max = plt.xlim()

        # Linear (log/log) regression
        no_na_df = cell_type_wide_df[['mouse_id', 'code', x_col, y_col]].dropna(axis='index')
        no_na_zero_df = no_na_df[
            (no_na_df[x_col] > 0) & \
            (no_na_df[y_col] > 0)
        ]
        ## Regression using scipy
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=np.log(no_na_zero_df['hsc_percent_engraftment']),
            y=np.log(no_na_zero_df[cell_type+'_percent_engraftment']),
        )
        r_squared = r_value**2
        print(Fore.CYAN + Style.BRIGHT + 'Log(' + cell_type + ') vs Log(hsc):')
        print(Fore.CYAN + " r-squared:"  + str(r_squared))
        print_p_value('', p_value)
        reg_y = [np.exp(np.log(x) * slope + intercept) for x in no_na_zero_df['hsc_percent_engraftment']]

        plt.plot(no_na_zero_df[x_col], reg_y, color='#e74c3c')

        # Plot addons: titles, vlines, hlines
        plt.vlines(thresholds['hsc'], y_min, y_max)
        plt.hlines(thresholds[cell_type], x_min, x_max)
        plt.ylabel(cell_type.title() + ' Cell Abundance (%WBC)')
        plt.xlabel('HSC Abundance')

        plt.title(
            cell_type.title() + ' vs HSC: '
        )
        plt.suptitle(
            'P-Value: ' + str(p_value)
            + ' R-Squared: ' + str(round(r_squared, 4))
        )
        fname = save_path + os.sep \
            + 'hsc_abundance_relation_' \
            + cell_type + '_' \
            + 'a' + str(abundance_cutoff).replace('.','-') \
            + desc_addon \
            + '.' + save_format
        save_plot(fname, save, save_format)

def hsc_blood_prod_over_time(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 5,
            'axes.linewidth': 3,
            'axes.labelsize': 10,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'figure.titlesize': 'small',
        }
    )
    hsc_df = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].rename(columns={'percent_engraftment': 'hsc_percent_engraftment'})
    not_hsc_df = clonal_abundance_df[clonal_abundance_df.cell_type != 'hsc']
    with_hsc_data_df = not_hsc_df.merge(
        hsc_df[['code','mouse_id', 'hsc_percent_engraftment']],
        on=['code', 'mouse_id'],
        how='inner',
        validate='m:1'
    )
    with_hsc_data_df = with_hsc_data_df[['mouse_id', 'group', 'code', timepoint_col, 'cell_type', 'hsc_percent_engraftment', 'percent_engraftment']].dropna(axis='index')
    g = sns.FacetGrid(
        with_hsc_data_df,
        col=timepoint_col,
        row='cell_type',
        hue='mouse_id',
        palette=COLOR_PALETTES['mouse_id'],
    )

    kws={
        "markeredgecolor":'white',
        "marker": 'o',
        'lw': 0,
    }
    g.map(
        plt.loglog,
        'hsc_percent_engraftment',
        'percent_engraftment',
        **kws
    )
    fname = save_path + os.sep \
        + 'hsc_blood_prod_over_time' \
        + '.' + save_format
    save_plot(fname, save, save_format)

def exhaust_persist_hsc_abund(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        save: bool,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 3,
            'lines.markeredgecolor': 'white',
            'axes.linewidth': 3,
            'axes.labelsize': 25,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'figure.titlesize': 'small',
        }
    )
    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].rename(columns={'percent_engraftment':'HSC Abundance'})
    without_hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type.isin(['gr','b'])]
    exhaustion_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col,
    )
    exhaust_with_hsc_df = exhaustion_df.merge(
        hsc_data[['code', 'mouse_id', 'HSC Abundance']],
        on=['code', 'mouse_id'],
        how='inner',
        validate='m:1'
    )

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Rank Sum Test of HSC Abundance in Exhausted vs. Survived'
    )

    t_s = exhaust_with_hsc_df[exhaust_with_hsc_df['survived'] == 'Survived']
    t_e = exhaust_with_hsc_df[exhaust_with_hsc_df['survived'] == 'Exhausted']
    stat, p_value = stats.ranksums(
        t_e['HSC Abundance'],
        t_s['HSC Abundance'],
    )
    context: str = ''
    print_p_value(context, p_value)

    plt.figure(figsize=(7,5))
    plt.yscale('log')
    ax = sns.violinplot(
        x='survived',
        y='HSC Abundance',
        order=['Exhausted', 'Survived'],
        data=exhaust_with_hsc_df,
        color='white',
        cut=0,
    )
    sns.swarmplot(
        x='survived',
        y='HSC Abundance',
        order=['Exhausted', 'Survived'],
        data=exhaust_with_hsc_df,
        hue='mouse_id',
        palette=COLOR_PALETTES['mouse_id'],
        ax=ax,
    )
    plt.legend().remove()
    fname = save_path + os.sep + 'exhaust_persist_hsc_abund.' + save_format
    save_plot(fname, save, save_format)

def hsc_to_ct_compare_outlier(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        invert: bool,
        by_mouse: bool,
        cell_type: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 6,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )
    clonal_abundance_df = find_last_clones_in_mouse(
        clonal_abundance_df,
        timepoint_col,
    )
    cell_type_wide_df = abundance_to_long_by_cell_type(clonal_abundance_df, timepoint_col)
    x_col = 'hsc_percent_engraftment'
    y_col = cell_type+'_percent_engraftment'
    desc_addon = ''

    # Invert means normalize to hsc abundance
    if invert:
        cell_type_wide_df[y_col] = cell_type_wide_df[y_col]/cell_type_wide_df[x_col] 
        desc_addon='_invert'
    
    clean_data_df = cell_type_wide_df[[
        'code',
        'mouse_id',
        'group',
        timepoint_col,
        x_col,
        y_col,
    ]].dropna(axis='index')

    # Linear (log/log) regression
    formula = 'np.log1p('+y_col+') ~ np.log1p('+x_col+')' 
    model = ols(formula, data=clean_data_df)
    res = model.fit()
    params = res.params
    reg_y = res.predict(clean_data_df[x_col], transform=True)
    outlier_res = res.outlier_test()
    print(res.summary())
    print(outlier_res)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xscale('linear')
    ax.set_yscale('linear')

    # Scatter plot each mouse data
    outlier_marked_df = mark_outliers(clean_data_df, outlier_res)
    outliers = outlier_marked_df[outlier_marked_df.outlier]
    not_outliers = outlier_marked_df[~outlier_marked_df.outlier]


    plt.loglog(
        not_outliers[x_col],
        not_outliers[y_col],
        'bo',
        markeredgecolor='white',
        markeredgewidth=.5,
    )
    plt.loglog(
        outliers[x_col],
        outliers[y_col],
        'ro',
        markeredgecolor='white',
        markeredgewidth=.5,
    )

    y_min, y_max = plt.ylim()
    x_min, x_max = plt.xlim()
    reg_y = [(np.expm1(np.log1p(x) * params[1]) + params[0]) for x in clean_data_df[x_col]]


    plt.loglog(clean_data_df[x_col], reg_y, 'o', color='#e74c3c')

    # Plot addons: titles, vlines, hlines
    plt.vlines(thresholds['hsc'], y_min, y_max)
    plt.hlines(thresholds[cell_type], x_min, x_max)
    plt.ylabel(cell_type.title() + ' Cell Abundance (%WBC)')
    plt.xlabel('HSC Abundance')

    plt.title(
        formula
    )
    fname = save_path + os.sep \
        + 'hsc_abundance_relation_outliers' \
        + cell_type + '_' \
        + 'a' + str(abundance_cutoff).replace('.','-') \
        + desc_addon \
        + '.' + save_format
    save_plot(fname, save, save_format)


def hsc_to_ct_compare_svm(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        invert: bool,
        by_mouse: bool,
        cell_type: str,
        n_clusters: int = 2,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 6,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )
    clonal_abundance_df = find_last_clones_in_mouse(
        clonal_abundance_df,
        timepoint_col,
    )
    cell_type_wide_df = abundance_to_long_by_cell_type(clonal_abundance_df, timepoint_col)
    x_col = 'hsc_percent_engraftment'
    y_col = cell_type+'_percent_engraftment'
    desc_addon = ''

    # Invert means normalize to hsc abundance
    if invert:
        cell_type_wide_df[y_col] = cell_type_wide_df[y_col]/cell_type_wide_df[x_col] 
        desc_addon='_invert'
    
    clean_data_df = cell_type_wide_df[[
        'code',
        'mouse_id',
        'group',
        timepoint_col,
        x_col,
        y_col,
    ]].dropna(axis='index')

    # KMeans Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
        np.log1p(clean_data_df[[x_col, y_col]])
    )
    clean_data_df['label'] = kmeans.labels_

    # Support Vector Machine Boundary Finder
    clf = LinearSVC(random_state=0)
    clf.fit(
        np.log1p(clean_data_df[[x_col, y_col]]),
        clean_data_df['label']
    )
    print(clf.coef_, clf.intercept_)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xscale('linear')
    ax.set_yscale('linear')


    for _, l_df in clean_data_df.groupby('label'):
        plt.loglog(
            l_df[x_col],
            l_df[y_col],
            'o',
            markeredgecolor='white',
            markeredgewidth=.5,
        )


    y_min, y_max = plt.ylim()
    x_min, x_max = plt.xlim()

    if n_clusters == 2:
        x_vals = np.geomspace(x_min/10, x_max/50, 100000)
        reg_y = [(np.expm1(np.log1p(x) * clf.coef_[0][0]) + clf.intercept_[0]) for x in x_vals]

        plt.loglog(
            x_vals,
            reg_y,
            'x',
            color='#e74c3c',
            markersize=2
        )

    # Plot addons: titles, vlines, hlines
    plt.vlines(thresholds['hsc'], y_min, y_max)
    plt.hlines(thresholds[cell_type], x_min, x_max)
    plt.ylabel(cell_type.title() + ' Cell Abundance (%WBC)')
    plt.xlabel('HSC Abundance')
    plt.title(cell_type.title())
    fname = save_path + os.sep \
        + 'hsc_abundance_relation_svm' \
        + cell_type + '_' \
        + 'a' + str(abundance_cutoff).replace('.','-') \
        + '_' + str(n_clusters) + 'clusters' \
        + desc_addon \
        + '.' + save_format
    save_plot(fname, save, save_format)


def heatmap_correlation_hsc_ct(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_mouse: bool,
        group: str,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    if group != 'all':
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.group == group]

    hsc_df = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].rename(columns={'percent_engraftment': 'hsc_percent_engraftment'})
    not_hsc_df = clonal_abundance_df[clonal_abundance_df.cell_type != 'hsc']
    with_hsc_data_df = not_hsc_df.merge(
        hsc_df[['code','mouse_id', 'hsc_percent_engraftment']],
        on=['code', 'mouse_id'],
        how='inner',
        validate='m:1'
    )
    with_hsc_data_df = with_hsc_data_df[['mouse_id', 'group', 'code', timepoint_col, 'cell_type', 'hsc_percent_engraftment', 'percent_engraftment']].dropna(axis='index')
    pearson_df = pd.DataFrame()
    for (cell_type, mouse_id, g, timepoint), t_df in with_hsc_data_df.groupby(['cell_type', 'mouse_id', 'group', timepoint_col]):
        pearson_r, p_value = stats.pearsonr(
            t_df['percent_engraftment'],
            t_df['hsc_percent_engraftment']
        )
        p_data = {
            'mouse_id': [mouse_id],
            'cell_type': [cell_type],
            timepoint_col: [timepoint],
            'pearson_r': [pearson_r],
            'p_value': [p_value],
            'group': [g]
        }
        pearson_df = pearson_df.append(pd.DataFrame.from_dict(p_data))

    desc_addon = 'avg'
    if by_mouse:
        desc_addon = 'by-mouse'
        index_col = 'mouse_id'
    elif by_group:
        desc_addon = 'by-group'
        index_col = 'group'
    if by_mouse or by_group:
        fig, axes = plt.subplots(figsize=(7, 7), nrows=2, ncols=1)
        i = 0
        for cell_type, data in pearson_df.groupby('cell_type'):
            if data[timepoint_col].nunique() <= 2:
                continue
            pivotted = data.pivot_table(
                values='pearson_r',
                index=index_col,
                columns=timepoint_col,
                aggfunc=np.mean,
            )
            sns.heatmap(
                pivotted,
                ax=axes[i]
            )
            axes[i].set_ylabel(cell_type.title())
            i += 1

        
    else:
        pivotted = pearson_df.pivot_table(
            values='pearson_r',
            index='cell_type',
            columns=timepoint_col,
            aggfunc=np.mean,
        )
        sns.heatmap(
            pivotted,
        )
    
    fname = save_path + os.sep \
        + 'hsc_blood_over_time_heatmap_' \
        + group + '_' \
        + desc_addon \
        + '.' + save_format

    save_plot(fname, save, save_format)

def exhausted_clone_hsc_abund(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        group: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    if group != 'all':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

    survival_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )
    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']\
        .rename(columns={'percent_engraftment': 'hsc_percent_engraftment'})
    total_hsc_per_mouse = pd.DataFrame(
        hsc_data.groupby(['mouse_id']).hsc_percent_engraftment.sum()
    ).reset_index().rename(
        columns={'hsc_percent_engraftment': 'hsc_total_abundance'}
    )
    hsc_data = hsc_data.merge(
        total_hsc_per_mouse,
        validate='m:1'
    )
    hsc_data = hsc_data.assign(
        perc_tracked_hsc=lambda x: 100 * x.hsc_percent_engraftment/x.hsc_total_abundance
    )


    
    survival_with_hsc_df = survival_df.merge(
        hsc_data[['mouse_id', 'code', 'perc_tracked_hsc']],
        on=['mouse_id', 'code'],
        how='inner',
        validate='m:1'
    )
    survival_sum_per_mouse = pd.DataFrame(
        survival_with_hsc_df.groupby(['mouse_id', timepoint_col, 'group', 'survived'])['perc_tracked_hsc'].sum()
    ).reset_index()

    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 1.5,
            'axes.linewidth': 4,
            'axes.labelsize': 25,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 28,
            'ytick.labelsize': 22,
            'figure.titlesize': 'small',
        }

        )
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Rank Sum Test of Exhausted vs. Survived'
    )
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\n  - Group: ' + group.replace('_', ' ').title()
    )
    fig, ax = plt.subplots(figsize=(9,6))
    # T-Test on interesting result

    for time_change, t_df in survival_with_hsc_df.groupby('time_change'):
        t_s = t_df[t_df['survived'] == 'Survived']
        t_e = t_df[t_df['survived'] == 'Exhausted']
        stat, p_value = stats.ranksums(
            t_e.perc_tracked_hsc,
            t_s.perc_tracked_hsc,
        )
        context: str = timepoint_col.title() + ' ' + str(int(time_change))
        print_p_value(context, p_value)
        

    ax = sns.boxplot(
        x=timepoint_col,
        y='perc_tracked_hsc',
        hue='survived',
        data=survival_sum_per_mouse,
        hue_order=['Exhausted', 'Survived'],
        showbox=True,
        whiskerprops={
            "alpha": 1
        },
        color='white',
        showcaps=False,
        fliersize=0,
    )
    sns.swarmplot(
        x=timepoint_col,
        y='perc_tracked_hsc',
        hue='survived',
        hue_order=['Exhausted', 'Survived'],
        data=survival_sum_per_mouse,
        dodge=True,
        size=8,
    )
    plt.xlabel(
        'End Point (' + timepoint_col.title() + ')'
    )
    plt.legend().remove()
    plt.ylabel('% Clonal Abundance In Final HSC Pool')
    plt.suptitle(
        'Exhausted Clone Abundance'
    )
    plt.title('Group: ' + group.replace('_', ' ').title())
    fname = save_path + os.sep \
        + 'exhausted_clone_hsc_abund' \
        + '_' + group \
        + '.' + save_format
    save_plot(fname, save, save_format)


