
""" Functions used to help plot data in plot_data.py

"""

from typing import List, Dict, Any
from itertools import combinations
import os
import json
import pandas as pd
import numpy as np
import progressbar
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
import matplotlib as mpl
from colorama import init, Fore, Style
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as co
import seaborn as sns
from colour import Color
from pyvenn import venn
import aggregate_functions as agg
import statistical_tests as stat_tests
from data_types import timepoint_type
import pingouin as pg

init(autoreset=True)
COLOR_PALETTES = json.load(open('lib/color_palettes.json', 'r'))
MARKERS = json.load(open('lib/markers.json', 'r'))
LINE_STYLES = json.load(open('lib/line_styles.json', 'r'))
NEW_GROUP_NAME_MAP = {
    'aging_phenotype': 'E-MOLD',
    'no_change': 'D-MOLD'
}
TENX_MICE = [
    'M2059',
    'M2012',
    'M2061',
    'M190',
]


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

def plot_palette(
        pal: str,
        save: bool = False,
        save_path: str = '',
        save_format: str = 'png'
    ) -> None:
    palette = COLOR_PALETTES[pal]    
    sns.palplot(
        sns.color_palette(palette.values())
    )
    ax = plt.gca()
    xticks = ax.get_xticks()
    xticks = [x + 0.5 for x in xticks]
    plt.xticks(xticks, palette.keys(), rotation=45, fontsize=20)
    fname = save_path + os.sep \
        + pal + '.' + save_format
    save_plot(fname, save, save_format)


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
    threshold_df = agg.filter_cell_type_threshold(
        input_df,
        thresholds, 
        analysed_cell_types)
    clone_counts = agg.agg.count_clones(threshold_df, timepoint_col)

    plot_clone_count(
        clone_counts,
        thresholds,
        analysed_cell_types,
        abundance_cutoff=abundance_cutoff,
        group=group,
        save=save,
        line=line,
        timepoint_col=timepoint_col,
        save_path=save_path
    )

def plot_clone_enriched_at_time(filtered_df: pd.DataFrame,
                                enrichement_months: List[Any],
                                enrichment_thresholds: Dict[str, float],
                                timepoint_col: str,
                                analyzed_cell_types: List[str],
                                by_clone: bool = False,
                                save: bool = False,
                                save_path: str = './output',
                                save_format: str = 'png',
                                ) -> None:
    """ Create a Line + Swarm plot of clones dominant at specified time

    Arguments:
        filtered_df {pd.DataFrame} -- Step7 output put through agg.filter_threshold()
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
        enriched_df = agg.combine_enriched_clones_at_time(
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
            enriched_df = agg.get_data_from_mice_missing_at_time(enriched_df, exclusion_timepoint=14, timepoint_column='month')
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
        filtered_df {pd.DataFrame} -- long format output of step7 passed through agg.filter_threshold()
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
        present_clones_df {pd.DataFrame} -- Clones filtered for presence by agg.filter_threshold()
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
    present_clones_df = agg.filter_mice_with_n_timepoints(present_clones_df, n_timepoints=4)
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
        if 'gr' in y_col:
            y_col = 'myeloid_percent_engraftment'
        if 'b' in y_col:
            y_col = 'lymphoid_percent_engraftment'
        lineage_bias_df = agg.bias_clones_to_abundance(
            lineage_bias_df,
            clonal_abundance_df,
            y_col
        )
        log_scale = True

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 1,
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
        lineage_bias_df = agg.bias_clones_to_abundance(
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
        thresholds = agg.find_top_percentile_threshold(input_df, percentile, analyzed_cell_types)
    clone_counts = agg.count_clones_at_percentile(input_df, percentile, analyzed_cell_types=analyzed_cell_types, thresholds=thresholds)

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


    max_df = agg.get_max_by_mouse_timepoint(input_df)
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
    max_df = agg.get_max_by_mouse_timepoint(input_df)
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
    max_df = agg.get_max_by_mouse_timepoint(input_df)

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
        clonal_abundance_df: pd.DataFrame,
        cell_type: str,
        timepoint_col: str,
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

    sns.set_context(
        'paper',
        font_scale=3,
        rc={
            'lines.linewidth': 5,
            'axes.linewidth': 4,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'small',
        }

        )


    plt.figure(figsize=(8,8))
    first_time_df = agg.find_first_clones_in_mouse(
        clonal_abundance_df,
        timepoint_col
    )
    first_time_df[timepoint_col] = 'first'
    contributions = agg.percentile_sum_engraftment(first_time_df, timepoint_col, cell_type=cell_type)
    exp_x, exp_y = agg.find_intersect(
        data=contributions,
        y=50.0,
    )
    plot = sns.lineplot(
        x='percentile',
        y='percent_sum_abundance',
        data=contributions,
        color=COLOR_PALETTES['time_desc']['first']
    )
    plt.xlabel('Percentile by Abundance')
    plt.ylabel('% ' + cell_type.title() + ' Abundance')
    plt.title('Cumulative Abundance at Percentiles for ' + cell_type)


    plt.vlines(exp_x, -5, exp_y + 5, linestyles='dashed')
    plt.hlines(exp_y, 0, exp_x + 5, linestyles='dashed')
    plt.text(0, 52, 'Expanded: (' + str(round(exp_x, 2)) + ', ' + str(round(exp_y, 2)) + ')')
    sns.despine()


    fname = save_path + os.sep + 'percentile_abundance_contribution_' + cell_type + '.' + save_format
    save_plot(fname, save, save_format)

def plot_change_contributions(
        changed_marked_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: Any,
        gfp_donor: pd.DataFrame,
        gfp_donor_thresh: float,
        force_order: bool,
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


    if timepoint_col == 'gen':
        changed_marked_df = changed_marked_df[changed_marked_df.gen != 8.5]
    percent_of_total = True
    changed_sum_df = agg.sum_abundance_by_change(
        agg.remove_month_17(
            changed_marked_df,
            timepoint_col
        ),
        timepoint_col=timepoint_col,
        percent_of_total=percent_of_total
    )
    changed_sum_df = changed_sum_df[changed_sum_df.cell_type.isin(['gr', 'b'])]
    changed_sum_with_gxd = changed_sum_df.merge(
        gfp_donor,
        how='inner',
        validate='m:1',
        on=['mouse_id', timepoint_col]
    )

    timepoint_df = agg.get_clones_at_timepoint(
        changed_sum_with_gxd,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )

    print('FILTERING FOR MICE FOUND IN FIRST AND LAST TIMEPOINT ABOVE GFP x DONOR THRESHOLD' )
    mice_left = changed_marked_df.mouse_id.unique()
    for t in ['first', 'last']:
        t_df = agg.get_clones_at_timepoint(
            changed_sum_with_gxd,
            timepoint_col,
            t,
            by_mouse=True,
        )
        t_filt_gxd = t_df[t_df.gfp_x_donor >= gfp_donor_thresh]
        mice_left = [m for m in mice_left if m in t_filt_gxd.mouse_id.unique()]
    

    filt_gxd = timepoint_df[timepoint_df.mouse_id.isin(mice_left)]
    print('Pre-filt gxd mice:', timepoint_df.mouse_id.nunique())
    print('Post-filt gxd mice:', filt_gxd.mouse_id.nunique())

    filt_gxd = filt_gxd.assign(total=100)
    filt_gxd = filt_gxd.sort_values(by='percent_engraftment', ascending=False)
    print(filt_gxd.groupby('mouse_id')[timepoint_col].unique())

    for group, g_df in filt_gxd.groupby('group'):
        avg_per_group = pd.DataFrame(
            g_df.groupby(
            ['change_type', 'cell_type']
            ).percent_engraftment.sum()/g_df.mouse_id.nunique()
        ).reset_index()
        avg_per_group['total'] = 100
        avg_per_group['mouse_id'] = 'Average'
        group_df = g_df.append(avg_per_group, sort=False)
        for (m, cell_type), m_df in group_df.groupby(['mouse_id', 'cell_type']):
            lymph = m_df[m_df.change_type == 'Lymphoid']
            if lymph.empty:
                temp_row = pd.DataFrame()
                temp_row['mouse_id'] = [m]
                temp_row['cell_type'] = [cell_type]
                temp_row['change_type'] = 'Lymphoid'
                temp_row['total'] = 100
                temp_row['percent_engraftment'] = 0
                lymph = temp_row
                group_df = group_df.append(temp_row, sort=False)
            elif len(lymph) > 1:
                print('Lymph > 1')
                print(m, cell_type, len(lymph))

            myl = m_df[m_df.change_type == 'Myeloid']
            if myl.empty:
                temp_row = pd.DataFrame()
                temp_row['mouse_id'] = [m]
                temp_row['cell_type'] = [cell_type]
                temp_row['change_type'] = 'Myeloid'
                temp_row['percent_engraftment'] = 0
                temp_row['total'] = 100
                myl = temp_row
                group_df = group_df.append(temp_row, sort=False)
            elif len(myl) > 1:
                print('Myeloid > 1')
                print(m, cell_type, len(myl))

            unk = m_df[m_df.change_type == 'Unknown']
            if unk.empty:
                temp_row = pd.DataFrame()
                temp_row['mouse_id'] = [m]
                temp_row['cell_type'] = [cell_type]
                temp_row['change_type'] = 'Unknown'
                temp_row['percent_engraftment'] = 0
                temp_row['total'] = 100
                unk = temp_row
                group_df = group_df.append(temp_row, sort=False)
            elif len(unk) > 1:
                print('Unknown > 1')
                print(m, cell_type, len(unk))

            #pd.options.mode.chained_assignment = None
            myeloid_mouse_bool = (
                (group_df.mouse_id == m) &
                (group_df.change_type == 'Myeloid') &
                (group_df.cell_type == cell_type),
            )
            group_df.loc[
                (group_df.mouse_id == m) &
                (group_df.change_type == 'Myeloid') &
                (group_df.cell_type == cell_type),
                'percent_engraftment'
            ] = myl.percent_engraftment.values[0] + lymph.percent_engraftment.values[0]
            group_df.loc[
                (group_df.mouse_id == m) &\
                (group_df.change_type == 'Unknown') &\
                (group_df.cell_type == cell_type),
                'percent_engraftment'
            ] = myl.percent_engraftment.values[0] + lymph.percent_engraftment.values[0] + unk.percent_engraftment.values[0]
            pd.options.mode.chained_assignment = 'warn'
        if force_order:
            if timepoint_col != 'month':
                raise ValueError('Cannot force order if not using aging study data')
            print(Fore.YELLOW + 'FORCING ORDER BASED ON HARD CODED VALUES')
            if group == 'aging_phenotype':
                force_order = ['M3003', 'M3007', 'M3010', 'M3013', 'M3015', 'M3011', 'M2012', 'M2059', 'M3012', 'M3025', 'M3016']
            elif group == 'no_change':
                force_order = ['M3022', 'M3008', 'M3019', 'M3023', 'M2061', 'M3009', 'M190', 'M3028', 'M3018', 'M3001', 'M3000', 'M3017']
            else:
                raise ValueError('Group not identified, order not forcable')
                
            order = [m for m in force_order if m in group_df.mouse_id.unique()]
        else: 
            order = list(
                pd.DataFrame(
                    group_df[
                        (group_df.change_type != 'Unchanged') &
                        (group_df.mouse_id != 'Average')
                    ].groupby(
                        ['mouse_id']
                    ).percent_engraftment.sum()
                ).reset_index().sort_values(
                by='percent_engraftment',
                ascending=False,
                ).mouse_id.unique()
            )
            print(group, ':', order)

        order.append('Average')
        plt.figure(figsize=(7,10))
        print(y_col_to_title(group) + ' Mice: ' + str(group_df.mouse_id.nunique()))
        ax = sns.barplot(
            x='total',
            y='mouse_id',
            order=order,
            hue='cell_type',
            hue_order=['gr', 'b'],
            data=group_df,
            palette=[COLOR_PALETTES['change_type']['Unchanged']]*2,
            saturation=1,
            #label="Unchanged"
        )
        sns.barplot(
            x='percent_engraftment',
            y='mouse_id',
            hue='cell_type',
            order=order,
            hue_order=['gr', 'b'],
            data=group_df[group_df.change_type == 'Unknown'],
            palette=[COLOR_PALETTES['change_type']['Unknown']]*2,
            saturation=1,
            ax=ax,
        )
        sns.barplot(
            x='percent_engraftment',
            y='mouse_id',
            hue='cell_type',
            order=order,
            hue_order=['gr', 'b'],
            data=group_df[group_df.change_type == 'Myeloid'],
            palette=[COLOR_PALETTES['change_type']['Myeloid']]*2,
            saturation=1,
            #label='Myeloid',
            ax=ax,
        )
        sns.barplot(
            x='percent_engraftment',
            y='mouse_id',
            hue='cell_type',
            order=order,
            hue_order=['gr', 'b'],
            data=group_df[group_df.change_type == 'Lymphoid'],
            palette=[COLOR_PALETTES['change_type']['Lymphoid']]*2,
            saturation=1,
            #label='Lymphoid',
            ax=ax,
        )
        plt.xlabel('Contribution of Changed Cells')
        plt.ylabel('')
        plt.suptitle(' Cumulative Abundance of Clones by Bias Change')
        plt.title('Group: ' + group.replace('_', ' ').title() + ' ' + timepoint_col.title() + ' ' + str(timepoint))
        plt.gca().legend(title='Change Direction', loc='lower right').remove()
        sns.despine(left=True, bottom=True)

        fname = save_path + os.sep + 'percent_contribution_changed_' \
            + '_' + group \
            + '_gxd-' + str(gfp_donor_thresh) \
            + '_' + timepoint_col[0] + '-' + str(timepoint) \
            + '.' + save_format
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
    changed_sum_df = agg.sum_abundance_by_change(changed_marked_df, percent_of_total=percent_of_total)
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


    _, thresholds = agg.calculate_thresholds_sum_abundance(
        input_df,
        timepoint_col=timepoint_col,
        abundance_cutoff=abundance_cutoff
    )

    if group != 'all':
        input_df = input_df.loc[input_df.group == group]

    filter_df = agg.filter_cell_type_threshold(input_df, thresholds, analyzed_cell_types)
    clone_counts = agg.count_clones(filter_df)
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
        group: str = 'all',
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
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )

    time_col = timepoint_col

    filtered_df = agg.filter_cell_type_threshold(
        input_df,
        thresholds,
        analyzed_cell_types=[cell_type]
    )

    filtered_df = agg.remove_month_17_and_6(
        filtered_df,
        timepoint_col
    )

    nunique_time = filtered_df[timepoint_col].nunique()
    filtered_df =agg.filter_mice_with_n_timepoints(
        filtered_df,
        nunique_time
    )

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    if by_group:
        for group, g_df in filtered_df.groupby('group'):
            stat_tests.friedman_wilcoxonSignedRank(
                data=g_df,
                timepoint_col=timepoint_col,
                id_col='code',
                value_col='percent_engraftment',
                overall_context=' '.join([group, cell_type]),
                show_ns=True,
                match_cols=['mouse_id', 'code'],
                merge_type='inner',
                fill_na=0,
                aggfunc=np.mean,
            )
        stat_tests.ranksums_test_group_time(
            data=filtered_df,
            test_col='percent_engraftment',
            timepoint_col=timepoint_col,
            overall_context=cell_type,
            show_ns=False,
        )
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.despine()
        plt.yscale('log')
        if cell_type == 'gr':
            ax.set_ylim([60e-6,5])
        elif cell_type == 'b':
            ax.set_ylim([30e-5,5])
        ax = sns.swarmplot(
            x=time_col,
            y='percent_engraftment',
            hue='group',
            hue_order=['aging_phenotype', 'no_change'],
            data=filtered_df,
            palette=COLOR_PALETTES['group'],
            dodge=True,
            zorder=0,
            linewidth=0.5,
            )
        sns.boxplot(
            x=time_col,
            y='percent_engraftment',
            hue='group',
            hue_order=['aging_phenotype', 'no_change'],
            data=filtered_df,
            ax=ax,
            fliersize=0,
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
        )

        
        title = cell_type.capitalize() \
            + ' > ' \
            + str(round(thresholds[cell_type],2)) + '% WBC'
        if abundance_cutoff != 0:
            plt.suptitle(title)
        plt.xlabel(time_col.title())
        plt.ylabel('Clone Abundance (% WBC)')
        plt.legend().remove()
        fname = save_path + os.sep + 'swamplot_abundance' \
                + '_' + cell_type + '_a' \
                + str(round(abundance_cutoff, 2)).replace('.','-') \
                + '_by-group' \
                + '.' + save_format
        save_plot(fname, save, save_format)
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.yscale('log')
        sns.despine()
        pal = COLOR_PALETTES['mouse_id']
        if cell_type == 'gr':
            ax.set_ylim([30e-6,5])
        elif cell_type == 'b':
            ax.set_ylim([30e-5,5])
        stat_tests.friedman_wilcoxonSignedRank(
            data=filtered_df,
            timepoint_col=timepoint_col,
            id_col='code',
            value_col='percent_engraftment',
            overall_context=cell_type,
            show_ns=True,
            match_cols=['mouse_id', 'code'],
            merge_type='inner',
            fill_na=0,
            aggfunc=np.mean,
        )
        sns.swarmplot(
            x=time_col,
            y='percent_engraftment',
            hue='mouse_id',
            data=filtered_df,
            palette=pal,
            ax=ax,
            zorder=0
            )
        sns.boxplot(
            x=time_col,
            y='percent_engraftment',
            data=filtered_df,
            boxprops={'facecolor': 'None'},
            ax=ax,
            fliersize=0,
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
        )

        title = cell_type.capitalize() \
            + ' > ' \
            + str(round(thresholds[cell_type],2)) + '% WBC'
        plt.suptitle(title)
        plt.xlabel(time_col.title())
        plt.ylabel('Clone Abundance (% WBC)')
        plt.legend().remove()
        fname = save_path + os.sep + 'swamplot_abundance' \
                + '_' + group \
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
    filtered_df = agg.filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds
        )
    between_gen_bias_change_df = agg.between_gen_bias_change(
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

    filtered_df = agg.filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds
        )
    across_gen_bias_change_df = agg.across_gen_bias_change(
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
            lineage_bias_df = lineage_bias_df.assign(gen=lambda x: agg.day_to_gen(x.day))
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen <= 8]

        if group != 'all':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

        bias_change_df = agg.calculate_bias_change(
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
        abundance_df = abundance_df.assign(gen=lambda x: agg.day_to_gen(x.day))
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
        abundance_change_df = agg.calculate_abundance_change(
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
            filt_abund = agg.filter_cell_type_threshold(
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

    sns.set_context(
        'paper',
        font_scale=2,
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 6,
            'axes.linewidth': 3,
            'axes.labelsize': 20,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )


    filt_lineage_bias_df = agg.filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds=thresholds
    )
    bias_change_df = agg.calculate_first_last_bias_change(
        filt_lineage_bias_df,
        timepoint_col,
        by_mouse=False,
        )
        
    timepoint_text = ''
    if timepoint is not None:
        bias_change_df = agg.filter_bias_change_timepoint(
            bias_change_df,
            timepoint
        )
        timepoint_text = ' - Clones Must have First or Last Time at: ' +str(timepoint)
    bias_change_df = bias_change_df[bias_change_df.time_change >= min_time_difference]
    if group != 'all':
        bias_change_df = bias_change_df.loc[bias_change_df.group == group]

    fig, ax = plt.subplots(figsize=(6,5))
    x, y, y1, y2, x_c, y_c, kde = agg.calculate_bias_change_cutoff(
        bias_change_df,
        min_time_difference=min_time_difference,
        timepoint=timepoint,
    )
    fill_ind = ((x>=0) * (x<=2))
    yfill = y[fill_ind].copy()
    bot = np.zeros(yfill.shape)
    print(x[fill_ind], yfill, bot)
    ax.fill_between(x[fill_ind], yfill, bot, color='silver', alpha=0.3)
    ax.plot(x, y, c='silver', alpha=0.3)
    ax.plot(x, y1, c=COLOR_PALETTES['change_status']['Unchanged'])
    ax.plot(x, y2, c=COLOR_PALETTES['change_status']['Changed'])
    ax.scatter(x_c, y_c, c='k')
    ax.axvline(x_c[0], c='k')
    sns.despine()
    plt.xticks([0, x_c[0], 2])

    if group != 'all':
        plt.title(' Group: ' + y_col_to_title(group) + timepoint_text)
    plt.xlabel('Magnitude of Lineage Bias Change')
    plt.ylabel('Probability Density')

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
        lineage_bias_df = lineage_bias_df.assign(gen=lambda x: agg.day_to_gen(x.day))
    else:
        timepoint_col = 'month'

    if cached_change is not None:
        bias_change_df = cached_change

    else:
        # If not, calculate and save cached
        bias_change_df = agg.calculate_bias_change(
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
    
    lineage_bias_df = agg.filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds
    )
    lineage_bias_df['code'] = 'Tracked'
    lineage_bias_df = lineage_bias_df.append(rest_of_clones_bias_df, ignore_index=True)
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
    biased_at_time_df = agg.filter_biased_clones_at_timepoint(
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
        biased_at_time_df = agg.bias_clones_to_abundance(
            biased_at_time_df,
            clonal_abundance_df,
            y_col
        )
        

    y_title = y_col_to_title(y_col)
    if invert_selection:
        plot_title = 'Bias within +/- ' \
            + str(round(bias_cutoff, 2)) + ' at ' \
            + timepoint_col.title() + ' ' + str(timepoint)
        if round(bias_cutoff, 2) in [0.71, 0.5]:
            plot_title = 'Balanced' \
                + ' At ' \
                + timepoint_col.title() + ' ' + str(timepoint)
            
            # Uncomment to remove clones with only 2 time points for balanced clone graph

            #print('Filtering with abundance > 0.01 at a minimum of 3 time points')
            #biased_at_time_df['uid'] = biased_at_time_df['mouse_id'].str.cat(biased_at_time_df['code'], sep='')
            #by_clone_group = biased_at_time_df.groupby('uid').aggregate(np.count_nonzero)
            #more_than_2_timepoints = by_clone_group[by_clone_group[timepoint_col] > 2].index
            #biased_at_time_df = biased_at_time_df[
                #biased_at_time_df['uid'].isin(more_than_2_timepoints)
            #]
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

    stable_clone_df = agg.filter_stable_initially(
        lineage_bias_df,
        t1=t1,
        timepoint_col=timepoint_col,
        bias_change_cutoff=bias_change_cutoff,
    )
    fname_prefix = save_path + os.sep + 'stable_clones'
    y_title = y_col_to_title(y_col)
    x_title = timepoint_col.title()

    if y_col != 'lineage_bias':
        stable_clone_df = agg.bias_clones_to_abundance(
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
    
    bias_dist_df = agg.calculate_first_last_bias_change_with_avg_data(
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

def plot_abund_change_bias_dist_group_vs(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        cutoff: float,
        mtd: int,
        timepoint: int,
        y_col: str,
        by_mouse: bool,
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
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=by_mouse
    )
    _, _, _, _, cutoffs, _ = agg.calculate_bias_change_cutoff(
        bias_change_df,
        mtd,
        timepoint=timepoint
    )
    bias_change_cutoff = cutoffs[0]
    
    plt.figure(figsize=(7,6))

    for gname, g_df in bias_change_df.groupby('group'):
        c = COLOR_PALETTES['group'][gname]
        sns.distplot(
            g_df.bias_change,
            rug=False,
            kde=False,
            hist=True,
            hist_kws={
                'histtype': 'step',
                'weights': g_df[y_col],
                'linewidth': 3,
                'alpha': .8,
                'color': c
            },
            color=c,
            label=NEW_GROUP_NAME_MAP[gname],

        )

        plt.title(
            y_col_to_title(y_col)
        )
        plt.xlabel('Change In Lineage Bias')
        plt.ylabel(y_col_to_title(y_col))
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
    plt.vlines(min_max_change, ymin, ymax, linestyles='dashed')
    plt.hlines(0, -2, 2, colors='gray', linestyles='dashed')
    fname = save_path + os.sep +'bias_change_dist_vs_group_' \
        + y_col \
        + '_' + str(cutoff).replace('.','-') \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_bias_dist_mean_abund_group_vs(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        timepoint: int,
        change_status: str,
        by_mouse: bool,
        y_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 22,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }

        )
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse
    )
    _, _, _, _, cutoffs, _, _ = agg.calculate_bias_change_cutoff(
        bias_change_df,
        mtd,
        timepoint=timepoint
    )
    bias_change_cutoff = cutoffs[0]
    bias_change_df = bias_change_df.assign(
        myeloid_change=lambda x: x.myeloid_percent_abundance_last - x.myeloid_percent_abundance_first,
        lymphoid_change=lambda x: x.lymphoid_percent_abundance_last - x.lymphoid_percent_abundance_first,
    )
    
    _, ax = plt.subplots(figsize=(6,5))
    if not (y_col in ['lymphoid_percent_abundance', 'myeloid_percent_abundance']):
        y_col = 'count'
        ax.set_yscale('symlog', linthreshy=50)
        ax.set_yticks([0, 10, 25, 50, 100])

    for gname, g_df in bias_change_df.groupby('group'):
        c = COLOR_PALETTES['group'][gname]
        hist_kws = {
            'histtype': 'step',
            'linewidth': 3,
            'alpha': .8,
            'color': c,
        }
        if y_col != 'count':
            print('WEIGHTING BY Change IN', y_col)
            hist_kws['weights'] = g_df[y_col.split('_')[0] + '_change']
        sns.distplot(
            g_df.bias_change,
            bins=20,
            rug=False,
            kde=False,
            hist=True,
            hist_kws=hist_kws,
            color=c,
            ax=ax,
            label=NEW_GROUP_NAME_MAP[gname],

        )

        plt.xlabel('Change In Lineage Bias')
        plt.ylabel(y_col.title())
        plt.xlim((-2,2.02))
        ## Shrink current axis's height by 10% on the bottom
        #ax = plt.gca()
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        #box.width, box.height * 0.9])

        ## Put a legend below current axis
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            #ncol=2)

    min_max_change = [bias_change_cutoff, -1 * bias_change_cutoff]
    sns.despine()
    plt.axvline(min_max_change[0], linestyle='dashed', c='k')
    plt.axvline(min_max_change[1], linestyle='dashed', c='k')
    fname = save_path + os.sep +'bias_change_dist_vs_group_' \
        + y_col \
        + '_change-type-'+str(change_status) \
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
    bias_dist_df = agg.calculate_first_last_bias_change_with_avg_data(
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
        t_df = agg.find_last_clones_in_mouse(lineage_bias_df, timepoint_col)
    else:
        t_df = lineage_bias_df[lineage_bias_df[timepoint_col].isin([timepoint])]
    filt_df = agg.filter_lineage_bias_anytime(
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

    with_avg_abund = agg.add_average_abundance_to_lineage_bias(
        lineage_bias_df,
    )

    with_time_desc = agg.add_first_last_to_lineage_bias(
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
    transplant_survival_df = agg.abundant_clone_survival(
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
    not_survived_df = agg.not_survived_bias_by_time_change(
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
    not_survived_df = agg.not_survived_bias_by_time_change(
        lineage_bias_df,
        timepoint_col
    )
    width = 0.35
    not_survived_df['bias_category'] = not_survived_df.bias_category.apply(
        lambda x: agg.MAP_LINEAGE_BIAS_CATEGORY[x]
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
    cats = [agg.MAP_LINEAGE_BIAS_CATEGORY[x] for x in cats]
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
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_average: bool,
        by_mouse: bool,
        group: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    
    if group != 'all':
        print('Filtering for group:', group)
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.group == group]

    print('...Removing gen 8.5 if it exists')
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        keep_hsc=False
    )
    print('...Removing month 17 and 6 if they exists')
    abund_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    if by_average:
        abund_df = agg.add_avg_abundance_until_timepoint_clonal_abundance_df(
            clonal_abundance_df,
            timepoint_col
        )
        y_col = 'avg_abundance'
        avg_desc='avg_until'
    else:
        y_col = 'percent_engraftment'
        avg_desc='avg_at'

    print('...Labeling clones with exhaustion data')
    survival_df = agg.label_exhausted_clones(
        None,
        abund_df,
        timepoint_col,
    )


    print('...Plotting')
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 1.5,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
        }

        )
    last_time = survival_df[timepoint_col].max()
    survival_df = survival_df[survival_df[timepoint_col] != last_time]
    if by_mouse:
        survival_df = pd.DataFrame(
            survival_df.groupby([
                'mouse_id',
                'cell_type',
                'group',
                'survived',
                timepoint_col
            #])[y_col].sum()
            ])[y_col].mean()
        ).reset_index()
        #avg_desc += '_sum_by_mouse'
        avg_desc += '_mean_by_mouse'

    for cell_type, c_df in survival_df.groupby('cell_type'):
        _, ax = plt.subplots(figsize=(8, 6))
        threshy = 10E-4
        if cell_type == 'b':
            threshy = 10E-3
        ax.set_yscale('symlog', linthreshy=threshy)
        if by_mouse:
            zero_filleds = []
            for t, t_df in c_df.groupby(timepoint_col):
                zero_filled = agg.fill_mouse_id_zeroes(
                    t_df,
                    ['group', timepoint_col],
                    y_col,
                    'survived',
                    ['Exhausted', 'Survived'],
                    fill_val=0,
                )
                zero_filleds.append(zero_filled)
            zero_filled_df = pd.concat(zero_filleds)
            #stripplot_mouse_markers_with_mean(
                #zero_filled_df,
                #timepoint_col,
                #y_col,
                #ax,
                #'survived',
                #['Exhausted', 'Survived']
            #)
            if group:
                color = COLOR_PALETTES['group'][group]
            else:
                color = 'gray'

            hatch_dict = {
                'Survived': '//',
                'Exhausted': None
            } 
            barplot_hatch_diff(
                data=zero_filled_df,
                x=timepoint_col,
                y=y_col,
                ax=ax,
                hatch_col='survived',
                hatch_order=['Exhausted', 'Survived'],
                color=color,
                hatch_dict=hatch_dict
            )
            #stripplot_label_markers_with_mean(
                #zero_filled_df,
                #timepoint_col,
                #y_col,
                #ax,
                #'survived',
                #['Exhausted', 'Survived'],
                #color=color
            #)
            stat_tests.ind_ttest_between_groups_at_each_time(
                data=zero_filled_df,
                test_col=y_col,
                timepoint_col=timepoint_col,
                overall_context=cell_type.title() + ' by_mouse',
                group_col='survived',
                show_ns=True,
            )
            for s, s_df in zero_filled_df.groupby('survived'):
                stat_tests.one_way_ANOVArm(
                    data=s_df,
                    timepoint_col=timepoint_col,
                    id_col='mouse_id',
                    value_col=y_col,
                    overall_context=s.title() + ' ' + cell_type.title() + ' by_mouse',
                    show_ns=True,
                    match_cols=['mouse_id', 'group'],
                    merge_type='inner',
                    fill_na=None,
                )
        else:
            ax.set_yscale('symlog', linthreshy=10e-3)
            sns.boxplot(
                x=timepoint_col,
                y=y_col,
                hue='survived',
                data=c_df,
                hue_order=['Exhausted', 'Survived'],
                saturation=1,
                palette=COLOR_PALETTES['survived'],
                ax=ax,
            )
            stat_tests.ranksums_test_group_time(
                data=c_df,
                test_col=y_col,
                timepoint_col=timepoint_col,
                overall_context=cell_type.title(),
                show_ns=True,
                group_col='survived'
            )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            fancybox=True, shadow=True, ncol=2)

        plt.title(cell_type.title())
        plt.ylabel(y_col)
        plt.xlabel(
            timepoint_col.title()
        )
        plt.legend().remove()
        sns.despine()
        fname = save_path + os.sep \
            + 'abundance_not_survived' \
            + '_' + cell_type \
            + '_' + avg_desc \
            + '_' + group \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_not_survived_count_box(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 15,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }
    )
    if timepoint_col == 'gen':
        print('Filtering for just mice with all generations of data')
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.has_all_time == 1]
        print(clonal_abundance_df.groupby('group').mouse_id.unique())
        print('\n')
    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col,
    )
    filt_abundance_df = clonal_abundance_df[
        clonal_abundance_df.percent_engraftment > 0
    ]
    last_clones = agg.get_clones_at_timepoint(
        filt_abundance_df,
        timepoint_col,
        timepoint='last',
        by_mouse=False,
    )
    mouse_last = pd.DataFrame(
        clonal_abundance_df.groupby('mouse_id')[timepoint_col].max()
    ).reset_index()

    # Filter for only those clones who's lat time point 
    #   is NOT the last time point of the mouse
    mouse_last['mouse_last'] = True
    last_clones = last_clones.merge(
        mouse_last,
        how='left',
        validate='m:1'
    )
    last_clones_without_mouse_last = last_clones[last_clones.mouse_last != True]

    count_df = pd.DataFrame(last_clones_without_mouse_last.groupby(
        ['mouse_id', 'group', timepoint_col]).code.nunique()
        ).reset_index()
    count_df = agg.fill_mouse_id_zeroes(
        count_df,
        info_cols=['group'],
        fill_col='code',
        fill_cat_col=timepoint_col,
        fill_cats=count_df[timepoint_col].unique(),
        fill_val=0,
    )
    plt.figure(figsize=(8,6))
    file_name_addon =''
    if by_group:
        ax = sns.boxplot(
            x=timepoint_col,
            y='code',
            hue='group',
            order=count_df.sort_values(by=timepoint_col)[timepoint_col].unique(),
            hue_order=['aging_phenotype', 'no_change'],
            data=count_df,
            palette=COLOR_PALETTES['group']
        )
        file_name_addon ='_by_group'
        stat_tests.ind_ttest_group_time(
            data=count_df,
            test_col='code',
            timepoint_col=timepoint_col,
            overall_context='Exhausted Clone Count',
            show_ns=True,
        )
        for group, g_df in count_df.groupby('group'):
            stat_tests.one_way_ANOVArm(
                data=g_df,
                match_cols=['mouse_id'],
                id_col='mouse_id',
                merge_type='outer',
                fill_na=0,
                value_col='code',
                timepoint_col=timepoint_col,
                overall_context=group.title() + ' Exhausted Clone Count',
                show_ns=True,
            )
    else:
        ax = sns.boxplot(
            x=timepoint_col,
            y='code',
            order=count_df[timepoint_col].unique(),
            data=count_df,
            palette=COLOR_PALETTES[timepoint_col]
        )
        stat_tests.one_way_ANOVArm(
            data=count_df,
            match_cols=['mouse_id'],
            id_col='mouse_id',
            merge_type='outer',
            fill_na=0,
            value_col='code',
            timepoint_col=timepoint_col,
            overall_context='Exhausted Clone Count',
            show_ns=False,
        )
    _, max_codes = plt.ylim()
    ax.set_ylim(0, max_codes)
    ax.set_ylabel('# Of Clones With Last Time Point')

    ax.set_xlabel('End Point (' + timepoint_col.title() + ')')
    plt.title(
        'Count of Exhausted Clones Over Time'
    )
    ax.legend().remove()
    sns.despine()
    fname = save_path + os.sep \
        + 'not_survived_count' \
        + file_name_addon \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_hsc_abund_bias_at_last(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_group: bool,
        by_count: bool,
        group: str,
        mtd: int,
        change_type: str,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:
    if group != 'all':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.group == group]
    lineage_bias_df = agg.remove_month_17(lineage_bias_df, timepoint_col)
    clonal_abundance_df = agg.remove_month_17(clonal_abundance_df, timepoint_col)
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 2,
            'axes.linewidth': 4,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.major.size': 8,
            'ytick.minor.width': 0,
            'ytick.minor.size': 0,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }

        )
    if clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].empty:
        raise ValueError('No HSC Cells in Clonal Abundance Data')

    file_add = ''
    if change_type:
        bias_change_df = agg.calculate_first_last_bias_change(
            lineage_bias_df,
            timepoint_col,
            by_mouse=False,
        )
        change_label = agg.mark_changed(
            lineage_bias_df,
            bias_change_df,
            mtd,
            
        )
        last_clones = agg.find_last_clones(
            change_label,
            timepoint_col
        )
        labeled_last_clones = agg.add_bias_category(
            last_clones
        )
        hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']
        
        myeloid_hsc_abundance_df = hsc_data.merge(
            labeled_last_clones[['code','mouse_id','bias_category', 'change_type']],
            on=['code','mouse_id'],
            how='inner',
            validate='m:m'
        )
        myeloid_hsc_abundance_df = myeloid_hsc_abundance_df[
            myeloid_hsc_abundance_df.change_type == change_type
        ]
        file_add += '_' + change_type + '_mtd' + str(mtd) 
    else:
        last_clones = agg.get_clones_at_timepoint(
            lineage_bias_df,
            timepoint_col,
            timepoint='last',
            by_mouse=True,
        )
        labeled_last_clones = agg.add_bias_category(
            last_clones
        )
        hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']
        
        myeloid_hsc_abundance_df = hsc_data.merge(
            labeled_last_clones[['code','mouse_id','bias_category']],
            on=['code','mouse_id'],
            how='inner',
            validate='m:m'
        )
    y_desc = 'abundance'
    y_col = 'percent_engraftment'
    unique_cats=['LB', 'B', 'MB']
    data_df = myeloid_hsc_abundance_df
    if by_count:
        y_desc = 'Clone Count'
        y_col = 'code'
        no_zero_df = myeloid_hsc_abundance_df[
            myeloid_hsc_abundance_df.percent_engraftment > 0
        ]
        count_df = pd.DataFrame(
            no_zero_df.groupby(['mouse_id', 'group', 'bias_category'])['code'].nunique(),
        ).reset_index()
        count_df = agg.fill_mouse_id_zeroes(
            count_df,
            info_cols=['group'],
            fill_col='code',
            fill_cat_col='bias_category',
            fill_cats=unique_cats,
            fill_val=0,
        )
        data_df = count_df

    print(data_df.mouse_id.unique())
    if by_group:
        _, ax = plt.subplots(figsize=(6,5))
        #stripplot_mouse_markers_with_mean(
            #data_df,
            #'bias_category',
            #y_col,
            #ax,
            #order=unique_cats,
        #)
        
        sns.barplot(
            y=y_col,
            x='bias_category',
            hue='group',
            hue_order=['Aging Phenotype', 'No Change'],
            order=unique_cats,
            data=group_names_pretty(data_df),
            palette=COLOR_PALETTES['group'],
            saturation=1,
            capsize=.2,
            ci=68,
            ax=ax,
            zorder=1,
        )
        sns.barplot(
            y=y_col,
            x='bias_category',
            hue='group',
            hue_order=['Aging Phenotype', 'No Change'],
            order=unique_cats,
            data=group_names_pretty(data_df),
            palette=COLOR_PALETTES['group'],
            saturation=1,
            ci=None,
            ax=ax,
            zorder=10
        )
        if not by_count:
            stat_tests.ranksums_test_group_time(
                data=data_df,
                test_col='percent_engraftment',
                timepoint_col='bias_category',
                overall_context='HSC Abundance',
                show_ns=True,
            )
            for g, g_df in data_df.groupby('group'):
                stat_tests.ranksums_test_group(
                    data=g_df,
                    test_col='percent_engraftment',
                    group_col='bias_category',
                    overall_context='HSC Abundance ' + g,
                    show_ns=True,
                )
            ax.set(yscale='log')
            ax.set_yticks([10E-1, 10E-2, 10E-3,])
        else:
            stat_tests.ind_ttest_between_groups_at_each_time(
                data=data_df,
                test_col=y_col,
                timepoint_col='bias_category',
                overall_context='HSC ' + y_desc,
                show_ns=True
            )
            for g, g_df in data_df.groupby('group'):
                stat_tests.one_way_ANOVArm(
                    data=g_df,
                    timepoint_col='bias_category',
                    id_col='mouse_id',
                    value_col=y_col,
                    overall_context=g + ' HSC ' + y_desc,
                    show_ns=True,
                    match_cols=['mouse_id', 'group'],
                    merge_type='inner',
                    fill_na=None
                )
        for k, spine in ax.spines.items():  #ax.spines is a dictionary
            spine.set_zorder(100)
        ax.legend().remove()
        ax.set_xlabel('')
        ax.set_ylabel('HSC ' + y_desc.title())
        ax.set_title(
            'HSC ' + y_desc.title() + ' by Bias at Last Time Point'
            )
        sns.despine()

        fname = save_path + os.sep \
            + y_desc \
            + '_hsc_biased_at_last_by-group' \
            + file_add \
            + '.' + save_format
        save_plot(fname, save, save_format)
    else:
        sems=[[],[]]
        means=[]
        colors=[]
        print(data_df)
        for bias_cat in unique_cats:
            cats_df = data_df[data_df.bias_category == bias_cat]
            sems[0].append(0)
            sems[1].append(cats_df[y_col].sem())
            means.append(cats_df[y_col].mean())
            colors.append(COLOR_PALETTES['bias_category'][bias_cat])
        
        coords = np.arange(len(unique_cats)) + 1
        width = 0.8
        _, ax = plt.subplots(figsize=(6, 5))
        ax.bar(
            x=coords,
            height=means,
            width=width,
            tick_label=unique_cats,
            color=colors,
        )
        _, caps, _ = ax.errorbar(
            coords,
            means,
            yerr=sems,
            color='black',
            capsize=10,
            capthick=3,
            ls='none',
            )
        caps[0].set_marker('_')
        caps[0].set_markersize(0)

        ax.legend().remove()
        ax.set_xlabel('')
        ax.set_ylabel('HSC ' + y_desc.title())
        ax.set_title(
            'HSC ' + y_desc.title() + ' by Bias at Last Time Point'
            )

        if not by_count:
            stat_tests.ranksums_test_group(
                data_df,
                y_col,
                'HSC Abundance',
                show_ns=True,
                group_col='bias_category'
            )
            ax.set(yscale='log')
            ax.set_yticks([1, 10E-1, 10E-2, 10E-3,])
            ax.set_ylim([10e-3, 1.5])
        else:
            #LOG TO LINEAR
            stat_tests.one_way_ANOVArm(
                data=data_df,
                timepoint_col='bias_category',
                id_col='mouse_id',
                value_col=y_col,
                overall_context=' HSC ' + y_desc,
                show_ns=True,
                match_cols=['mouse_id', 'group'],
                merge_type='inner',
                fill_na=None
            )

        sns.despine()
        fname = save_path + os.sep \
            + y_desc \
            + '_hsc_biased_at_last' \
            + file_add \
            + '_' + str(group) \
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
    bias_change_df = agg.get_bias_change(
        lineage_bias_df,
        timepoint_col
    )
    plot_clones_df = lineage_bias_df
    if y_col != 'lineage_bias':
        plot_clones_df = agg.bias_clones_to_abundance(
            lineage_bias_df,
            clonal_abundance_df,
            y_col
        )
    changed_marked_df = agg.mark_changed(
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

    stable_clone_df = agg.filter_stable_initially(
        lineage_bias_df,
        t1=t1,
        timepoint_col=timepoint_col,
        bias_change_cutoff=bias_change_cutoff,
    )
    abundant_at_timepoint = agg.find_enriched_clones_at_time(
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
        stable_clone_df = agg.bias_clones_to_abundance(
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
        clonal_abundance_df: pd.DataFrame,
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
    fname_prefix = save_path + os.sep
    if by_clone:
        y_col = 'exhausted_count'
        y_label = 'Number Per Mouse of Exhausted Clones'
        fname_prefix += 'count_survive_bias_'
    else:
        y_col = 'exhausted_perc'
        y_label = 'Percent of Exhausted Clones Within Category'
        fname_prefix += 'perc_survive_bias_'


    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    lineage_bias_df = agg.remove_month_17_and_6(
        lineage_bias_df,
        timepoint_col
    )
    with_bias_cats_df = agg.add_bias_category(lineage_bias_df)
    survival_df = agg.label_exhausted_clones(
        with_bias_cats_df,
        clonal_abundance_df,
        timepoint_col
    )


    survival_perc = agg.calculate_survival_perc(
        survival_df,
        timepoint_col,
    )

    cats = [
        'LB',
        'B',
        'MB'
    ]

    palette = COLOR_PALETTES['bias_category']
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
            errwidth=2,
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
        errwidth=2,
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
    bias_change_df = agg.get_bias_change(
        lineage_bias_df,
        timepoint_col
    )
    change_marked_df = agg.mark_changed(
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
    t_df = agg.filter_first_last_by_mouse(
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
        merge_type: str,
        timepoint: Any = None,
        sum: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    change_col = 'change_status'
    hue_order = ['Unchanged', 'Changed']

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
    bias_change_df = agg.get_bias_change(
        lineage_bias_df,
        timepoint_col
    )
    change_marked_df = agg.mark_changed(
        clonal_abundance_df,
        bias_change_df,
        min_time_difference=mtd,
        timepoint=timepoint,
        merge_type=merge_type,
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
                change_col,
                timepoint_col,
            ]).percent_engraftment.sum()
        ).reset_index()
        desc_add = 'sum'
    
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Ranksums Test Between Change Status '+ desc_add.title() + ' Abundance at each time point'
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
            hue=change_col,
            hue_order=hue_order,
            palette=COLOR_PALETTES[change_col],
        )
        min_val = c_df[c_df.percent_engraftment > 0].percent_engraftment.min()
        ax.set_yscale('symlog', linthreshy=min_val*10)
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
            for (a, b) in combinations(c_df[change_col].unique(),2):
                a_vals = t_df[t_df[change_col] == a]
                b_vals = t_df[t_df[change_col] == b]
                stat, p_value = stats.ranksums(
                    a_vals['percent_engraftment'],
                    b_vals['percent_engraftment'],
                )
                context: str = str(timepoint) + ' - ' + a + ' vs ' + b
                stat_tests.print_p_value(context, p_value)

    group = 'all'
    for cell_type, c_df in change_marked_df.groupby(['cell_type']):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\n  - Group: ' + group.replace('_', ' ').title()
            + '  Cell Type: ' + cell_type.title()
        )
        for timepoint, t_df in c_df.groupby(timepoint_col):
            for (a, b) in combinations(c_df[change_col].unique(),2):
                a_vals = t_df[t_df[change_col] == a]
                b_vals = t_df[t_df[change_col] == b]
                stat, p_value = stats.ttest_ind(
                    a_vals['percent_engraftment'],
                    b_vals['percent_engraftment'],
                )
                context: str = str(timepoint) + ' - ' + a + ' vs ' + b
                stat_tests.print_p_value(context, p_value)
        plt.figure(figsize=(7, 5))
        plt.title(
            cell_type.title() + ' '
            + desc_add.title()
            + ' Abundance by Lineage Bias Change'
        )
        ax = sns.boxplot(
            x=timepoint_col,
            y='percent_engraftment',
            data=c_df,
            hue=change_col,
            hue_order=hue_order,
            palette=COLOR_PALETTES[change_col],
        )
        min_val = c_df[c_df.percent_engraftment > 0].percent_engraftment.min()
        ax.set_yscale('symlog', linthreshy=min_val*10)
        plt.xlabel(timepoint_col.title())
        plt.ylabel(y_col_to_title(cell_type+'_percent_engraftment'))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            fancybox=True, shadow=True, ncol=3, title='').remove()
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
    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    all_timepoints =agg.filter_mice_with_n_timepoints(
        clonal_abundance_df,
        clonal_abundance_df[timepoint_col].nunique()
    )
    
    most_abund_contrib = pd.DataFrame(
        all_timepoints.groupby(
            ['mouse_id', 'group', timepoint_col, 'cell_type']
        )['percent_engraftment'].apply(lambda x: x.nlargest(n).sum())
    ).reset_index()
    times = most_abund_contrib[timepoint_col].unique()

    for cell_type, c_df in most_abund_contrib.groupby(['cell_type']):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test on ' 
            + ' ' + cell_type.title() + ' ' + str(n)
            + ' Most Abundant Clone Between Groups'
        )
        for t1, t_df in c_df.groupby(timepoint_col):
            _, p_value = stats.ttest_ind(
                t_df[t_df.group == 'aging_phenotype']['percent_engraftment'],
                t_df[t_df.group == 'no_change']['percent_engraftment'],
            )
            context: str = timepoint_col.title() + ' ' + str(t1) 
            stat_tests.print_p_value(context, p_value, show_ns=True)
        stat_tests.rel_ttest_group_time(
            data=c_df,
            match_cols=['mouse_id'],
            merge_type='inner',
            fill_na=None,
            test_col='percent_engraftment',
            timepoint_col=timepoint_col,
            overall_context=cell_type.title() + ' ' + str(n) + ' Most Abundant Clone',
            show_ns=False,
        )

        medianprops = dict(
            linewidth=0,
        )
        meanprops = dict(
            linestyle='solid',
            linewidth=3,
            color='black'
        )
        plt.figure(figsize=(7,5))
        ax = sns.boxplot(
            x=timepoint_col,
            y='percent_engraftment',
            hue='group',
            palette=COLOR_PALETTES['group'],
            hue_order=['aging_phenotype', 'no_change'],
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            data=c_df,
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
            fliersize=0,
        )
        c_times = c_df[timepoint_col].unique().tolist()
        c_times.sort()
        for mouse_id, m_df in c_df.groupby('mouse_id'):
            sns.stripplot(
                x=timepoint_col,
                y='percent_engraftment',
                order=c_times,
                hue='group',
                hue_order=['aging_phenotype', 'no_change'],
                palette=COLOR_PALETTES['group'],
                marker=MARKERS['mouse_id'][mouse_id],
                dodge=True,
                size=10,
                linewidth=.7,
                ax=ax,
                data=m_df,
                zorder=0
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

def plot_expanded_abundance_per_mouse(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        abundance_cutoff: float,
        thresholds: Dict[str, float],
        by_sum: bool,
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

    filt_df = agg.filter_cell_type_threshold(
        clonal_abundance_df,
        thresholds,
        analyzed_cell_types=clonal_abundance_df.cell_type.unique(),
    )
    y_col = 'percent_engraftment'
    if by_sum:
        desc = 'sum'
        abundance_per_mouse = pd.DataFrame(
            filt_df.groupby(
            ['group', 'mouse_id', 'cell_type', timepoint_col]
            ).percent_engraftment.sum()
        ).reset_index()
    else:
        desc = 'mean'
        abundance_per_mouse = pd.DataFrame(
            filt_df.groupby(
            ['group', 'mouse_id', 'cell_type', timepoint_col]
            ).percent_engraftment.mean()
        ).reset_index()


    times = abundance_per_mouse[timepoint_col].unique()

    for cell_type, c_df in abundance_per_mouse.groupby(['cell_type']):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test on ' 
            + ' ' + cell_type.title() + ' ' + str(abundance_cutoff)
            + ' Abundant Clone Between Groups'
        )
        for t1, t_df in c_df.groupby(timepoint_col):
            _, p_value = stats.ttest_ind(
                t_df[t_df.group == 'aging_phenotype']['percent_engraftment'],
                t_df[t_df.group == 'no_change']['percent_engraftment'],
            )
            context: str = timepoint_col.title() + ' ' + str(t1) 
            stat_tests.print_p_value(context, p_value, show_ns=False)

        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test on ' 
            + ' ' + cell_type.title() + ' ' + str(abundance_cutoff)
            + ' Abundant Clone Across Time Per Group'
        )
        for group, g_df in c_df.groupby('group'):
            for (t1, t2) in combinations(times, 2):
                t1_df = g_df[g_df[timepoint_col] == t1]['percent_engraftment']
                t2_df = g_df[g_df[timepoint_col] == t2]['percent_engraftment']

                stat, p_value = stats.ttest_ind(
                    t1_df,
                    t2_df, 
                )
                context: str = y_col_to_title(group) + ' ' \
                    + timepoint_col.title() + ' ' + str(t1) \
                    + ' vs ' + str(t2)
                stat_tests.print_p_value(context, p_value)

        plt.figure(figsize=(7,5))
        ax = sns.boxplot(
            x=timepoint_col,
            y='percent_engraftment',
            hue='group',
            palette=COLOR_PALETTES['group'],
            hue_order=['aging_phenotype', 'no_change'],
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            showcaps=False,
            fliersize=0,
            data=c_df,
        )
        c_times = c_df[timepoint_col].unique().tolist()
        c_times.sort()

        for mouse_id, m_df in c_df.groupby('mouse_id'):
            sns.stripplot(
                x=timepoint_col,
                y='percent_engraftment',
                order=c_times,
                hue='group',
                hue_order=['aging_phenotype', 'no_change'],
                palette=COLOR_PALETTES['group'],
                marker=MARKERS['mouse_id'][mouse_id],
                dodge=True,
                size=15,
                linewidth=0.7,
                data=m_df,
                ax=ax,
                zorder=0
            )
        ax.set(yscale='log')
        plt.title(
            cell_type.title()
            + ' Abundance Cutoff: ' + str(abundance_cutoff) 
        )
        plt.ylabel('Cumulative Abundance (% WBC)')
        plt.xlabel(timepoint_col.title())
        plt.legend().remove()
        file_name = save_path + os.sep \
            + 'a-' + str(abundance_cutoff).replace('.', '-') \
            + '_' + desc + '-abundance' \
            + '_' + cell_type \
            + '.' + save_format

        save_plot(file_name, save, save_format)
        
def plot_clone_count_swarm(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        analyzed_cell_types: List[str],
        by_group: bool,
        line: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    save_labels = False
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
    no_6_or_17_month_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    nunique_times = no_6_or_17_month_df[timepoint_col].nunique()
    filt_clonal_abundance_df =agg.filter_mice_with_n_timepoints(
        no_6_or_17_month_df,
        nunique_times,
        time_col=timepoint_col,
    )
    print(Fore.RED + timepoint_col)
    if timepoint_col == 'gen':
        print('Filtering for 8 mice with all timepoints')
        filt_clonal_abundance_df = agg.remove_gen_8_5(
            clonal_abundance_df,
            timepoint_col,
            keep_hsc=False,
        )
        filt_clonal_abundance_df = filt_clonal_abundance_df[filt_clonal_abundance_df.has_all_time == 1]
    threshold_df = agg.filter_cell_type_threshold(
        filt_clonal_abundance_df,
        thresholds, 
        analyzed_cell_types
    )
        
    clone_counts = agg.count_clones(threshold_df, timepoint_col)

    if save_labels:
        labels_df = clonal_abundance_df[['mouse_id', 'group', 'code']].drop_duplicates()
        labels_df.rename(columns={'group': 'label_name'}).to_csv(
            os.path.join(
                save_path,
                'MOLD_labels.csv'
            ),
            index=False
        )
    for cell_type, c_df in clone_counts.groupby(['cell_type']):
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming One Way Repeated Measurement ANOVA for ' + cell_type.title()
        )
        n_timepoints = c_df[timepoint_col].nunique()
        count_cell_df = agg.fill_mouse_id_zeroes(
            c_df,
            info_cols=['group'],
            fill_col='code',
            fill_cat_col=timepoint_col,
            fill_cats=c_df[timepoint_col].unique(),
            fill_val=0

        )
        plt.figure(figsize=(1.5*n_timepoints,5))
        if line:
            ax = sns.lineplot(
                x=timepoint_col,
                y='code',
                hue='mouse_id',
                palette=COLOR_PALETTES['mouse_id'],
                data=count_cell_df,
            )
            sns.scatterplot(
                x=timepoint_col,
                y='code',
                hue='mouse_id',
                palette=COLOR_PALETTES['mouse_id'],
                data=count_cell_df,
                ax=ax
            )
            desc = 'line'
        else:
            if by_group:
                hue_col='group'
                hue_order=['aging_phenotype', 'no_change']
                dodge=True
                box_hue = 'group'
                box_palette = COLOR_PALETTES[hue_col]
                desc = 'by_group_'
                for group, g_df in count_cell_df.groupby('group'):
                    stat_tests.one_way_ANOVArm(
                        data=g_df,
                        timepoint_col=timepoint_col,
                        id_col='mouse_id',
                        value_col='code',
                        overall_context=cell_type + ' ' + group,
                        show_ns=True,
                        match_cols=['mouse_id'],
                        merge_type='inner',
                        fill_na=0,
                    )
                stat_tests.ind_ttest_between_groups_at_each_time(
                    count_cell_df,
                    'code',
                    timepoint_col,
                    cell_type,
                    show_ns=True,
                    group_col='group'
                )

            else:
                hue_col='mouse_id'
                hue_order=None
                box_hue=None
                box_palette=None
                dodge=False
                desc = ''
                stat_tests.one_way_ANOVArm(
                    data=count_cell_df,
                    timepoint_col=timepoint_col,
                    id_col='mouse_id',
                    value_col='code',
                    overall_context=cell_type,
                    show_ns=True,
                    match_cols=['mouse_id'],
                    merge_type='inner',
                    fill_na=0,
                )

            medianprops = dict(
                linewidth=0,
            )
            meanprops = dict(
                linestyle='solid',
                linewidth=3,
                color='black'
            )

            ax = sns.boxplot(
                x=timepoint_col,
                y='code',
                hue=box_hue,
                hue_order=hue_order,
                dodge=dodge,
                showbox=False,
                whiskerprops={
                    "alpha": 0
                },
                palette=box_palette,
                showcaps=False,
                showmeans=True,
                meanline=True,
                meanprops=meanprops,
                medianprops=medianprops,
                data=count_cell_df,
                fliersize=0,
            )
            sns.despine()
            c_times = c_df[timepoint_col].unique().tolist()
            c_times.sort()
            for mouse_id, m_df in count_cell_df.groupby(['mouse_id']):
                sns.stripplot(
                    x=timepoint_col,
                    y='code',
                    order=c_times,
                    hue=hue_col,
                    hue_order=hue_order,
                    palette=COLOR_PALETTES[hue_col],
                    marker=MARKERS['mouse_id'][mouse_id],
                    size=15,
                    linewidth=.5,
                    dodge=dodge,
                    data=m_df,
                    ax=ax,
                    alpha=0.8,
                    zorder=0,
                )
            desc += 'swarm'
        if abundance_cutoff == 0.0:
            plt.title(
                cell_type.title() 
                + ' Clone Count'
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


    no_m17_df = agg.remove_month_17(
        lineage_bias_df,
        timepoint_col
    )
    first_clones = agg.find_first_clones_in_mouse(
        lineage_bias_df,
        timepoint_col
    ).assign(time_desc='First')
    last_clones = agg.find_last_clones_in_mouse(
        lineage_bias_df,
        timepoint_col
    ).assign(time_desc='Last')
    with_time_desc_df = first_clones.append(last_clones)

    filt_df = agg.filter_lineage_bias_thresholds(
        with_time_desc_df,
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
            plt.figure(figsize=(6,6))
            ax = sns.violinplot(
                x='time_desc',
                y='lineage_bias',
                order=['First', 'Last'],
                color='white',
                data=g_df,
                dodge=False,
                cut=0,
                linewidth=5,
                edgecolor='black',
            )
            sns.swarmplot(
                x='time_desc',
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
            sns.despine()
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
            x='time_desc',
            y='lineage_bias',
            color='white',
            order=['First', 'Last'],
            data=filt_df,
            dodge=False,
            cut=0,
        )
        sns.swarmplot(
            x='time_desc',
            y='lineage_bias',
            order=['First', 'Last'],
            hue='group',
            palette=COLOR_PALETTES['group'],
            data=filt_df,
            ax=ax,
        )
        sns.despine()
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
    survival_df = agg.create_lineage_bias_survival_df(
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
                stat_tests.print_p_value(context, p_value)
                

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
    
    filt_survival_df = agg.filter_lymphoid_exhausted_at_time(
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
        filt_survival_all_df = agg.bias_clones_to_abundance(
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
    cats = [agg.MAP_LINEAGE_BIAS_CATEGORY[x] for x in cats]
    colors = get_myeloid_to_lymphoid_colors(cats)
    palette = dict(zip(cats, colors))

    with_bias_cats = agg.add_bias_category(
        lineage_bias_df,
    )    
    first_last_by_mouse_df = agg.filter_first_last_by_mouse(
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
    save_labels = False

    filt_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        keep_hsc=False,
    ) 
    if timepoint_col == 'gen':
        print(Fore.YELLOW + 'Removing M8')
        filt_df = filt_df[filt_df.mouse_id != 'M8']
    threshold_df = agg.filter_cell_type_threshold(
        filt_df,
        thresholds, 
        analyzed_cell_types
    )
        
    clone_counts = agg.count_clones(threshold_df, timepoint_col)
    clone_counts = agg.filter_first_last_by_mouse(
        clone_counts,
        timepoint_col
    )
    print(clone_counts.groupby('mouse_id')[timepoint_col].unique())

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

    if save_labels:
        last_df = agg.get_clones_at_timepoint(
            clonal_abundance_df,
            timepoint_col,
            timepoint='last',
            by_mouse=True
        )
        thresh_last = agg.filter_cell_type_threshold(
            last_df,
            thresholds,
            analyzed_cell_types
        ).assign(is_expanded='Expanded')
        label_cols = ['code', 'group', 'cell_type', 'mouse_id', 'is_expanded']
        labels_df = last_df[label_cols[:-1]].drop_duplicates().merge(
            thresh_last[label_cols].drop_duplicates(),
            how='outer',
            validate='1:1'
        )
        labels_df.loc[labels_df.is_expanded.isna(), 'is_expanded'] = 'not_expanded'

        label_path = os.path.join(
            save_path,
            'expanded_first_last_labels.csv'
        )
        print(Fore.YELLOW + 'Saving labels to: ' + label_path)
        labels_df.to_csv(label_path, index=False)
    for cell_type, c_df in clone_counts.groupby(['cell_type']):
        filled_counts = agg.fill_mouse_id_zeroes(
            c_df,
            ['group'],
            fill_col='code',
            fill_cat_col='mouse_time_desc',
            fill_cats=['First', 'Last'],
            fill_val=0
        )
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test on ' 
            + ' ' + cell_type.title() + ' Clone Clone Counts Between Groups'
        )
        for t1, t_df in filled_counts.groupby('mouse_time_desc'):
            _, p_value = stats.ttest_ind(
                t_df[t_df.group == 'aging_phenotype']['code'],
                t_df[t_df.group == 'no_change']['code'],
            )
            context: str = timepoint_col.title() + ' ' + str(t1)\
                + 'E-MOLD Mice: ' + str(t_df[t_df.group == 'aging_phenotype'].mouse_id.nunique())\
                + ', D-MOLD Mice: ' + str(t_df[t_df.group == 'no_change'].mouse_id.nunique())
            stat_tests.print_p_value(context, p_value, show_ns=True)

        coords = np.arange(2)
        width = 0.4
        sems={}
        means={}
        times = ['First', 'Last']
        colors={}
        _, ax = plt.subplots(figsize=(6,5))
        i = -1
        for group in ['aging_phenotype', 'no_change']:
            means[group]=[]
            colors[group]=[]
            sems[group] = [[], []]
            g_df = filled_counts[filled_counts.group == group]
            for time in times:
                t_df = g_df[g_df['mouse_time_desc'] == time]
                sems[group][0].append(0)
                sems[group][1].append(t_df.code.sem())
                colors[group].append(COLOR_PALETTES['group'][group])
                means[group].append(t_df.code.mean())
        
            ax.bar(
                x=coords + (i*width/2),
                height=means[group],
                width=width,
                tick_label=times,
                color=colors[group],
            )

            _, caps, _ = ax.errorbar(
                coords + (i*width/2),
                means[group],
                yerr=sems[group],
                color='black',
                capsize=10,
                capthick=2,
                ls='none',
                )
            i = i * -1
            caps[0].set_marker('_')
            caps[0].set_markersize(0)
        sns.despine()
        ax.set_xticks(coords)

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
    sns.set_context(
        'paper',
        font_scale=2,
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 25,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 22,
            'ytick.labelsize': 22,
            'figure.titlesize': 'small',
        }
    )

    if timepoint_col == 'gen':
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df[timepoint_col] != 8.5]
        clonal_abundance_df[timepoint_col] = clonal_abundance_df[timepoint_col].astype(int)

    threshold_df = agg.filter_cell_type_threshold(
        clonal_abundance_df,
        thresholds, 
        analyzed_cell_types
    )
        
    clone_counts = agg.count_clones(threshold_df, timepoint_col)
    plt.figure(figsize=(10,6))
    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    ax = sns.boxplot(
        x=timepoint_col,
        y='code',
        hue='cell_type',
        hue_order=analyzed_cell_types ,
        palette=COLOR_PALETTES['cell_type'],
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
        fliersize=0,
        data=clone_counts,
    )
    times = clone_counts[timepoint_col].unique().tolist()
    times.sort()
    stat_tests.ind_ttest_between_groups_at_each_time(
        data=clone_counts[clone_counts.cell_type.isin(analyzed_cell_types)],
        test_col='code',
        timepoint_col=timepoint_col,
        overall_context='count',
        show_ns=True,
        group_col='cell_type'
    )
    for cell_type, c_df in clone_counts.groupby('cell_type'):
        filled_c = agg.fill_mouse_id_zeroes(
            c_df,
            info_cols=['cell_type'],
            fill_col='code',
            fill_cat_col=timepoint_col,
            fill_cats=clone_counts[timepoint_col].unique(),
            fill_val=0
        )
        stat_tests.one_way_ANOVArm(
            data=filled_c,
            match_cols=['mouse_id', 'cell_type'],
            id_col='mouse_id',
            merge_type='inner',
            fill_na=0,
            value_col='code',
            timepoint_col=timepoint_col,
            overall_context=cell_type + ' Counts',
            show_ns=False,
        )
    for mouse_id, m_df in clone_counts.groupby('mouse_id'):
        sns.stripplot(
            x=timepoint_col,
            order=times,
            y='code',
            hue='cell_type',
            hue_order=analyzed_cell_types ,
            palette=COLOR_PALETTES['cell_type'],
            dodge=True,
            ax=ax,
            data=m_df,
            marker=MARKERS['mouse_id'][mouse_id],
            size=10,
            linewidth=.8,
            alpha=0.8,
            zorder=0,
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
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[3:],
        labels[3:],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=3
        )
    plt.legend().remove()
    sns.despine()
    fname = save_path + os.sep \
        + 'clone_count_cell_vs' \
        + '_a' + str(abundance_cutoff).replace('.', '-') \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_perc_survival_bias_heatmap(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
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

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        keep_hsc=False
    )
    lineage_bias_df = agg.remove_month_17_and_6(
        lineage_bias_df,
        timepoint_col
    )
    with_bias_cats_df = agg.add_bias_category(lineage_bias_df)
    survival_df = agg.label_exhausted_clones(
        with_bias_cats_df,
        clonal_abundance_df,
        timepoint_col
    )
    #if timepoint_col == 'gen':
        #survival_df = add_time_difference(
            #with_bias_cats_df,
            #timepoint_col
        #).assign(
            #isLast=lambda x: x.total_time_change == x.time_change
        #)
        #survival_df['survived'] = survival_df.isLast.map(
            #{
                #True: 'Exhausted',
                #False: 'Survived'
            #}
        #)
    if timepoint_col == 'gen':
        survival_df = survival_df[survival_df.has_all_time == 1]
    survival_perc = agg.calculate_survival_perc(
        survival_df,
        timepoint_col,
    )
    cats = [
        'LB',
        'B',
        'MB',
    ]
    filled_dfs = []
    for t, t_df in survival_perc.groupby('last_time'):
        if timepoint_col == 'gen':
            print(Fore.YELLOW + 'Removing generations above 6')
            t_df = t_df[t_df['gen'] < 7]
        if timepoint_col == 'month':
            print(Fore.YELLOW + 'Removing months above 12')
            t_df = t_df[t_df['month'] < 15]
        filled_dfs.append(
            agg.fill_mouse_id_zeroes(
            t_df,
            info_cols=['group', 'last_time'],
            fill_col=y_col,
            fill_cat_col='bias_category',
            fill_cats=cats,
            fill_val=0,
            )
        )
    survival_perc =  pd.concat(filled_dfs, sort=False)
    survival_perc = survival_perc[[
        'last_time',
        y_col,
        'bias_category',
        'mouse_id',
        'group',
    ]].dropna()
    times = survival_perc['last_time'].unique()
    #cats = [agg.MAP_LINEAGE_BIAS_CATEGORY[x] for x in cats]
    #survival_perc['bias_category'] = survival_perc.bias_category.apply(
        #lambda x: agg.MAP_LINEAGE_BIAS_CATEGORY[x]
    #)
    func = np.median
    annot = False
    annot_kws = {
        'fontsize': 20
    }

    group = 'all'
    pivot_df = survival_perc.pivot_table(
        columns='last_time',
        index='bias_category',
        values=y_col,
        aggfunc=func,
        fill_value=0
    )
    vmin = 0
    vmax = 55
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        pivot_df[times].reindex(cats),
        cmap='cividis',
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        annot_kws=annot_kws,
        ax=ax
    )
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()

    hlines = np.linspace(ylims[0], ylims[1], 4)
    ax.hlines(hlines, *xlims, colors='white')

    vlines = np.linspace(*xlims, survival_perc.last_time.nunique() + 1)
    ax.vlines(vlines, *ylims, colors='white')

    plt.title('Group: ' + y_col_to_title(group) + ' n = ' + str(survival_perc.mouse_id.nunique()))
    plt.xlabel('End Point (' + timepoint_col.title() + ')')
    plt.ylabel('Lineage Bias Category')
    plt.suptitle(y_label)
    fname = fname_prefix + group + '.' + save_format
    save_plot(fname, save, save_format)

    survival_perc_str = survival_perc
    for group, g_df in survival_perc.groupby('group'):
        print('Mice: ', g_df.mouse_id.unique())
        pivot_df = g_df.pivot_table(
            columns='last_time',
            index='bias_category',
            values=y_col,
            aggfunc=func,
            fill_value=0
        )
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(
            pivot_df[times].reindex(cats),
            cmap='cividis',
            vmin=vmin,
            vmax=vmax,
            annot_kws=annot_kws,
            annot=annot,
            ax=ax
        )
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        hlines = np.linspace(ylims[0], ylims[1], 4)
        ax.hlines(hlines, *xlims, colors='white')

        vlines = np.linspace(*xlims, survival_perc.last_time.nunique() + 1)
        ax.vlines(vlines, *ylims, colors='white')
        plt.title('Group: ' + y_col_to_title(group) + ' n = ' + str(g_df.mouse_id.nunique()))
        plt.xlabel('End Point (' + timepoint_col.title() + ')')
        plt.ylabel('Lineage Bias Category')
        plt.suptitle(y_label)
        fname = fname_prefix + group + '.' + save_format
        save_plot(fname, save, save_format)


        print(group)
        stat_tests.two_way_ANOVA_rm(
            data=g_df,
            cat1='last_time',
            cat2='bias_category',
            test_col=y_col,
            subject_col='mouse_id'
        )

def plot_hsc_pie_mouse(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:

    clonal_abundance_df = agg.filter_first_last_by_mouse(
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
            bias_change_df = agg.get_bias_change(
                lineage_bias_df,
                timepoint_col,
            )
            lineage_bias_df = agg.mark_changed(
                lineage_bias_df,
                bias_change_df,
                min_time_difference=1,
            )
            lineage_bias_df = lineage_bias_df[lineage_bias_df['change_type'] != 'Unchanged']
        elif change_type.lower() == 'unchanged':
            desc_addon = 'unchanged_clones_'
            bias_change_df = agg.get_bias_change(
                lineage_bias_df,
                timepoint_col,
            )
            lineage_bias_df = agg.mark_changed(
                lineage_bias_df,
                bias_change_df,
                min_time_difference=1,
            )
            lineage_bias_df = lineage_bias_df[lineage_bias_df['change_type'] == 'Unchanged']

    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

    lineage_bias_at_time_df = agg.get_clones_at_timepoint(
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
        by_group: bool,
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
    clonal_abundance_df = agg.find_last_clones_in_mouse(
        clonal_abundance_df,
        timepoint_col,
    )
    cell_type_wide_df = agg.abundance_to_long_by_cell_type(clonal_abundance_df, timepoint_col)
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
        if by_group:
            desc_addon = '_by_group'
            palette_col = 'group'
            color_val_id = group
        else:
            palette_col = 'mouse_id'
            color_val_id = mouse_id
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
                stat_tests.print_p_value('', p_value)
            except ValueError:
                continue
            fig, ax = plt.subplots(figsize=(7,5))

        plt.loglog(
            m_df[x_col],
            m_df[y_col],
            'o',
            markeredgecolor='white',
            markeredgewidth=.5,
            alpha=0.8,
            color=COLOR_PALETTES[palette_col][color_val_id]
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
        no_na_df = cell_type_wide_df[['mouse_id', 'group', 'code', x_col, y_col]].dropna(axis='index')
        no_na_zero_df = no_na_df[
            (no_na_df[x_col] > 0) & \
            (no_na_df[y_col] > 0)
        ]
        if by_group:
            for group, g_df in no_na_zero_df.groupby('group'):
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x=np.log(g_df[x_col]),
                    y=np.log(g_df[y_col]),
                )
                r_squared = r_value**2
                print(Fore.CYAN + Style.BRIGHT + y_col_to_title(group) + ' Log(' + cell_type + ') vs Log(hsc):')
                print(Fore.CYAN + " r-squared:"  + str(r_squared))
                stat_tests.print_p_value('', p_value)
                reg_y = [np.exp(np.log(x) * slope + intercept) for x in g_df['hsc_percent_engraftment']]
                plt.plot(g_df[x_col], reg_y, color=COLOR_PALETTES['group'][group])
        ## Regression using scipy
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=np.log(no_na_zero_df['hsc_percent_engraftment']),
            y=np.log(no_na_zero_df[cell_type+'_percent_engraftment']),
        )
        r_squared = r_value**2
        print(Fore.CYAN + Style.BRIGHT + 'Log(' + cell_type + ') vs Log(hsc):')
        print(Fore.CYAN + " r-squared:"  + str(r_squared))
        stat_tests.print_p_value('', p_value)
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
        group: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 8,
            'axes.linewidth': 3,
            'axes.labelsize': 10,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'figure.titlesize': 'small',
        }
    )
    if group != 'all':
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.group == group]
    clonal_abundance_df = agg.remove_month_17(
        clonal_abundance_df,
        timepoint_col
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
    with_hsc_data_df = with_hsc_data_df[
        with_hsc_data_df.hsc_percent_engraftment > 0
    ]
    g = sns.FacetGrid(
        with_hsc_data_df,
        col=timepoint_col,
        row='cell_type',
        hue='mouse_id',
        palette=COLOR_PALETTES['mouse_id'],
        sharey=False,
        sharex=False,
    )
    kws={
        "marker": 'o',
        "lw": 0,
        "mfc": 'none',
        "mew": 1.5,
    }
    g.map(
        plt.loglog,
        'hsc_percent_engraftment',
        'percent_engraftment',
        **kws
    )
    for ax in g.axes.flat:
        ax.set_yscale('symlog', linthreshy=10e-3)
        ax.set_xscale('symlog', linthreshx=10e-3)
    fname = save_path + os.sep \
        + group + '_' \
        + 'hsc_blood_prod_over_time' \
        + '.' + save_format
    save_plot(fname, save, save_format)

def exhaust_persist_abund(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_count: bool,
        by_sum: bool,
        by_group: bool,
        plot_average: bool,
        cell_type: str,
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
    clonal_abundance_df =  agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )

    exhaustion_labeled_df = agg.label_exhausted_clones(
        None,
        clonal_abundance_df,
        timepoint_col
    )
    abund_desc = ''
    if plot_average and not by_count:
        print('PLOTTING AVERAGE ABUNDANCE UNTIL TIME')
        survival_df = agg.add_avg_abundance_until_timepoint_clonal_abundance_df(
            exhaustion_labeled_df,
            timepoint_col
        )
        y_col = 'avg_abundance'
        abund_desc = '_avg_until'
    else:
        print('PLOTTING ABUNDANCE AT TIME')
        survival_df = exhaustion_labeled_df
        y_col = 'percent_engraftment'
        abund_desc = '_avg_at'

    if timepoint_col  == 'month':
        survival_df = agg.get_clones_at_timepoint(
            survival_df,
            timepoint_col,
            timepoint='last',
            by_mouse=True,
            n=2,
        )
    else:
        print(Fore.YELLOW + 'Warning: counts or abundance will be plotted based on first time point')
        survival_df = agg.get_clones_at_timepoint(
            survival_df,
            timepoint_col,
            timepoint='first',
            by_mouse=True,
        )
    if not by_count:
        survival_df = survival_df[survival_df['cell_type'] == cell_type]
    if by_sum:
        y_desc = cell_type.title() + ' Mean Abundance (% WBC)'
        filename_addon='mean'
        survival_sum_per_mouse = pd.DataFrame(
            survival_df.groupby(['mouse_id', timepoint_col, 'group', 'survived'])[y_col].mean()
            ).reset_index()
        survival_mean_filled = agg.fill_mouse_id_zeroes(
            survival_sum_per_mouse,
            ['group', 'cell_type'],
            fill_col=y_col,
            fill_cat_col='survived',
            fill_cats=survival_sum_per_mouse.survived.unique(),
            fill_val=0,
        )
        survival_cell_df = survival_mean_filled
    elif by_count:
        filename_addon='count'
        y_col = 'code'
        survival_df = survival_df[survival_df.percent_engraftment > 0]
        # Account for 0 abundant clones that don't actually exist in HSC
        survival_cell_df = pd.DataFrame(
            survival_df.groupby(['mouse_id', 'group','survived']).code.nunique()
            )[y_col].unstack(
                fill_value=0
                ).stack().reset_index(name=y_col)
        y_desc = '# of Clones'
    else:
        y_desc = cell_type.title() + ' Average Abundance (% WBC)'
        filename_addon='avg'
        survival_cell_df = survival_df

    show_ns=True
    if by_count or by_sum:
        match_cols = ['mouse_id']
        stat_tests.rel_ttest_group_time(
            survival_cell_df,
            match_cols=match_cols,
            merge_type='outer',
            fill_na=0,
            test_col=y_col,
            timepoint_col='survived',
            overall_context=y_desc,
            show_ns=show_ns
        )
        stat_tests.ind_ttest_group_time(
            survival_cell_df,
            y_col,
            'survived',
            y_desc,
            show_ns=show_ns
        )
    else:
        markers=True
        stat_tests.ranksums_test_group_time(
            survival_cell_df,
            y_col,
            'survived',
            y_desc,
            show_ns=show_ns
        )
        stat_tests.ranksums_test_group_time(
            survival_cell_df,
            y_col,
            'group',
            y_desc,
            show_ns=show_ns,
            group_col='survived'
        )
    



    fig, ax = plt.subplots(figsize=(7,5))

    if by_group:
        hue_col = 'survived'
        filename_addon += '_by-group'
        order = ['aging_phenotype', 'no_change']
        hue_order = ['Exhausted', 'Survived']
        x_col='group'
        palette = COLOR_PALETTES[hue_col]
        dodge = True
    else:
        hue_col = None
        hue_order = None
        x_col='survived'
        order = ['Exhausted', 'Survived']
        palette = COLOR_PALETTES['survived']
        dodge = False

    if not by_count:
        survival_cell_df[y_col] = np.log10(
            10**(-3) + survival_cell_df[y_col]
        )
        ax.set_ylim([-3.1, 1.1])
        ax.set_yticks([-3, -2, -1, 0, 1])

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )


    if by_count or by_sum:
        survival_cell_df['gs'] = survival_cell_df.group.astype(str)\
            + '-' + survival_cell_df.survived.astype(str)
        survival_cell_df['color'] = survival_cell_df.group.map(COLOR_PALETTES['group'])
        df_palette = survival_cell_df[['gs','group', 'color']].drop_duplicates()

        palette = {
            gs: df_palette[df_palette.gs == gs].color.values[0]
            for gs in df_palette.gs.unique()
        }
        stripplot_mouse_markers_with_mean(
            survival_cell_df,
            'gs',
            y_col,
            ax,
            hue_col=None,
            hue_order=None,
            palette=palette
        )
        ax.tick_params(axis='x', labelsize=10)
        #stripplot_label_markers_with_mean(
            #survival_cell_df,
            #x_col,
            #y_col,
            #ax,
            #hue_col,
            #hue_order,
            #order,
        #)
    else:
        sns.boxplot(
            x=x_col,
            y=y_col,
            order=order,
            data=survival_cell_df,
            hue=hue_col,
            hue_order=hue_order,
            dodge=dodge,
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            ax=ax,
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
            fliersize=0,
        )
        sns.swarmplot(
            x=x_col,
            y=y_col,
            order=order,
            data=survival_cell_df,
            hue=hue_col,
            hue_order=hue_order,
            palette=palette,
            linewidth=.5,
            ax=ax,
            dodge=dodge,
            zorder=0
        )
    plt.ylabel(y_desc)
    sns.despine()
    plt.legend().remove()
    fname = save_path + os.sep + 'exhaust_persist_cell_abund' \
        + abund_desc \
        + '_' + cell_type \
        + '_' + filename_addon \
        + '.' + save_format
    save_plot(fname, save, save_format)

def exhaust_persist_hsc_abund(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_count: bool,
        by_clone: bool,
        by_sum: bool,
        by_group: bool,
        save: bool,
        save_path: str = './output',
        save_format: str = 'png'
    ):
    """ Swarmplot + median bar of boxplot overlay of HSC abundance
    split by survived/exhausted clones
    
    Arguments:
        lineage_bias_df {pd.DataFrame}
        clonal_abundance_df {pd.DataFrame}
        timepoint_col {str}
        by_clone {bool} -- Plot Clone Count
        by_sum {bool} -- Plot Sum abbundance
        by_group {bool} -- Split by phenotypic group
        save {bool}
    
    Keyword Arguments:
        save_path {str}
        save_format {str}
    """
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

    save_labels = False
    without_halfgen = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        keep_hsc=True,
    )

    hsc_data = without_halfgen[
        without_halfgen.cell_type == 'hsc'
        ].rename(columns={'percent_engraftment': 'hsc_percent_engraftment'})
        
    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col,
    )
    survival_with_hsc_df = agg.label_exhausted_clones(
        hsc_data[['code', 'mouse_id', 'group', 'hsc_percent_engraftment']],
        clonal_abundance_df,
        timepoint_col,
    )

    if timepoint_col == 'gen':
        print('Unique mice in HSC Data:', hsc_data.mouse_id.unique())
        print('Unique mice in Abundance Data:', clonal_abundance_df.mouse_id.unique())
    survival_with_hsc_df = survival_with_hsc_df[
        survival_with_hsc_df.survived.isin(['Exhausted', 'Survived'])
    ]

    if save_labels:
        labels_df = survival_with_hsc_df[['mouse_id', 'group', 'code', 'survived']].drop_duplicates()
        labels_df.to_csv(
            os.path.join(
                save_path,
                'exhaustion_labels.csv'
            ),
            index=False)
    
    filename_addon = ''
    y_col = 'hsc_percent_engraftment'
    y_desc = 'Average % Abundance in Final HSC Pool'
    show_ns = True

    group_cols = ['mouse_id', 'group', 'survived']
    if by_clone:
        group_cols.append('code')
        filename_addon += '_by_clone'

    if by_sum:
        print('Plotting Sum HSC Abundance')
        filename_addon +='sum'
        y_desc = 'Sum % Abundance in Final HSC Pool'

        survival_with_hsc_df = survival_with_hsc_df[
            group_cols + [y_col]
        ].drop_duplicates()

        survival_sum_per_mouse = pd.DataFrame(
            survival_with_hsc_df.groupby(group_cols)[y_col].sum()
            )[y_col].unstack(
                    fill_value=0
                ).stack().reset_index(name=y_col)

        survival_sum_per_mouse = survival_sum_per_mouse[group_cols + [y_col]].drop_duplicates()
        survival_hsc_df = survival_sum_per_mouse

    elif by_count:
        print('Plotting HSC Count')
        filename_addon +='count'
        y_col = 'code'
        survival_with_hsc_df = survival_with_hsc_df[survival_with_hsc_df.hsc_percent_engraftment > 0]
        survival_hsc_df = pd.DataFrame(
            survival_with_hsc_df.groupby(group_cols).code.nunique()
        ).reset_index()
        survival_hsc_df = agg.fill_mouse_id_zeroes(
            survival_hsc_df,
            info_cols=['group'],
            fill_col=y_col,
            fill_cat_col='survived',
            fill_cats=['Exhausted', 'Survived'],
            fill_val=0
        )
        print(survival_hsc_df)
        y_desc = '# of HSC clones'

    else:
        print('\nPlotting Average HSC Abundance ')
        survival_with_hsc_df = survival_with_hsc_df[
            group_cols + [y_col]
        ]

        if not by_clone:
            print(survival_with_hsc_df.groupby('mouse_id')[y_col].describe(percentiles=[.5]))
            survival_mean_per_mouse = pd.DataFrame(
                survival_with_hsc_df.groupby(group_cols)[y_col].mean()
                )[y_col].unstack(
                        fill_value=0
                    ).stack().reset_index(name=y_col)
            survival_mean_per_mouse = survival_mean_per_mouse[group_cols + [y_col]].drop_duplicates()
            survival_hsc_df = survival_mean_per_mouse 
            survival_hsc_df = agg.fill_mouse_id_zeroes(
                survival_hsc_df,
                info_cols=['group'],
                fill_col=y_col,
                fill_cat_col='survived',
                fill_cats=['Exhausted', 'Survived'],
                fill_val=0
            )
        else:
            survival_hsc_df = survival_with_hsc_df
        filename_addon += 'avg'
        y_desc = 'Average HSC abundance'
    
    if not by_clone:
        match_cols = ['mouse_id']
        if by_clone:
            match_cols.append('code')
        stat_tests.rel_ttest_group_time(
            survival_hsc_df,
            match_cols=match_cols,
            merge_type='outer',
            fill_na=0,
            test_col=y_col,
            timepoint_col='survived',
            overall_context=y_desc,
            show_ns=show_ns
        )
        stat_tests.ind_ttest_group_time(
            survival_hsc_df,
            y_col,
            'survived',
            y_desc,
            show_ns=show_ns
        )
    else:
        markers=True
        stat_tests.ranksums_test_group(
            survival_hsc_df,
            y_col,
            y_desc,
            group_col='survived',
            show_ns=show_ns
        )
        stat_tests.ranksums_test_group_time(
            survival_hsc_df,
            y_col,
            'survived',
            y_desc,
            show_ns=show_ns
        )

    plt.figure(figsize=(7,5))
    if by_group:
        hue_col = 'group'
        filename_addon += '_by-group'
        hue_order = ['aging_phenotype', 'no_change']
        palette = COLOR_PALETTES['group']
        dodge = True
    else:
        hue_col = None
        hue_order = None
        palette = COLOR_PALETTES['survived']
        dodge = False


    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    if by_clone:
        meanprops['linewidth'] = 0
    ax = sns.boxplot(
        x='survived',
        y=y_col,
        order=['Exhausted', 'Survived'],
        data=survival_hsc_df,
        hue=hue_col,
        hue_order=hue_order,
        dodge=dodge,
        fliersize=0,
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
    )


    if not (by_count or by_sum):
        ax.set_yscale('symlog', linthreshy=survival_hsc_df[survival_hsc_df[y_col] > 0][y_col].min())

    if not by_clone:
        for mouse_id, m_df in survival_hsc_df.groupby('mouse_id'):
            sns.stripplot(
                x='survived',
                y=y_col,
                order=['Exhausted', 'Survived'],
                data=m_df,
                hue=hue_col,
                hue_order=hue_order,
                palette=palette,
                marker=MARKERS['mouse_id'][mouse_id],
                linewidth=1,
                ax=ax,
                size=15,
                alpha=0.8,
                dodge=dodge,
                zorder=0
            )
    else:
        sns.boxplot(
            x='survived',
            y=y_col,
            order=['Exhausted', 'Survived'],
            data=survival_hsc_df,
            hue=hue_col,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            dodge=dodge,
            zorder=0
        )
    y_ticks= ax.get_yticks()
    ax.set_yticks([0] + y_ticks)
    sns.despine()
    plt.ylabel(y_desc)
    plt.legend().remove()
    fname = save_path + os.sep + 'exhaust_persist_hsc_abund' \
        + '_' + filename_addon \
        + '.' + save_format
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
    clonal_abundance_df = agg.find_last_clones_in_mouse(
        clonal_abundance_df,
        timepoint_col,
    )
    cell_type_wide_df = agg.abundance_to_long_by_cell_type(clonal_abundance_df, timepoint_col)
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
    outlier_marked_df = agg.mark_outliers(clean_data_df, outlier_res)
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
    clonal_abundance_df = agg.find_last_clones_in_mouse(
        clonal_abundance_df,
        timepoint_col,
    )
    cell_type_wide_df = agg.abundance_to_long_by_cell_type(clonal_abundance_df, timepoint_col)
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
    if group != 'all':
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.group == group]
    
    filt_abundance = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col,
    )
    hsc_df = filt_abundance[filt_abundance.cell_type == 'hsc'].rename(columns={'percent_engraftment': 'hsc_percent_engraftment'})
    not_hsc_df = filt_abundance[filt_abundance.cell_type != 'hsc']
    with_hsc_data_df = not_hsc_df.merge(
        hsc_df[['code','mouse_id', 'hsc_percent_engraftment']],
        on=['code', 'mouse_id'],
        how='inner',
        validate='m:1'
    )
    with_hsc_data_df = with_hsc_data_df[['mouse_id', 'group', 'code', timepoint_col, 'cell_type', 'hsc_percent_engraftment', 'percent_engraftment']].dropna(axis='index')
    with_hsc_data_df = agg.remove_month_17_and_6(
        with_hsc_data_df,
        timepoint_col
    )

    if timepoint_col == 'gen':
        with_hsc_data_df = with_hsc_data_df[with_hsc_data_df[timepoint_col] != 8.5]
        with_hsc_data_df[timepoint_col] = with_hsc_data_df[timepoint_col].astype(int)
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
    elif by_group:
        desc_addon = 'by-group'
        index_col = 'group'
        plt.figure(figsize=(10,7))
        i = 0
        pivotted = pearson_df.pivot_table(
            values='pearson_r',
            index=['cell_type', index_col],
            columns=timepoint_col
        )
        index_order = [
            ('gr', 'aging_phenotype'),
            ('gr', 'no_change'),
            ('b', 'aging_phenotype'),
            ('b', 'no_change'),
        ]
        sns.heatmap(
            pivotted.reindex(index_order),
            vmin=0,
            vmax=1
        )

        
    else:
        pivotted = pearson_df.pivot_table(
            values='pearson_r',
            index='cell_type',
            columns=timepoint_col,
            aggfunc=np.mean,
        )
        sns.heatmap(
            pivotted.reindex(['gr', 'b']),
            vmin=0,
            vmax=1
        )
    
    fname = save_path + os.sep \
        + 'hsc_blood_over_time_heatmap_' \
        + group + '_' \
        + desc_addon \
        + '.' + save_format

    save_plot(fname, save, save_format)

def exhausted_clone_abund(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        cell_type: str,
        group: str,
        by_sum: bool,
        by_count: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    if group != 'all':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

    survival_df = agg.create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )


    y_col = cell_type + '_percent_engraftment'
    if by_sum:
        desc = 'sum'
        abundance_per_mouse = pd.DataFrame(
            survival_df.groupby(
            ['group', 'mouse_id', 'survived', timepoint_col]
            )[y_col].sum()
        ).reset_index()
    elif by_count:
        desc = 'count'
        y_col = 'code'
        abundance_per_mouse = pd.DataFrame(
            survival_df.groupby(
            ['group', 'mouse_id', 'survived', timepoint_col]
            )['code'].nunique()
        ).reset_index()
    else:
        desc = 'mean'
        abundance_per_mouse = pd.DataFrame(
            survival_df.groupby(
            ['group', 'mouse_id', 'survived', timepoint_col]
            )[y_col].mean()
        ).reset_index()
    if timepoint_col == 'gen':
        abundance_per_mouse = abundance_per_mouse[abundance_per_mouse[timepoint_col] != 8.5]
        abundance_per_mouse['gen'] = abundance_per_mouse.gen.astype(int)
    else:
        # Remove last time point
        max_time = abundance_per_mouse[timepoint_col].max()
        abundance_per_mouse = abundance_per_mouse[
            abundance_per_mouse[timepoint_col] != max_time
            ]

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
        + '\nPerforming Paired T-Test of Exhausted vs. Survived'
    )
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\n  - Group: ' + group.replace('_', ' ').title()
    )
    fig, ax = plt.subplots(figsize=(9,6))
    # T-Test on interesting result

    for time, t_df in abundance_per_mouse.groupby(timepoint_col):
        t_s = []
        t_e = []
        for m, m_df in t_df.groupby('mouse_id'):
            m_s = [0]
            m_e = [0]

            m_t_s_df = m_df[m_df['survived'] == 'Survived']
            if not m_t_s_df.empty:
                m_s = m_t_s_df[y_col].values

            m_t_e_df = m_df[m_df['survived'] == 'Exhausted']
            if not m_t_e_df.empty:
                m_e = m_t_e_df[y_col].values

            if (len(m_e) > 1) or (len(m_s) > 1):
                print('\n Mouse: ' + m + ' Time: ' +str(time))
                print(m_e)
                print(m_s)

            t_s.append(m_s[0])
            t_e.append(m_e[0])
            
                
        stat, p_value = stats.ttest_rel(
            t_e,
            t_s,
        )
        context: str = cell_type.title() + ' ' + timepoint_col.title() + ' ' + str(int(time))
        stat_tests.print_p_value(context, p_value)
        

    ax = sns.boxplot(
        x=timepoint_col,
        y=y_col,
        hue='survived',
        data=abundance_per_mouse,
        hue_order=['Exhausted', 'Survived'],
        showbox=True,
        whiskerprops={
            "alpha": 1
        },
        color='white',
        showcaps=False,
        fliersize=0,
    )
    if not by_count:
        ax.set(yscale='log')
        plt.ylabel('Abundance (% WBC)')
    else:
        plt.ylabel('# of Clones')
    
    times = abundance_per_mouse[timepoint_col].unique().tolist()
    times.sort()
    for mouse_id, m_df in abundance_per_mouse.groupby('mouse_id'):
        sns.stripplot(
            x=timepoint_col,
            order=times,
            y=y_col,
            hue='survived',
            hue_order=['Exhausted', 'Survived'],
            palette=COLOR_PALETTES['survived'],
            marker=MARKERS['mouse_id'][mouse_id],
            data=m_df,
            dodge=True,
            size=10,
            linewidth=1
        )
    plt.xlabel(
        timepoint_col.title() 
    )
    plt.legend().remove()
    plt.title('Group: ' + group.replace('_', ' ').title())
    fname = save_path + os.sep \
        + 'exhausted_clone_abund-' \
        + desc \
        + '_' + cell_type \
        + '_' + group \
        + '.' + save_format
    save_plot(fname, save, save_format)

def exhausted_clone_hsc_abund(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        min_hsc_per_mouse: pd.DataFrame,
        group: str,
        by_sum: bool,
        by_count: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    if group != 'all':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

    survival_df = agg.create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )
    hsc_data = agg.get_hsc_abundance_perc_per_mouse(clonal_abundance_df)
    hsc_data = agg.merge_hsc_min_abund(
        hsc_data,
        min_hsc_per_mouse
    )
    real_hsc_data = hsc_data[hsc_data['hsc_percent_engraftment'] > hsc_data['min_eng_hsc']]


    
    survival_with_hsc_df = survival_df.merge(
        real_hsc_data[['mouse_id', 'code', 'perc_tracked_hsc']],
        on=['mouse_id', 'code'],
        how='inner',
        validate='m:1'
    )
    if by_sum:
        desc = 'sum'
        y_col='perc_tracked_hsc'
        abundance_per_mouse = pd.DataFrame(
            survival_with_hsc_df.groupby(
            ['group', 'mouse_id', 'survived', timepoint_col]
            ).perc_tracked_hsc.sum()
        )[y_col].unstack(fill_value=0).stack().reset_index(name=y_col)
    elif by_count:
        desc = 'count'
        y_col='code'
        abundance_per_mouse = pd.DataFrame(
            survival_with_hsc_df.groupby(
            ['group', 'mouse_id', 'survived', timepoint_col]
            )[y_col].nunique()
        )[y_col].unstack(fill_value=0).stack().reset_index(name=y_col)
    else:
        desc = 'mean'
        y_col='perc_tracked_hsc'
        abundance_per_mouse = pd.DataFrame(
            survival_with_hsc_df.groupby(
                ['group', 'mouse_id', 'survived', timepoint_col]
            ).perc_tracked_hsc.mean()
        )[y_col].unstack(fill_value=0).stack().reset_index(name=y_col)
    if timepoint_col == 'gen':
        abundance_per_mouse = abundance_per_mouse[abundance_per_mouse[timepoint_col] != 8.5]
        abundance_per_mouse['gen'] = abundance_per_mouse.gen.astype(int)

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
        + '\nPerforming Paired T-Test of Exhausted vs. Survived'
    )
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\n  - Group: ' + group.replace('_', ' ').title()
    )
    fig, ax = plt.subplots(figsize=(9,6))
    # T-Test on interesting result

    stat_tests.ttest_rel_at_each_time(
        data=abundance_per_mouse,
        value_col=y_col,
        timepoint_col=timepoint_col,
        group_col='survived',
        merge_type='outer',
        match_cols=['group', 'mouse_id', 'survived'],
        overall_context='Exhausted vs Survived ' + desc.title(),
        fill_na=0,
        show_ns=True

    )

    ax = sns.boxplot(
        x=timepoint_col,
        y=y_col,
        hue='survived',
        data=abundance_per_mouse,
        hue_order=['Exhausted', 'Survived'],
        showbox=True,
        whiskerprops={
            "alpha": 1
        },
        color='white',
        showcaps=False,
        fliersize=0,
    )
    times = abundance_per_mouse[timepoint_col].unique().tolist()
    times.sort()
    for mouse_id, m_df in abundance_per_mouse.groupby('mouse_id'):
        sns.stripplot(
            x=timepoint_col,
            order=times,
            y=y_col,
            hue='survived',
            hue_order=['Exhausted', 'Survived'],
            palette=COLOR_PALETTES['survived'],
            marker=MARKERS['mouse_id'][mouse_id],
            data=m_df,
            dodge=True,
            size=10,
            linewidth=1
        )
    plt.xlabel(
        'End Point (' + timepoint_col.title() + ')'
    )
    plt.legend().remove()
    if not by_count:
        plt.ylabel('% Clonal Abundance In Final HSC Pool')
        plt.suptitle(
            'Exhausted Clone Abundance'
        )
    else:
        plt.ylabel('# of HSCs')
    plt.title('Group: ' + group.replace('_', ' ').title())
    plt.axhline(y=0, linestyle='dashed', color='gray')
    fname = save_path + os.sep \
        + 'exhausted_clone_hsc_abund-' \
        + desc \
        + '_' + group \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_dist_bias_at_time_vs_group_facet_grid(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        bins: int,
        change_type: str = None,
        change_status: str = None,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    
    y_cols =  ['count', 'myeloid_percent_abundance', 'lymphoid_percent_abundance']
    lineage_bias_df['count'] = 1
    sns.set_context(
        'paper',
        font_scale=1,
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 10,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }
    )
    bin_edge_count = bins + 1
    dodge_amount = 0.02
    bin_edges = np.linspace(-1, 1, bin_edge_count)
    center_points = (bin_edges[1:] + bin_edges[:-1]) / 2
    dodged_center_points = center_points + dodge_amount
    lineage_bias_df = agg.remove_month_17(
        lineage_bias_df,
        timepoint_col
    )
    desc_addon = ''
    if change_status is not None:
        bias_change_df = agg.calculate_first_last_bias_change(
            lineage_bias_df,
            timepoint_col,
            by_mouse=False,
            exclude_month_17=True,
        )
        lineage_bias_df = agg.mark_changed(
            lineage_bias_df,
            bias_change_df,
            min_time_difference=3,
            merge_type='inner'
        )
        if change_status.lower() == 'changed':
            desc_addon = 'changed_clones_'
            lineage_bias_df = lineage_bias_df[lineage_bias_df['change_type'] != 'Unchanged']
            if change_type in ['Lymphoid', 'Myeloid']:
                print('Filtering on change_type:', change_type)
                lineage_bias_df = lineage_bias_df[lineage_bias_df['change_type'] == change_type]
                desc_addon += change_type+ '_'
        elif change_status.lower() == 'unchanged':
            desc_addon = 'unchanged_clones_'
            lineage_bias_df = lineage_bias_df[lineage_bias_df['change_type'] == 'Unchanged']

    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

    lineage_bias_df = agg.get_clones_exist_first_and_last_per_mouse(
        lineage_bias_df,
        timepoint_col
    ) 

    lineage_bias_at_time_df = agg.get_clones_at_timepoint(
        lineage_bias_df,
        timepoint_col,
        'last',
        by_mouse=True,
    ).assign(time_desc='last')
    lineage_bias_at_time_df = lineage_bias_at_time_df.append(
        agg.get_clones_at_timepoint(
            lineage_bias_df,
            timepoint_col,
            'first',
            by_mouse=True,
        ).assign(time_desc='first')
    )



    hist_df = pd.DataFrame()
    for (mouse_id, group, time_desc), m_df in lineage_bias_at_time_df.groupby(['mouse_id', 'group', 'time_desc']):
        hists = {
            y_col: np.histogram(
                m_df.lineage_bias,
                bins=bin_edges,
                weights=m_df[y_col]
                )[0]
                for y_col in y_cols
        }

        # Slight x-axis shift to no_change group allowign for visibility of lines/error bars
        if group == 'no_change':
            m_row = pd.DataFrame.from_dict(
                {
                    **hists,
                    'lineage_bias': dodged_center_points,
                    'mouse_id': [mouse_id] * (bin_edge_count - 1),
                    'group': [group] * (bin_edge_count - 1),
                    'time_desc': time_desc,
                }
            )
        else:
            m_row = pd.DataFrame.from_dict(
                {
                    **hists,
                    'lineage_bias': center_points,
                    'mouse_id': [mouse_id] * (bin_edge_count - 1),
                    'group': [group] * (bin_edge_count - 1),
                    'time_desc': time_desc,
                }
            )
        hist_df = hist_df.append(m_row, ignore_index=True)

    for y_col in y_cols:
        for time, t_df in hist_df.groupby('time_desc'):
            temp_df = t_df.copy()
            temp_df.loc[temp_df.group == 'no_change', 'lineage_bias'] = temp_df.loc[temp_df.group == 'no_change', 'lineage_bias'] - dodge_amount
            stat_tests.ind_ttest_group_time(
                data=temp_df,
                test_col=y_col,
                timepoint_col='lineage_bias',
                overall_context=desc_addon + ' ' + time + ' ' + y_col,
                show_ns=False
            )

    hist_df = pd.melt(
        hist_df,
        id_vars=['mouse_id', 'group', 'lineage_bias', 'time_desc'],
        value_vars=y_cols,
        )

    g = sns.FacetGrid(
        hist_df,
        col="time_desc",
        row='variable',
        hue='group',
        palette=COLOR_PALETTES['group'],
        sharey='row',
        aspect=2,
    )
    g = (g.map(
        sns.lineplot,
        "lineage_bias",
        "value",
        err_style='bars',
    ))
    for ax in g.axes.flat:
        ax.tick_params(axis='y', labelleft=True)
        ax.set_ylabel('')

    fname = save_path + os.sep \
        + 'face_grid_lineage_bias_dist_' \
        + desc_addon \
        + '_' + str(bins) + '-bins' \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_hsc_abundance_by_change(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        by_clone: bool,
        by_group: bool,
        timepoint: Any = None,
        by_sum: bool = False,
        by_mean: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    save_labels = False
    if by_sum and by_clone:
        raise ValueError('Cannot both count clones and sum of percent engraftment')
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 5,
            'axes.linewidth': 3,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.major.size': 4,
            'ytick.minor.size': 4,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'medium',
        }
        )
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False,
    )
    change_marked_df = agg.mark_changed(
        agg.remove_month_17(
            clonal_abundance_df,
            timepoint_col
        ),
        bias_change_df,
        min_time_difference=mtd,
        timepoint=timepoint
    )
    if save_labels:
        label_path = os.path.join(
            save_path,
            'changed_labels.csv'
        )
        print(Fore.YELLOW + 'Saving Labels To: ' + label_path )
        change_marked_df[[
            'code',
            'group',
            'mouse_id',
            'change_type',
            'change_status',
        ]].drop_duplicates().to_csv(label_path, index=False)
        print('Mice/Groups in labels\n', change_marked_df[['mouse_id', 'group']].drop_duplicates())
    desc_add = 'each_clone'
    y_col = 'percent_engraftment'
    y_desc = 'HSC Abundance (%HSC)'
    hsc_data = change_marked_df[change_marked_df.cell_type == 'hsc']
    mouse_markers=False
    if by_sum:
        hsc_data = pd.DataFrame(
            hsc_data.groupby([
                'mouse_id',
                'group',
                'cell_type',
                'change_type',
                timepoint_col,
            ]).percent_engraftment.sum()
        )[y_col].unstack(
                fill_value=0
            ).stack().reset_index(name=y_col)
        hsc_data = agg.fill_mouse_id_zeroes(
            hsc_data,
            info_cols=['group'],
            fill_col=y_col,
            fill_cat_col='change_type',
            fill_cats=['Unchanged', 'Lymphoid', 'Myeloid'],
            fill_val=0
        )
        desc_add = 'sum'
        mouse_markers = True
    elif by_mean:
        hsc_data = pd.DataFrame(
            hsc_data.groupby([
                'mouse_id',
                'group',
                'cell_type',
                'change_type',
                timepoint_col,
            ]).percent_engraftment.mean()
        )[y_col].unstack(
                fill_value=0
            ).stack().reset_index(name=y_col)
        hsc_data = agg.fill_mouse_id_zeroes(
            hsc_data,
            info_cols=['group'],
            fill_col=y_col,
            fill_cat_col='change_type',
            fill_cats=['Unchanged', 'Lymphoid', 'Myeloid'],
            fill_val=0
        )
        desc_add = 'mean'
        mouse_markers = True
    elif by_clone:
        hsc_data = hsc_data[hsc_data.percent_engraftment > 0]
        y_col = 'code'
        hsc_data = pd.DataFrame(
            hsc_data.groupby([
                'mouse_id',
                'group',
                'cell_type',
                'change_type',
            ]).code.nunique()
        )[y_col].unstack(
                fill_value=0
            ).stack().reset_index(name=y_col)
        hsc_data = agg.fill_mouse_id_zeroes(
            hsc_data,
            info_cols=['group'],
            fill_col=y_col,
            fill_cat_col='change_type',
            fill_cats=['Unchanged', 'Lymphoid', 'Myeloid'],
            fill_val=0
        )
        #print('SAVING DATA TO:', os.path.join(save_path, 'data_counts.csv'))
        #hsc_data.to_csv(os.path.join(save_path, 'data_counts.csv'), index=False)
        desc_add = 'count'
        y_desc = '# Of HSC Clones'
        mouse_markers = True

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Tests On: '+ desc_add.title() 
    )
    show_ns = True
    if mouse_markers:
        match_cols = ['mouse_id']
        for group, g_df in hsc_data.groupby('group'):
            #stat_tests.one_way_ANOVArm(
                #data=g_df,
                #timepoint_col='change_type',
                #id_col='mouse_id',
                #value_col=y_col,
                #overall_context=group + ' ' + y_desc,
                #show_ns=show_ns,
                #match_cols=['mouse_id', 'group'],
                #merge_type='inner',
                #fill_na=0
            #)
            stat_tests.anova_oneway(
                data=g_df,
                category_col='change_type',
                value_col=y_col,
                overall_context=group + ' ' + y_desc,
                show_ns=show_ns
            )
        stat_tests.ind_ttest_between_groups_at_each_time(
            hsc_data,
            test_col=y_col,
            timepoint_col='change_type',
            overall_context=y_desc,
            show_ns=show_ns
        )
    else:
        markers=True
        stat_tests.ranksums_test_group_time(
            hsc_data,
            y_col,
            'change_type',
            'HSC Abundance',
            show_ns=show_ns
        )
        if by_group:
            for group, g_df in hsc_data.groupby('group'):
                stat_tests.ranksums_test_group(
                    g_df,
                    group_col='change_type',
                    test_col=y_col,
                    overall_context=group + ' HSC Abundance',
                    show_ns=show_ns,
                )
        else:
            stat_tests.one_way_ANOVArm(
                hsc_data,
                timepoint_col='change_type',
                id_col='code',
                value_col=y_col,
                overall_context='HSC Abundance',
                show_ns=show_ns,
                match_cols=['mouse_id', 'code'],
                merge_type='inner',
                fill_na=0
            )

    fig, ax = plt.subplots(figsize=(7,5))
    sns.despine()
    if not by_clone:
        ax.set_yscale('log')
    if by_group:
        hue_col = 'group'
        desc_add += '_by-group'
        hue_order = ['aging_phenotype', 'no_change']
        palette = COLOR_PALETTES['group']
        dodge = True
    else:
        hue_col = None
        hue_order = None
        palette = COLOR_PALETTES['change_type']
        dodge = False

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    sns.boxplot(
        x='change_type',
        y=y_col,
        hue=hue_col,
        hue_order=hue_order,
        palette=palette,
        order=['Unchanged', 'Lymphoid', 'Myeloid'],
        data=hsc_data,
        ax=ax,
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        dodge=dodge,
        showcaps=False,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
        fliersize=0,
    )
    if mouse_markers:
        for mouse_id, m_df in hsc_data.groupby('mouse_id'):
            sns.stripplot(
                x='change_type',
                y=y_col,
                hue=hue_col,
                hue_order=hue_order,
                marker=MARKERS['mouse_id'][mouse_id],
                palette=palette,
                order=['Unchanged', 'Lymphoid', 'Myeloid'],
                data=m_df,
                ax=ax,
                zorder=0,
                size=15,
                linewidth=.8,
                dodge=dodge,
            )
    else:
        sns.swarmplot(
            x='change_type',
            y=y_col,
            hue=hue_col,
            hue_order=hue_order,
            palette=palette,
            order=['Unchanged', 'Lymphoid', 'Myeloid'],
            data=hsc_data,
            linewidth=.5,
            ax=ax,
            zorder=0,
            dodge=dodge,
        )
    plt.ylabel(y_desc)
    plt.xlabel('')
    plt.legend().remove()
    fname_prefix = save_path + os.sep \
        + 'abundance_by_bias_change' \
        + 't' + str(timepoint) \
        + '_mtd' + str(mtd)
    fname = fname_prefix \
        + '_' + desc_add \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_abundance_changed_group_grid(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    """ Plot a facet grid of Gr and B abundance between initial 
    and last timepoint. One line per mouse
    
    Arguments:
        lineage_bias_df {pd.DataFrame}
        timepoint_col {str}
    
    Keyword Arguments:
        save {bool}
        save_path {str}
        save_format {str}
    
    Returns:
        None
    """
    
    lineage_bias_df = agg.get_clones_exist_first_and_last_per_mouse(
        lineage_bias_df,
        timepoint_col
    ) 
    y_cols =  ['gr_percent_engraftment', 'b_percent_engraftment']
    group_cols = ['mouse_id', 'group', 'time_desc', 'change_status']

    sns.set_context(
        'paper',
        font_scale=1,
        rc={
            'lines.linewidth': 1,
            'axes.linewidth': 3,
            'axes.labelsize': 5,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'figure.titlesize': 'small',
        }
    )
    bias_change_df = agg.get_bias_change(
        lineage_bias_df,
        timepoint_col,
    )
    lineage_bias_df = agg.mark_changed(
        lineage_bias_df,
        bias_change_df,
        min_time_difference=1,
    )

    if timepoint_col == 'gen':
        lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

    lineage_bias_at_time_df = agg.find_last_clones_in_mouse(
        lineage_bias_df,
        timepoint_col,
        'last'
    ).assign(time_desc='last')
    lineage_bias_at_time_df = lineage_bias_at_time_df.append(
        agg.get_clones_at_timepoint(
            lineage_bias_df,
            timepoint_col,
            'first'
        ).assign(time_desc='first')
    )

    avg_abund_per_mouse = pd.DataFrame(lineage_bias_at_time_df.groupby(
        group_cols
        )[y_cols].sum()).reset_index()

    melt_df = pd.melt(
        avg_abund_per_mouse,
        id_vars=group_cols,
        value_vars=y_cols
    )
    if not melt_df[melt_df['value'] == 0].empty:
        zero_val = melt_df[melt_df['value'] != 0].value.min()/100
        melt_df['value'] = melt_df['value'] + zero_val
        print(Fore.YELLOW + 'Zero found, adding: ' + str(zero_val) + ' To All Values')

    melt_df['change_status'] = melt_df['change_status'].str[0]
    melt_df['group'] = melt_df['group'].str[0]
    melt_df['change-group'] = melt_df['change_status'].str.cat(melt_df['group'], sep='-')
    col_order = ['U-a', 'U-n', 'C-a', 'C-n']
    g = sns.FacetGrid(
        melt_df,
        col='change-group',
        col_order=col_order,
        row='variable',
        hue='mouse_id',
        palette=COLOR_PALETTES['mouse_id'],
        sharey='row',
    )
    g = (g.map(
        sns.pointplot,
        "time_desc",
        "value",
        order=['first', 'last'],
        dodge=True,
    ).set(yscale='log'))
    for ax in g.axes.flat:
        ax.tick_params(axis='y', labelleft=True)

    fname = save_path + os.sep \
        + 'face_grid_abundance_change-type' \
       + '.' + save_format
    save_plot(fname, save, save_format)


def plot_abundance_change_changed_group_grid(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        by_mouse:bool,
        mtd: int,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    """ Plot a scatter/stripplot of abundance change between first 
    and last time point per mouse.
    
    Arguments:
        lineage_bias_df {pd.DataFrame}
        timepoint_col {str} 
        by_mouse {bool} -- set to plot avg change per mouse
        by_clone {bool} -- set to plot absolute difference instead of log2xChange
    
    Keyword Arguments:
        save {bool} 
        save_path {str}
        save_format {str}
    
    Returns:
        None
    """
    change_param = 'change_type'
    #change_param = 'change_status
    y_cols =  ['myeloid_change', 'lymphoid_change']
    group_cols = ['mouse_id', 'group', change_param]

    if by_mouse:
        group_desc = 'by-mouse'
    else:
        group_desc = 'by-clone'

    # NOTE By_Clone here is used as a flag for log2 fold vs linear change
    math_desc = 'absolute_change'

    sns.set_context(
        'paper',
        font_scale=1,
        rc={
            'lines.linewidth': 1,
            'axes.linewidth': 3,
            'axes.labelsize': 5,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'figure.titlesize': 'small',
        }
    )
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False,
    )
    marked_bias_change_df = agg.mark_changed(
        bias_change_df,
        bias_change_df,
        min_time_difference=mtd,
    )
    marked_bias_change_df= marked_bias_change_df.assign(
        myeloid_change=lambda x: x.myeloid_percent_abundance_last - x.myeloid_percent_abundance_first,
        lymphoid_change=lambda x: x.lymphoid_percent_abundance_last - x.lymphoid_percent_abundance_first,
    )
    if timepoint_col == 'gen':
        marked_bias_change_df = marked_bias_change_df[marked_bias_change_df.gen != 8.5]


    if by_mouse:
        avg_abund_per_mouse = pd.DataFrame(marked_bias_change_df.groupby(
            group_cols
            )[y_cols].mean()).reset_index()
        
        melt_df = pd.melt(
            avg_abund_per_mouse,
            id_vars=group_cols,
            value_vars=y_cols
        )
    else:
        melt_df = pd.melt(
            marked_bias_change_df,
            id_vars=['code'] + group_cols,
            value_vars=y_cols
        )
    melt_df[change_param] = melt_df[change_param].str[0]
    melt_df['group_first'] = melt_df['group'].str[0]
    melt_df['change-group'] = melt_df[change_param].str.cat(melt_df['group_first'], sep='-')
    if change_param == 'change_status':
        col_order = ['U-a', 'U-n', 'C-a', 'C-n']
    elif change_param == 'change_type':
        col_order = ['U-a', 'U-n', 'L-a', 'L-n', 'M-a', 'M-n']

    # Find sig difference
    nan_inf_handle = 'propagate'

    show_ns=True
    stat_tests.ttest_1samp(
        data=melt_df,
        group_vars=['change-group', 'variable'],
        value_var='value',
        null_mean=0,
        overall_context='Phenotype-Change Clone Abundance Change ' + math_desc + ' ' + group_desc,
        show_ns=show_ns,
        handle_nan=nan_inf_handle,
        handle_inf=nan_inf_handle,
    )

    for var, v_df in melt_df.groupby('variable'):
        stat_tests.ranksums_test_group_time(
            data=v_df,
            test_col='value',
            timepoint_col=change_param,
            overall_context=var,
            group_col='group',
            show_ns=show_ns,
        )
    print(Fore.RED + 'FACET STARTING') 
    g = sns.FacetGrid(
        melt_df,
        row='variable',
        hue='group',
        palette=COLOR_PALETTES['group'],
        sharey='row',
        aspect=3
    )
    print(Fore.RED + 'FORMATTING AXES') 
    for ax in g.axes.flat:
        ax.tick_params(axis='y', labelleft=True)
        ax.axhline(y=0, color='gray', linestyle='dashed', linewidth=1.5, zorder=0)
        ax.set_yscale('symlog', linthreshy=10e-2)

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=2,
        color='#2f3640'
    )
    def violin_mean(x, y, **kwargs):
        ax = sns.boxenplot(
            x=x,
            y=y,
            lw=0,
            **kwargs,
        )
        sns.boxplot(
            x=x,
            y=y,
            whiskerprops={
                "alpha": 0
            },
            order=col_order,
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
            ax=ax,
            fliersize=0,
            showbox=False,
        )
    if by_mouse:
        g.map(
            sns.swarmplot,
            "change-group",
            "value",
            order=col_order,
            zorder=0,
        )
    else:
        print(Fore.RED + 'MAPPING TO FACETGRID') 
        g.map(
            violin_mean,
            "change-group",
            "value",
            order=col_order,
        )


    fname = save_path + os.sep \
        + 'face_grid_abundance_change_' \
        + change_param \
        + '_' + math_desc \
        + '_' + group_desc \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_violin_mean(x, y, **kwargs):
    data = x.to_frame().join(y)
    cols = data.columns
    means = pd.DataFrame(data.groupby(cols[0])[cols[1]].mean()).reset_index()
    
    sns.violinplot(
        x=x,
        y=y,
        **kwargs
    )
    plt.scatter(
        means[cols[0]],
        means[cols[1]],
        marker='_',
        s=200,
        c='black',
        linewidth=3,
        zorder=2,
    )

def plot_hsc_and_blood_clone_count(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        exclude_timepoints: List[Any],
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    """ Plot a stripplot of HSC Clone count and Blood Clone count
    
    Arguments:
        clonal_abundance_df {pd.DataFrame}
        timepoint_col {str}
        exclude_timepoints {List[Any]} -- List of time points to not use for clone counting
        by_group {bool} -- split stripplot by group
    
    Keyword Arguments:
        save {bool
        save_path {str}
        save_format {str}
    
    Returns:
        None
    """
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
    if len(exclude_timepoints):
        print(
            Fore.YELLOW 
            + ' Warning: Excluding Timepoints from blood count  - ' 
            + ', '.join([str(x) for x in exclude_timepoints])
        )
        exclude_timepoints_str = 'e' + '-'.join([str(x) for x in exclude_timepoints])
    else:
        exclude_timepoints_str = 'e-None'
    desc_addon = ''

    if timepoint_col == 'month':
        clonal_abundance_df = agg.remove_month_17_and_6(
            clonal_abundance_df,
            timepoint_col
        )
    all_timepoints =agg.filter_mice_with_n_timepoints(
        clonal_abundance_df,
        clonal_abundance_df[timepoint_col].nunique()
    )

    real_hsc_data = all_timepoints[
        (all_timepoints.cell_type == 'hsc') &\
        (all_timepoints.percent_engraftment > 0)
    ]

    blood_data = all_timepoints[
        all_timepoints.cell_type != 'hsc'
    ]
    last_blood_data = agg.get_clones_at_timepoint(
        blood_data,
        timepoint_col,
        'last',
        by_mouse=True,
    )
    last_blood_data = last_blood_data[last_blood_data.percent_engraftment > 0]
    excluded_time_blood_data = blood_data[
        ~blood_data[timepoint_col].isin(exclude_timepoints) &\
        blood_data['percent_engraftment'] > 0
    ]
    print(
        'With time length:',
        len(blood_data),
        'Without Time Length:',
        len(excluded_time_blood_data)
    )

    hsc_counts = pd.DataFrame(
        real_hsc_data.groupby(['group', 'mouse_id'])['code'].nunique()
    ).reset_index().assign(sample_type='HSC')


    blood_counts = pd.DataFrame(
        excluded_time_blood_data.groupby(['group', 'mouse_id'])['code'].nunique()
    ).reset_index().assign(sample_type='Blood All')

    last_blood_counts = pd.DataFrame(
        last_blood_data.groupby(['group', 'mouse_id'])['code'].nunique()
    ).reset_index().assign(sample_type='Blood Last')

    counts = hsc_counts.append(blood_counts, ignore_index=True).append(last_blood_counts, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6,5))
    ax = plt.gca()
    order=['HSC', 'Blood Last']
    y_col = 'code'
    counts = counts[counts.sample_type.isin(order)]

    show_ns = True
    stat_tests.rel_ttest_group_time(
        data=counts,
        match_cols=['mouse_id'],
        merge_type='outer',
        fill_na=0,
        test_col='code',
        timepoint_col='sample_type',
        overall_context='Blood vs HSC Clone Count',
        show_ns=show_ns
    )
    stat_tests.ind_ttest_between_groups_at_each_time(
        data=counts,
        test_col='code',
        timepoint_col='sample_type',
        overall_context='Blood vs HSC Clone Count',
        show_ns=show_ns,
    )
    sns.despine()

    if by_group:
        desc_addon = '_by-group'
        hue_col = 'group'
        box_hue_col = 'group'
        hue_order = ['aging_phenotype', 'no_change']
        palette = COLOR_PALETTES['group']
        box_palette = COLOR_PALETTES['group']
        dodge = True
    else:
        hue_col = 'mouse_id'
        hue_order = None
        box_hue_col = None
        palette = COLOR_PALETTES['mouse_id']
        box_palette = None
        dodge = False

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    sns.boxplot(
        x='sample_type',
        y=y_col,
        hue=box_hue_col,
        hue_order=hue_order,
        palette=box_palette,
        order=order,
        data=counts,
        ax=ax,
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        fliersize=0,
        dodge=dodge,
        meanline=True,
        showmeans=True,
        meanprops=meanprops,
        medianprops=medianprops,
    )
    for mouse_id, m_df in counts.groupby('mouse_id'):
        sns.stripplot(
            x='sample_type',
            y=y_col,
            hue=hue_col,
            hue_order=hue_order,
            marker=MARKERS['mouse_id'][mouse_id],
            palette=palette,
            order=order,
            data=m_df,
            ax=ax,
            zorder=0,
            size=15,
            linewidth=.8,
            alpha=0.8,
            dodge=dodge,
        )
    plt.legend().remove()

    fname = save_path + os.sep \
        + 'hsc_vs_blood_count' \
        + desc_addon \
        + '_' + exclude_timepoints_str \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_hsc_vs_cell_type_abundance_bootstrapped(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        alpha: float,
        thresholds: Dict[str, float],
        cell_type: str,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> pd.DataFrame:
    sns.set_context(
        'paper',
        font_scale=1.5,
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 25,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }
    )
    tenx_mice_only = False # Plot only 10X sequenced mice
    plot_b = False # Plots only B abundance on Y-axis, if cell_type gr, plots B abundance of isolated clones
    show_0 = True # Show symlog plot with 0 abundant HSCs
    save_labels = False # Save code-hsc abundance isolated labels
    plot_boundary = False# Plot decision boundary
    plot_upper = False # Plot decision boundary upper
    mark_decision = True # Wether to create bootstrap confidence intervals or na
    plot_young = False # wether to import and also plot young mice
    do_stats = False 

    
    clonal_abundance_df = agg.remove_month_17(
        clonal_abundance_df,
        timepoint_col
    )
    if plot_young:
        young_path = '/Users/akre96/Data/HSC_aging_project/young_mice/Ania_M_allAnia 10x hsc 8mice rerun_percent-engraftment_NO filter_013120_long.csv'
        print(Fore.YELLOW + 'ADDING YOUNG MOUSE DATA FROM: ' + young_path)
        young_abundance_df = pd.read_csv(young_path)
        young_abundance_df['group'] = 'young'
        young_abundance_df['month'] = agg.day_to_month(young_abundance_df.day)
        clonal_abundance_df = pd.concat([clonal_abundance_df,young_abundance_df])
    if show_0:
        filt_df = clonal_abundance_df[
            clonal_abundance_df.cell_type.isin([cell_type, 'hsc'])
        ]
    else:
        filt_df = agg.filter_cell_type_threshold(
            clonal_abundance_df,
            thresholds,
            analyzed_cell_types=[cell_type, 'hsc']
        )
    print(filt_df)
    y_col = cell_type + '_percent_engraftment'
    x_col = 'hsc_percent_engraftment'
    wide_ct_abundance_df = agg.abundance_to_long_by_cell_type(
        filt_df,
        timepoint_col
    ).dropna(subset=[x_col, y_col])
    wide_ct_abundance_df = wide_ct_abundance_df[
        wide_ct_abundance_df.hsc_percent_engraftment > 0
    ]

    _, ax = plt.subplots(figsize=(7,5))

    sub_sample_amount = .95
    sub_sample_count = round(len(wide_ct_abundance_df) * sub_sample_amount)
    n_sub_samples = 100
    # Log Transform Data:
    wide_ct_abundance_df[x_col] = np.log10(1+(wide_ct_abundance_df[x_col] * 1000))
    wide_ct_abundance_df[y_col] = np.log10(1+(wide_ct_abundance_df[y_col] * 1000))
    

    if mark_decision:
        for i in progressbar.progressbar(range(n_sub_samples)):
            sub_sample_df = wide_ct_abundance_df.sample(
                n=sub_sample_count,
                random_state=i
            )
            reg_model = sm.OLS(
                sub_sample_df[y_col],
                sub_sample_df[x_col]
            )
            res = reg_model.fit()
            prstd, iv_l, iv_u = wls_prediction_std(res, alpha=alpha)
            x, y = agg.sort_xy_lists(sub_sample_df[x_col], iv_l)
            xu, yu = agg.sort_xy_lists(sub_sample_df[x_col], iv_u)


            if i == 0:
                lower_lims = pd.DataFrame.from_dict({'hsc_percent_engraftment': x, str(i): y}).drop_duplicates()
                upper_lims = pd.DataFrame.from_dict({'hsc_percent_engraftment': xu, str(i): yu}).drop_duplicates()
            else:
                lower_lims = lower_lims.merge(
                    pd.DataFrame.from_dict({'hsc_percent_engraftment': x, str(i): y}).drop_duplicates(),
                    how='outer',
                    validate='1:1'
                )
                upper_lims = upper_lims.merge(
                    pd.DataFrame.from_dict({'hsc_percent_engraftment': xu, str(i): yu}).drop_duplicates(),
                    how='outer',
                    validate='1:1'
                )

        boundary = lower_lims.set_index('hsc_percent_engraftment').min(axis=1)
        boundary = boundary.reset_index().rename(columns={0: 'boundary'})
        upper_boundary = upper_lims.set_index('hsc_percent_engraftment').min(axis=1)
        upper_boundary = upper_boundary.reset_index().rename(columns={0: 'boundary'})
        with_boundary_df = wide_ct_abundance_df.merge(
            boundary,
            how='inner',
            validate='m:1'
        )
        with_boundary_df['in_boundary'] = with_boundary_df[y_col] < with_boundary_df['boundary']
    else:
        with_boundary_df = wide_ct_abundance_df
        with_boundary_df['in_boundary'] = False

    if save_labels:
        labels_df = with_boundary_df[['code', 'mouse_id', 'group', 'in_boundary', y_col, 'hsc_percent_engraftment']].drop_duplicates()
        label_path = os.path.join(
            save_path,
            cell_type + '_hsc_bound_labels.csv'
        )
        print(Fore.YELLOW + 'Saving labels to: ' + label_path)
        labels_df.to_csv(label_path, index=False)
    # PLOTS B ABUNDANCE FOR CELLS IN GR ISOLATED
    if by_group:
        hue_col = 'group'
    else:
        hue_col = 'mouse_id'
    
    if tenx_mice_only:
        print(with_boundary_df.mouse_id.unique())
        with_boundary_df = with_boundary_df[with_boundary_df.mouse_id.isin(TENX_MICE)]
    inv_desc = ''
    if (cell_type == 'gr') and plot_b:
        inv_desc = '_b_from_gr'
        print('PLOTTING B ABUNDANCE')
        cell_type = 'b'
        alt_ct_df = clonal_abundance_df[
            clonal_abundance_df.cell_type.isin(['b', 'hsc'])
        ]
        y_col = 'b' + '_percent_engraftment'
        wide_alt_ct_abundance_df = agg.abundance_to_long_by_cell_type(
            alt_ct_df,
            timepoint_col
        ).dropna(subset=[x_col, y_col])
        wide_alt_ct_abundance_df[x_col] = np.log10(1+(wide_alt_ct_abundance_df[x_col] * 1000))
        wide_alt_ct_abundance_df[y_col] = np.log10(1+(wide_alt_ct_abundance_df[y_col] * 1000))
        for h_val, m_df in with_boundary_df.groupby(hue_col):
            in_boundary = m_df[m_df.in_boundary]
            not_in_boundary = m_df[~m_df.in_boundary]
            not_in_boundary_alt = wide_alt_ct_abundance_df.merge(
                not_in_boundary[['code', 'mouse_id']].drop_duplicates()
            )
            in_boundary_alt = wide_alt_ct_abundance_df.merge(
                in_boundary[['code', 'mouse_id']].drop_duplicates()
            )
            ax.scatter(
                not_in_boundary_alt[x_col],
                not_in_boundary_alt[y_col],
                edgecolors=COLOR_PALETTES[hue_col][h_val],
                s=100,
                linewidths=2,
                c=[(0, 0, 0, 0)] * len(not_in_boundary_alt[x_col])
            )
            ax.scatter(
                in_boundary_alt[x_col],
                in_boundary_alt[y_col],
                c=COLOR_PALETTES[hue_col][h_val],
                s=100,
                alpha=0.8,
                linewidths=2,
                edgecolors='#d63031'
            )
    else:
        for h_val, m_df in with_boundary_df.groupby(hue_col):
            not_in_boundary = m_df[~m_df.in_boundary]
            in_boundary = m_df[m_df.in_boundary]
            if plot_young and h_val == 'young':
                ax.scatter(
                    not_in_boundary[x_col],
                    not_in_boundary[y_col],
                    c=COLOR_PALETTES[hue_col][h_val],
                    marker='x',
                    alpha=0.7,
                    s=50,
                    linewidths=2,
                )
            else:
                ax.scatter(
                    not_in_boundary[x_col],
                    not_in_boundary[y_col],
                    edgecolors=COLOR_PALETTES[hue_col][h_val],
                    s=100,
                    linewidths=2,
                    c=[(0, 0, 0, 0)] * len(not_in_boundary[x_col])
                )
            ax.scatter(
                in_boundary[x_col],
                in_boundary[y_col],
                c=COLOR_PALETTES[hue_col][h_val],
                s=100,
                alpha=0.8,
                linewidths=2,
                edgecolors='#d63031'
            )
    totals = pd.DataFrame(
        with_boundary_df.groupby(
            ['mouse_id', 'group']
        ).code.nunique()
    )
    totals = totals.reset_index().rename(columns={'code':'total'})
    totals_in_out = pd.DataFrame(
        with_boundary_df.groupby(
            ['mouse_id', 'group', 'in_boundary']
        ).code.nunique()
    ).reset_index()
    if by_group:
        totals_in_out = agg.fill_mouse_id_zeroes(
            totals_in_out,
            info_cols=['group'],
            fill_col='code',
            fill_cat_col='in_boundary',
            fill_cats=[0, 1],
            fill_val=0
        )

        percent_in = totals_in_out.merge(
            totals,
            how='inner',
            validate='m:1'
        ).assign(perc_bound=lambda x: 100* x.code/x.total)
        percent_in['bound_desc'] = percent_in.in_boundary.map(
            {
                0: 'Normal',
                1: 'Lower Right'
            }
        )
        if do_stats:
            stat_tests.ind_ttest_between_groups_at_each_time(
                data=percent_in[percent_in.in_boundary == 1],
                test_col='perc_bound',
                timepoint_col='bound_desc',
                overall_context=cell_type + ' in Boundary',
                show_ns=True
            )
    else:
        totals_in_out = agg.fill_mouse_id_zeroes(
            totals_in_out,
            info_cols=['group'],
            fill_col='code',
            fill_cat_col='in_boundary',
            fill_cats=[0, 1],
            fill_val=0
        )

        percent_in = totals_in_out.merge(
            totals,
            how='inner',
            validate='m:1'
        ).assign(perc_bound=lambda x: 100* x.code/x.total)
        percent_in['bound_desc'] = percent_in.in_boundary.map(
            {
                0: 'Normal',
                False: 'Normal',
                1: 'Lower Right',
                True: 'Lower Right',
            }
        )
        print('\n',cell_type.title())
        #print(percent_in[percent_in.in_boundary==1][['mouse_id', 'bound_desc', 'perc_bound']])
        print(list(percent_in[percent_in.in_boundary==1]['perc_bound'].values))


    ticks = ticker.FuncFormatter(
        lambda x, pos: r'$10^{' + r'{0:g}'.format(x - 3) + r'}$'
    )
    ax.xaxis.set_major_formatter(ticks)
    ax.yaxis.set_major_formatter(ticks)
    if plot_boundary:
        rel_boundary = boundary[boundary['boundary'] > 0]
        rel_boundary = rel_boundary.sort_values(by='boundary')
        if plot_upper:
            rel_boundary_upper = upper_boundary[upper_boundary['boundary'] > 0]
            rel_boundary_upper = rel_boundary_upper.sort_values(by='boundary')
            ax.plot(
                rel_boundary_upper['hsc_percent_engraftment'][::1],
                rel_boundary_upper['boundary'][::1],
                c='#535c68',
                lw=2,
                ls='dashed',
            )
        ax.plot(
            rel_boundary['hsc_percent_engraftment'][::1],
            rel_boundary['boundary'][::1],
            c='#535c68',
            lw=2,
            ls='dashed',
        )
    sns.despine()
    ax.set_xlabel('HSC Abundance')
    ax.set_ylabel(cell_type.title() + ' Abundance')
    plt.title(
        cell_type.title()
        + ' sub-samples: ' + str(n_sub_samples)
        + ' sample-perc: ' + str(100 * sub_sample_amount)
        + ' alpha: ' + str(alpha)
        )
    fname = save_path + os.sep \
        + 'hsc_bootstrap_' \
        + cell_type  \
        + inv_desc \
        + '_a' + str(thresholds[cell_type]).replace('.','-') \
        + '_alpha' + str(alpha).replace('.','-') \
        + '_nregs' + str(n_sub_samples) \
        + '_sampratio' + str(sub_sample_amount).replace('.', '-') \
        + '_color-' + hue_col \
        + '.' + save_format

    save_plot(fname, save, save_format)

def plot_abundance_changed_bygroup(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        by_mouse: bool,
        merge_type: str,
        timepoint: Any = None,
        sum: bool = False,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    change_col = 'change_type'
    hue_order = ['aging_phenotype', 'no_change']

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
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False
    )
    change_marked_df = agg.mark_changed(
        clonal_abundance_df,
        bias_change_df,
        min_time_difference=mtd,
        timepoint=timepoint,
        merge_type=merge_type,
    )
    change_marked_df = agg.remove_month_17_and_6(
        change_marked_df,
        timepoint_col
    )
    fname_prefix = save_path + os.sep \
        + 'abundance_by_group_change' \
        + 't' + str(timepoint) \
        + '_mtd' + str(mtd) \
        + 'merge-' + merge_type
    desc_add = ''
    if sum and by_mouse:
        raise ValueError('Cannot set both by_mouse (average) and sum flags, pick one')

    if sum:
        change_marked_df = pd.DataFrame(
            change_marked_df.groupby([
                'mouse_id',
                'group',
                'cell_type',
                change_col,
                timepoint_col,
            ]).percent_engraftment.sum()
        ).reset_index()
        desc_add = 'sum'
    elif by_mouse:
        change_marked_df = pd.DataFrame(
            change_marked_df.groupby([
                'mouse_id',
                'group',
                'cell_type',
                change_col,
                timepoint_col,
            ]).percent_engraftment.mean()
        ).reset_index()
        desc_add = 'mean'
    
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Ranksums Test Between ' + change_col.title() + desc_add.title() + ' Abundance at each time point'
    )
    for (change_type, cell_type), c_df in change_marked_df.groupby([change_col, 'cell_type']):
        if cell_type == 'hsc':
            continue
        if (cell_type == 'b') and (by_mouse) and (change_type == 'Lymphoid'):
            file_n = os.path.join(
                save_path,
                'b_lymphoid_avg_abundance.csv'
            )
            print('Saving lymphoid clones to:', file_n)
            c_df.to_csv(
                file_n,
                index=False,
            )
        plt.figure(figsize=(7,5))
        plt.title(
            cell_type.title() + ' '
            + desc_add.title() 
            + ' ' + change_type
        )
        medianprops = dict(
            linewidth=0,
        )
        meanprops = dict(
            linestyle='solid',
            linewidth=3,
            color='black'
        )
        ax = sns.boxplot(
            x=timepoint_col,
            y='percent_engraftment',
            data=c_df,
            hue='group',
            hue_order=hue_order,
            palette=COLOR_PALETTES['group'],
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
            fliersize=0,
        )
        min_val = c_df[c_df.percent_engraftment > 0].percent_engraftment.min()
        ax.set_yscale('symlog', linthreshy=min_val*10)
        if (sum or by_mouse):
            times = c_df.sort_values(by=timepoint_col)[timepoint_col].unique()
            for mouse_id, m_df in c_df.groupby('mouse_id'):
                sns.stripplot(
                    x=timepoint_col,
                    y='percent_engraftment',
                    order=times,
                    hue='group',
                    hue_order=hue_order,
                    palette=COLOR_PALETTES['group'],
                    dodge=True,
                    ax=ax,
                    data=m_df,
                    marker=MARKERS['mouse_id'][mouse_id],
                    size=12,
                    linewidth=.8,
                    alpha=0.8,
                    zorder=0,
                )
        else:
            sns.swarmplot(
                x=timepoint_col,
                y='percent_engraftment',
                data=c_df,
                hue='group',
                hue_order=hue_order,
                palette=COLOR_PALETTES['group'],
                ax=ax,
                dodge=True,
                linewidth=0.5,
                zorder=0,
            )
        sns.despine()
        plt.xlabel(timepoint_col.title())
        plt.ylabel(y_col_to_title(cell_type+'_percent_engraftment'))
        plt.legend().remove()
        fname = fname_prefix + '_' + cell_type \
            + '_' + change_type \
            + '_' + desc_add \
            + '.' + save_format
        save_plot(fname, save, save_format)
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\n  - Change-Status: ' + change_type.replace('_', ' ').title()
            + '  Cell Type: ' + cell_type.title()
        )
        if (sum or by_mouse):
            show_ns=True
            for group, g_df in c_df.groupby('group'):
                stat_tests.one_way_ANOVArm(
                    data=g_df,
                    timepoint_col=timepoint_col,
                    id_col='mouse_id',
                    value_col='percent_engraftment',
                    overall_context=group + ' ' + cell_type.title(),
                    show_ns=show_ns,
                    match_cols=['mouse_id', 'group'],
                    merge_type='inner',
                    fill_na=0
                )
            stat_tests.ind_ttest_between_groups_at_each_time(
                data=c_df,
                test_col='percent_engraftment',
                timepoint_col=timepoint_col,
                overall_context=cell_type.title() + ' ' + change_type.title(),
                show_ns=show_ns,
            )
        else:
            for group, g_df in c_df.groupby('group'):
                stat_tests.friedman_wilcoxonSignedRank(
                    data=g_df,
                    timepoint_col=timepoint_col,
                    id_col='code',
                    value_col='percent_engraftment',
                    overall_context=group + ' ' + cell_type.title(),
                    show_ns=True,
                    match_cols=['mouse_id', 'code'],
                    merge_type='inner',
                    fill_na=0,
                )
            stat_tests.ranksums_test_group_time(
                data=c_df,
                test_col='percent_engraftment',
                timepoint_col=timepoint_col,
                overall_context=cell_type.title() + ' ' + change_type.title(),
                show_ns=True,
            )


def plot_abundance_bias_change_type_heatmap(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        merge_type: str,
        plot_average: bool,
        change_type: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ) -> pd.DataFrame:

    if change_type is None:
        change_type = 'Unchanged'

    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False
    )
    change_marked_df = agg.mark_changed(
        clonal_abundance_df,
        bias_change_df,
        min_time_difference=mtd,
        merge_type=merge_type,
    )
    bias_with_cats = agg.add_bias_category(
        lineage_bias_df
    )
    change_marked_df = agg.remove_month_17_and_6(
        change_marked_df,
        timepoint_col
    )
    only_n_timepoint_mice =agg.filter_mice_with_n_timepoints(
        change_marked_df,
        n_timepoints=4,
    )
    bias_cat_abundance_df = only_n_timepoint_mice.merge(
        bias_with_cats[['code', 'mouse_id', 'bias_category', timepoint_col]].drop_duplicates(),
        how='inner',
        validate='m:1'
    )
    only_change_type_df = bias_cat_abundance_df[bias_cat_abundance_df.change_status == change_type]

    only_change_type_df = agg.add_short_group(
        only_change_type_df
    )
    order =[
        ('LB', 'E'),
        ('LB', 'D'),
        ('B', 'E'),
        ('B', 'D'),
        ('MB', 'E'),
        ('MB', 'D'),
        ]

    y_col = 'percent_engraftment'
    fname_prefix = save_path + os.sep \
        + 'abundance_bias_change_type_heatmap' \
        + '_mtd' + str(mtd) \
        + 'merge-' + merge_type

    if plot_average:
        y_col = 'avg_abundance'
        only_change_type_df = agg.add_avg_abundance_until_timepoint_clonal_abundance_df(
            only_change_type_df,
            timepoint_col
        )
    for cell_type, c_df in only_change_type_df.groupby('cell_type'):
        pivotted = c_df.pivot_table(
            values=y_col,
            index=['bias_category', 'group_short'],
            columns=timepoint_col,
            aggfunc=np.mean,
            fill_value=0,
        )
        fig, ax = plt.subplots()
        sns.heatmap(
            pivotted.reindex(order),
            ax=ax
        )
        plt.title(cell_type + ' ' + change_type)


        fname = fname_prefix + '_' + cell_type \
            + '_' + change_type \
            + '_' + y_col \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_balanced_at_second_to_last(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ) -> pd.DataFrame:

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
    if timepoint_col == 'month':
        s2l_time = 12
    elif timepoint_col == 'gen':
        s2l_time = 7
    else:
        s2l_time = clonal_abundance_df.sort_values(
                by=[timepoint_col],
                ascending=False,
            )[timepoint_col].unique()[1]

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    lineage_bias_df = agg.remove_month_17_and_6(
        lineage_bias_df,
        timepoint_col
    )
    with_bias_cats_df = agg.add_bias_category(lineage_bias_df)
    survival_df = agg.label_exhausted_clones(
        with_bias_cats_df,
        clonal_abundance_df,
        timepoint_col
    )
    s2l_df = agg.get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        timepoint=s2l_time,
        by_mouse=False,
    )
    survival_abundance = s2l_df.merge(
        survival_df[['bias_category', 'mouse_id', 'code', 'survived', timepoint_col]].drop_duplicates(),
        how='inner',
        validate='m:1'
    )
    balanced_abundance_df = survival_abundance[survival_abundance.bias_category == 'B']
    print(balanced_abundance_df.bias_category.unique())

    # Isolate clones at second to last time point
    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    hue_col = 'group'
    hue_order = ['aging_phenotype', 'no_change']
    palette = COLOR_PALETTES['group']
    y_col = 'percent_engraftment'
    dodge=True
    filename_addon = 'by_group'
    for cell_type, c_df in balanced_abundance_df.groupby('cell_type'):
        _, ax = plt.subplots(figsize=(7,5)) 
        ax.set_yscale('symlog', linthreshy=10e-3)
        sns.boxplot(
            x='survived',
            y=y_col,
            order=['Exhausted', 'Survived'],
            data=c_df,
            hue=hue_col,
            hue_order=hue_order,
            dodge=dodge,
            showbox=False,
            whiskerprops={
                "alpha": 0
            },
            ax=ax,
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
            fliersize=0,
        )
        sns.swarmplot(
            x='survived',
            y=y_col,
            order=['Exhausted', 'Survived'],
            data=c_df,
            hue=hue_col,
            hue_order=hue_order,
            palette=palette,
            linewidth=.5,
            ax=ax,
            dodge=dodge,
            zorder=0
        )
        sns.despine()
        plt.legend().remove()
        fname = save_path + os.sep + 'exhaust_persist_cell_abund' \
            + '_' + cell_type \
            + '_' + filename_addon \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_hsc_abund_bias_at_last_change(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_group: bool,
        mtd: int,
        merge_type: str,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.minor.width': 5,
            'ytick.minor.size': 5,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }

        )
    if clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].empty:
        raise ValueError('No HSC Cells in Clonal Abundance Data')
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False
    )
    change_marked_df = agg.mark_changed(
        lineage_bias_df,
        bias_change_df,
        min_time_difference=mtd,
        merge_type=merge_type,
    )

    last_clones = agg.find_last_clones(
        change_marked_df,
        timepoint_col
    )
    labeled_last_clones = agg.add_bias_category(
        last_clones
    )
    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']
    
    myeloid_hsc_abundance_df = hsc_data.merge(
        labeled_last_clones[['code','mouse_id','bias_category', 'change_status']],
        on=['code','mouse_id'],
        how='inner',
        validate='m:m'
    )
    unique_cats=['LB', 'B', 'MB']
    sems={}
    means={}
    colors={}
    coords = np.arange(len(unique_cats)) + 1
    width = 0.4
    _, ax = plt.subplots(figsize=(6, 5))
    i = -1

    for change_status, c_df in myeloid_hsc_abundance_df.groupby('change_status'):
        means[change_status] = []
        sems[change_status] = [[],[]]
        colors[change_status] = []
        for bias_cat in unique_cats:
            cats_df = c_df[c_df.bias_category == bias_cat]
            sems[change_status][0].append(0)
            sems[change_status][1].append(cats_df.percent_engraftment.sem())
            means[change_status].append(cats_df.percent_engraftment.mean())
            colors[change_status].append(COLOR_PALETTES['change_status'][change_status])
    
        ax.bar(
            x=coords + (i*width/2),
            height=means[change_status],
            width=width,
            tick_label=unique_cats,
            color=colors[change_status],
            log=True,
        )
        _, caps, _ = ax.errorbar(
            coords + (i * width/2),
            means[change_status],
            yerr=sems[change_status],
            color='black',
            capsize=10,
            capthick=2,
            ls='none',
            )
        i = i * -1
        caps[0].set_marker('_')
        caps[0].set_markersize(0)
    stat_tests.ranksums_test_group_time(
        data=myeloid_hsc_abundance_df,
        test_col='percent_engraftment',
        timepoint_col='bias_category',
        overall_context='HSC Abundance',
        show_ns=True,
        group_col='change_status',
    )
    plt.xlabel('')
    plt.ylabel('HSC Abundance (% HSCs)')
    plt.title(
        'HSC Abundance by Bias at Last Time Point'
        )

    sns.despine()
    fname = save_path + os.sep \
        + 'abund_hsc_biased_at_last_change' \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_balanced_clone_abundance(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        group: str,
        save: bool,
        save_path: str,
        save_format: str='png',
) -> None:
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 2,
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
    save_labels=True
    if group != 'all':
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.group == group]
        lineage_bias_df = lineage_bias_df[lineage_bias_df.group == group]

    clonal_abundance_df = agg.remove_month_17(
        clonal_abundance_df,
        timepoint_col
    )
    lineage_bias_df = agg.remove_month_17_and_6(
        lineage_bias_df,
        timepoint_col
    )
    with_bias_cats = agg.add_bias_category(lineage_bias_df)
    balanced_clones = with_bias_cats[with_bias_cats.bias_category == 'B']
    balanced_at_dfs = []
    if save_labels:
        last_clones = agg.get_clones_at_timepoint(
            with_bias_cats,
            timepoint_col,
            'last',
            by_mouse=True
        )
        labels_df = last_clones[['mouse_id', 'group','bias_category', 'code']].drop_duplicates()
        labels_df.rename(columns={'bias_category': 'label_name'}).to_csv(
            os.path.join(
                save_path,
                'MOLD_Bias_labels.csv'
            ),
            index=False
        )
    for time, t_df in balanced_clones.groupby(timepoint_col):
        filt_t_df = agg.filter_lineage_bias_thresholds(
            t_df,
            thresholds,
        )
        balanced_at_time_df = clonal_abundance_df.merge(
            filt_t_df[['code', 'mouse_id']].drop_duplicates(),
            how='inner',
            validate='m:1',
        )
        balanced_at_time_df['balance_time'] = time
        balanced_at_dfs.append(balanced_at_time_df)
    balanced_df = pd.concat(balanced_at_dfs).sort_values(by=timepoint_col)

    for (cell, balance_time), g_df in balanced_df.groupby(['cell_type', 'balance_time']):
        if cell == 'hsc' :
            continue
        _, ax = plt.subplots(figsize=(4,4))
        if cell == 'gr':
            ax.set_yscale('symlog', linthreshy=10e-4)
        else:
            ax.set_yscale('symlog', linthreshy=10e-3)
        plt.title(group + ' ' + cell + ' ' + str(balance_time))
        sns.lineplot(
            x=timepoint_col,
            y='percent_engraftment',
            hue='mouse_id',
            units='code',
            estimator=None,
            palette=COLOR_PALETTES['mouse_id'],
            data=g_df,
            ax=ax
        )
        ax.legend().remove()
        sns.despine()
        thresh_str = '_'.join(
            [str(key) + '-' + str(round(value,2)) for key, value in thresholds.items()]
        )
        fname = save_path + os.sep + 'balanced_clone_abundance' \
            + '_' + group + '_' + cell + '_' + str(balance_time) \
            + '_' + thresh_str \
            + '.' + save_format
        save_plot(fname, save, save_format)    

def plot_lymphoid_committed_vs_bias_hsc(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        max_myeloid_abundance: float,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 4,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.minor.width': 5,
            'ytick.minor.size': 5,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }

        )
    if clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].empty:
        raise ValueError('No HSC Cells in Clonal Abundance Data')

    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']
    labeled_bias = agg.label_lymphoid_comitted(lineage_bias_df, max_myeloid_abundance)
    labeled_hsc_abundance = hsc_data.merge(
        labeled_bias[
            ['code', 'mouse_id', timepoint_col, 'bias_category']
        ].drop_duplicates(),
        how='inner',
        validate='1:1'
    )
    sems=[[],[]]
    means=[]
    colors=[]
    unique_cats=['LC', 'LB', 'B', 'MB']
    for bias_cat in unique_cats:
        cats_df = labeled_hsc_abundance[labeled_hsc_abundance.bias_category == bias_cat]
        sems[0].append(0)
        sems[1].append(cats_df.percent_engraftment.sem())
        means.append(cats_df.percent_engraftment.mean())
        colors.append(COLOR_PALETTES['bias_category'][bias_cat])
    
    coords = np.arange(len(unique_cats)) + 1
    width = 0.8
    _, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        x=coords,
        height=means,
        width=width,
        tick_label=unique_cats,
        color=colors,
        log=True,
    )
    _, caps, _ = ax.errorbar(
        coords,
        means,
        yerr=sems,
        color='black',
        capsize=10,
        capthick=3,
        ls='none',
        )
    caps[0].set_marker('_')
    caps[0].set_markersize(0)
    plt.xlabel('')
    plt.ylabel('HSC Abundance (% HSCs)')
    plt.title(
        'HSC Abundance by Bias at Last Time Point' 
        )

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Ranksums Test of HSC Abundance Between Bias Types\n'
        + 'N-Mice: ' + str(labeled_hsc_abundance.mouse_id.nunique())
    )
    stat_tests.ranksums_test_group(
        labeled_hsc_abundance,
        'percent_engraftment',
        'HSC Abundance',
        show_ns=True,
        group_col='bias_category'
    )

    sns.despine()
    fname = save_path + os.sep \
        + 'abund_hsc_biased_at_last_lymphoid_comitted' \
        + 'max_myel_' + str(max_myeloid_abundance).replace('.', '-') \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_abundance_clones_per_mouse(
    clonal_abundance_df: pd.DataFrame,
    timepoint_col: str,
    cell_type: str,
    thresholds: Dict[str, float],
    by_group: bool,
    save: bool,
    save_path: str,
    save_format: str = 'png',
    ) -> None:
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 4,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )

    if by_group:
        group_col = 'group'
    else:
        raise ValueError('Not implemented plotting if its not by group.\n\tTry re-running with both --by-group and --by-mouse flags')

    filtered_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    filtered_df = agg.filter_cell_type_threshold(
        filtered_df,
        thresholds,
        analyzed_cell_types=[cell_type]
    )

    nunique_time = filtered_df[timepoint_col].nunique()
    filtered_df =agg.filter_mice_with_n_timepoints(
        filtered_df,
        nunique_time
    )
    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.despine()
    plt.yscale('log')
    if cell_type == 'gr':
        ax.set_ylim([30e-4,1])
    elif cell_type == 'b':
        ax.set_ylim([30e-3,1])
    
    agg_df = pd.DataFrame(
        filtered_df.groupby(
            ['mouse_id', group_col, timepoint_col]
        ).percent_engraftment.mean()
    ).reset_index()
    times = agg_df.sort_values(by=timepoint_col)[timepoint_col].unique()
    print(agg_df.groupby('group').mouse_id.unique())
    for mouse_id, m_df in agg_df.groupby('mouse_id'):
        sns.stripplot(
            x=timepoint_col,
            order=times,
            y='percent_engraftment',
            hue=group_col,
            hue_order=['aging_phenotype', 'no_change'],
            palette=COLOR_PALETTES[group_col],
            dodge=True,
            ax=ax,
            data=m_df,
            marker=MARKERS['mouse_id'][mouse_id],
            size=12,
            linewidth=.8,
            alpha=0.8,
            zorder=0,
        )
    sns.boxplot(
        x=timepoint_col,
        y='percent_engraftment',
        hue=group_col,
        hue_order=['aging_phenotype', 'no_change'],
        data=agg_df,
        ax=ax,
        fliersize=0,
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
    )
    ax.set_title(cell_type)
    ax.legend().remove()
    for group, g_df in agg_df.groupby('group'):
        stat_tests.one_way_ANOVArm(
            data=g_df,
            timepoint_col=timepoint_col,
            id_col='mouse_id',
            value_col='percent_engraftment',
            overall_context=cell_type + ' ' + group,
            show_ns=True,
            match_cols=['mouse_id', 'group'],
            merge_type='inner',
            fill_na=0
        )
    stat_tests.ind_ttest_between_groups_at_each_time(
        agg_df,
        'percent_engraftment',
        timepoint_col,
        overall_context=cell_type,
        show_ns=True,
        group_col=group_col,
    )

    fname = save_path + os.sep \
        + 'swarmplot_abundance_by-mouse' \
        + '_' + cell_type \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_percent_balanced_expanded(
    clonal_abundance_df: pd.DataFrame,
    lineage_bias_df: pd.DataFrame,
    timepoint_col: str,
    thresholds: Dict[str, float],
    by_group: bool,
    bias_cat: str,
    save: bool,
    save_path: str,
    save_format: str = 'png',
    ) -> None:
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 4,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )

    filt_time_abundance = agg.remove_month_17(clonal_abundance_df, timepoint_col)
    filt_time_bias = agg.remove_month_17(lineage_bias_df, timepoint_col)

    first = agg.get_clones_at_timepoint(
        filt_time_abundance,
        timepoint_col,
        'first',
        by_mouse=True
    )
    last = agg.get_clones_at_timepoint(
        filt_time_abundance,
        timepoint_col,
        'last',
        by_mouse=True
    )
    first['time_desc'] = 'first'
    last['time_desc'] = 'last'
    first_last = pd.concat([first, last])
    mouse_time = first_last[['mouse_id', 'time_desc', 'group']].drop_duplicates()
    expanded_clones = agg.filter_cell_type_threshold(
        first_last,
        thresholds,
        analyzed_cell_types=thresholds.keys(),
    )
    bias_cat_df = agg.add_bias_category(lineage_bias_df)
    expanded_bias_df = bias_cat_df.merge(
        expanded_clones[['code', 'mouse_id', 'time_desc', timepoint_col]].drop_duplicates(),
        how='inner',
        validate='1:1'
    )
    bias_counts = pd.DataFrame(
        expanded_bias_df \
            .groupby(['mouse_id', 'time_desc', 'group', 'bias_category']) \
            .code.nunique()
    ).reset_index().rename(
        columns={'code': '# of Expanded Clones'}
    )
    totals = pd.DataFrame(
        bias_counts.groupby(['mouse_id', 'time_desc'])\
            ['# of Expanded Clones'].sum()
    ).reset_index().rename(
        columns={'# of Expanded Clones': 'total'}
    )
    bias_counts_total = bias_counts.merge(
        totals,
        how='inner',
        validate='m:1'
    )
    bias_counts_total['Percent Expanded Clones'] = \
        100*bias_counts_total['# of Expanded Clones']/bias_counts_total['total']

    _, ax = plt.subplots(figsize=(8,8))
    # Uncomment to look at just balanced clones
    balanced_percent = bias_counts_total[bias_counts_total.bias_category == bias_cat]
    balanced_percent_zero = balanced_percent.merge(
        mouse_time,
        how='outer',
        validate='1:1',
    ).fillna(0)
    stripplot_mouse_markers_with_mean(
        balanced_percent_zero,
        'time_desc',
        'Percent Expanded Clones',
        ax,
        'group',
        ['aging_phenotype', 'no_change']
    )
    stat_tests.ind_ttest_between_groups_at_each_time(
        balanced_percent_zero,
        'Percent Expanded Clones',
        'time_desc',
        overall_context=bias_cat,
        show_ns=True,
        group_col='group',
    )
    stat_tests.rel_ttest_group_time(
        balanced_percent_zero,
        match_cols=['mouse_id'],
        merge_type='outer',
        fill_na=0,
        test_col='Percent Expanded Clones',
        timepoint_col='time_desc',
        overall_context=bias_cat,
        show_ns=True
    )

    fname = save_path + os.sep \
        + 'percent_expanded_by-group_by-mouse' \
        + '_' + bias_cat \
        + '.' + save_format
    save_plot(fname, save, save_format)
    
def stripplot_mouse_markers_with_mean(
    agg_df: pd.DataFrame,
    timepoint_col: str,
    y_col: str,
    ax,
    hue_col: str = 'group',
    hue_order: List[str] = ['aging_phenotype', 'no_change'],
    order: List = None,
    palette: Any = None,
    size: int = 12,
    ):

    if order:
        times = order
    else:
        times = agg_df.sort_values(by=timepoint_col)[timepoint_col].unique()
    if palette is None:
        palette = COLOR_PALETTES[hue_col]
    for mouse_id, m_df in agg_df.groupby('mouse_id'):
        sns.stripplot(
            x=timepoint_col,
            order=times,
            y=y_col,
            hue=hue_col,
            hue_order=hue_order,
            palette=palette,
            dodge=True,
            ax=ax,
            data=m_df,
            marker=MARKERS['mouse_id'][mouse_id],
            size=size,
            linewidth=.8,
            alpha=0.8,
            zorder=0,
        )

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )

    sns.boxplot(
        x=timepoint_col,
        y=y_col,
        order=times,
        hue=hue_col,
        hue_order=hue_order,
        data=agg_df,
        ax=ax,
        fliersize=0,
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
    )
    sns.despine()
    ax.legend().remove()

def plot_count_biased_changing_at_time(
    lineage_bias_df: pd.DataFrame,
    timepoint_col: str,
    by_group: bool,
    mtd: int,
    timepoint: int,
    save: bool,
    save_path: str,
    save_format: str = 'png',
    ):

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 4,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )
    filt_df = agg.remove_month_17_and_6(
        lineage_bias_df,
        timepoint_col
    )


    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False
    )
    change_marked_df = agg.mark_changed(
        lineage_bias_df,
        bias_change_df,
        min_time_difference=mtd,
        merge_type='inner',
    )
    with_bias_cat = agg.add_bias_category(change_marked_df)

    total_df = pd.DataFrame(with_bias_cat.groupby(
        ['mouse_id', 'group', timepoint_col]
        ).code.nunique()
    ).reset_index().rename(columns={
        'code': 'total_clones'
    })
    count_df = pd.DataFrame(with_bias_cat.groupby(
        ['mouse_id', 'group', timepoint_col, 'bias_category_short', 'change_type']
        ).code.nunique()
    ).reset_index().rename(columns={
        'code': '# of Clones'
    })
    count_df = count_df.merge(total_df)
    count_df['percent_clones'] = count_df['# of Clones'] / count_df['total_clones']
    count_at_time = count_df[count_df[timepoint_col] == timepoint]
    _, axes = plt.subplots(ncols=2, figsize=(16,8))
    cats = ['Lymphoid', 'Myeloid']
    print(timepoint_col, timepoint)
    for i in range(2):
        cat = cats[i]
        ax = axes[i]
        cat_df = count_at_time[count_at_time.change_type == cat]
        cat_zero_df = agg.fill_mouse_id_zeroes(
            cat_df,
            info_cols=['group'],
            fill_col='percent_clones',
            fill_cat_col='bias_category_short',
            fill_cats=['Balanced', 'Ly Biased', 'My Biased'],
            fill_val=0,
        )
        stripplot_mouse_markers_with_mean(
            cat_zero_df,
            'bias_category_short',
            'percent_clones',
            ax,
            'group',
            ['aging_phenotype', 'no_change']
        )
        ax.set_title(cat)
        stat_tests.ind_ttest_between_groups_at_each_time(
            data=cat_zero_df,
            test_col='percent_clones',
            timepoint_col='bias_category_short',
            overall_context=cat,
            show_ns=True,
            group_col='group'
        )
    fname = save_path + os.sep \
        + 'count_biased_per_mouse' \
        + '_' + timepoint_col[0] + str(timepoint) \
        + '.' + save_format
    save_plot(fname, save, save_format)
    print('Location')

def cell_type_expanded_hsc_vs_group(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        cell_type: str,
        timepoint: int,
        save: bool,
        save_path: str,
        by_mouse: bool = False,
        by_count: bool = False,
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 4,
            'axes.linewidth': 4,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )
    filt_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    timepoint_df = agg.get_clones_at_timepoint(
        filt_df,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )
    expanded_at_time_df = agg.filter_threshold(
        timepoint_df,
        thresholds[cell_type],
        [cell_type],
    )
    if expanded_at_time_df.cell_type.nunique() != 1:
        print(expanded_at_time_df.cell_type.unique())
        raise ValueError('Too many cell types found')
    if expanded_at_time_df[timepoint_col].nunique() != 1:
        print('Timepoints:', expanded_at_time_df[timepoint_col].unique())

    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].rename(
        columns={'percent_engraftment': 'hsc_abundance'}
    )

    expanded_time_clones = expanded_at_time_df[['mouse_id', 'group', 'code']].drop_duplicates()

    expanded_time_hsc = hsc_data[['mouse_id', 'group', 'code', 'hsc_abundance']].merge(
        expanded_time_clones,
        how='inner',
        validate='1:1'
    )
    _, ax = plt.subplots(figsize=(6,5))
    if by_mouse:
        hsc_abundance_per_mouse = pd.DataFrame(
            expanded_time_hsc.groupby(['mouse_id', 'group'])\
                .hsc_abundance\
                .sum()
        ).reset_index()
        mice_with_data = hsc_data[['mouse_id', 'group']].drop_duplicates()
        hsc_abundance_df = hsc_abundance_per_mouse.merge(
            mice_with_data,
            how='outer',
            validate='1:1'
        ).fillna(0)
        hsc_abundance_df['time'] = timepoint
        stripplot_mouse_markers_with_mean(
            hsc_abundance_df,
            'time',
            'hsc_abundance',
            ax,
            'group',
            ['aging_phenotype', 'no_change']
        )
        ax.set_ylabel('Sum HSC Abundance')
        stat_tests.ind_ttest_between_groups_at_each_time(
            data=hsc_abundance_df,
            test_col='hsc_abundance',
            timepoint_col='time',
            overall_context='mean hsc abundance'.title(),
            show_ns=True,
        )
        desc='by_mouse'
    elif by_count:
        filt_hsc_df = expanded_time_hsc[expanded_time_hsc.hsc_abundance > 0.01]
        hsc_abundance_per_mouse = pd.DataFrame(
            expanded_time_hsc.groupby(['mouse_id', 'group'])\
                .code\
                .nunique()
        ).reset_index()
        mice_with_data = hsc_data[['mouse_id', 'group']].drop_duplicates()
        hsc_abundance_df = hsc_abundance_per_mouse.merge(
            mice_with_data,
            how='outer',
            validate='1:1'
        ).fillna(0)
        hsc_abundance_df['time'] = timepoint
        stripplot_mouse_markers_with_mean(
            hsc_abundance_df,
            'time',
            'code',
            ax,
            'group',
            ['aging_phenotype', 'no_change']
        )
        ax.set_ylabel('Unique Clone Count')
        stat_tests.ind_ttest_between_groups_at_each_time(
            data=hsc_abundance_df,
            test_col='code',
            timepoint_col='time',
            overall_context='expanded ' + cell_type + ' HSC clone count'.title(),
            show_ns=True,
        )
        desc='by_count'

    else:
        hsc_abundance_df = expanded_time_hsc
        hsc_abundance_df['time'] = timepoint
        ax.set_yscale('symlog', linthresh_y=10E-3)
        swarmplot_with_mean(
            hsc_abundance_df,
            'time',
            'hsc_abundance',
            ax,
        )
        desc='by_clone'
        stat_tests.ranksums_test_group_time(
            data=hsc_abundance_df,
            test_col='hsc_abundance',
            timepoint_col='time',
            overall_context=cell_type,
            show_ns=True
        )

    sns.despine()
    ax.set_title(cell_type, fontsize=15)
    fname = save_path + os.sep \
        + cell_type \
        + '_' + timepoint_col[0] + str(timepoint)  \
        + '_expanded_hsc_vs_group' \
        + '_' + desc \
        + '.' + save_format
    save_plot(fname, save, save_format)

def swarmplot_with_mean(
    clonal_abundance_df: pd.DataFrame,
    timepoint_col: str,
    y_col: str,
    ax,
    hue_col: str = 'group',
    hue_order: List[str] = ['aging_phenotype', 'no_change'],
    ):

    order = clonal_abundance_df[timepoint_col].unique()
    sns.swarmplot(
        x=timepoint_col,
        y=y_col,
        order=order,
        hue=hue_col,
        hue_order=hue_order,
        palette=COLOR_PALETTES[hue_col],
        dodge=True,
        ax=ax,
        data=clonal_abundance_df,
    )

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )

    sns.boxplot(
        x=timepoint_col,
        y=y_col,
        order=order,
        hue=hue_col,
        hue_order=hue_order,
        data=clonal_abundance_df,
        ax=ax,
        fliersize=0,
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
    )
    sns.despine()
    ax.legend().remove()

def plot_clone_count_swarm_mean_first_last(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        thresholds: Dict[str, float],
        abundance_cutoff: float,
        analyzed_cell_types: List[str],
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    save_labels = False
    threshold_df = agg.filter_cell_type_threshold(
        clonal_abundance_df,
        thresholds, 
        analyzed_cell_types
    )
        
    clone_counts = agg.count_clones(threshold_df, timepoint_col)
    clone_counts = agg.filter_first_last_by_mouse(
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

    if save_labels:
        last_df = agg.get_clones_at_timepoint(
            clonal_abundance_df,
            timepoint_col,
            timepoint='last',
            by_mouse=True
        )
        thresh_last = agg.filter_cell_type_threshold(
            last_df,
            thresholds,
            analyzed_cell_types
        ).assign(is_expanded='Expanded')
        label_cols = ['code', 'group', 'cell_type', 'mouse_id', 'is_expanded']
        labels_df = last_df[label_cols[:-1]].drop_duplicates().merge(
            thresh_last[label_cols].drop_duplicates(),
            how='outer',
            validate='1:1'
        )
        labels_df.loc[labels_df.is_expanded.isna(), 'is_expanded'] = 'not_expanded'

        label_path = os.path.join(
            save_path,
            'expanded_first_last_labels.csv'
        )
        print(Fore.YELLOW + 'Saving labels to: ' + label_path)
        labels_df.to_csv(label_path, index=False)
    for cell_type, c_df in clone_counts.groupby(['cell_type']):
        filled_counts = agg.fill_mouse_id_zeroes(
            c_df,
            ['group'],
            fill_col='code',
            fill_cat_col='mouse_time_desc',
            fill_cats=['First', 'Last'],
            fill_val=0
        )
        _, ax = plt.subplots(figsize=(6,5))
        stripplot_mouse_markers_with_mean(
            filled_counts,
            'mouse_time_desc',
            'code',
            ax,
            'group',
            ['aging_phenotype', 'no_change']
        )
        if abundance_cutoff == 50:
            ax.set_title(
                cell_type.title() 
                + ' Expanded Clones'
            )
        else:
            ax.set_title(
                cell_type.title() 
                + ' Clone Counts with Abundance Cutoff ' 
                + str(100 - abundance_cutoff)
            )
        ax.set_xlabel(timepoint_col.title())
        ax.set_ylabel('Number of Clones')
        print(
            Fore.CYAN + Style.BRIGHT 
            + '\nPerforming Independent T-Test on ' 
            + ' ' + cell_type.title() + ' Clone Clone Counts Between Groups'
        )
        for t1, t_df in filled_counts.groupby('mouse_time_desc'):
            _, p_value = stats.ttest_ind(
                t_df[t_df.group == 'aging_phenotype']['code'],
                t_df[t_df.group == 'no_change']['code'],
            )
            context: str = timepoint_col.title() + ' ' + str(t1)\
                + 'E-MOLD Mice: ' + str(t_df[t_df.group == 'aging_phenotype'].mouse_id.nunique())\
                + ', D-MOLD Mice: ' + str(t_df[t_df.group == 'no_change'].mouse_id.nunique())
            stat_tests.print_p_value(context, p_value, show_ns=True)


        fname = save_path + os.sep \
            + 'clone_count_' + cell_type \
            + '_a' + str(abundance_cutoff).replace('.', '-') \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_compare_change_contrib(
        changed_marked_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: Any,
        bar_col: str,
        bar_types: List,
        gfp_donor: pd.DataFrame,
        gfp_donor_thresh: float,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
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

    if timepoint_col == 'gen':
        changed_marked_df = changed_marked_df[changed_marked_df.gen != 8.5]
    percent_of_total = True

    filt_bar_types = changed_marked_df[changed_marked_df[bar_col].isin(bar_types)]
    if filt_bar_types.shape != changed_marked_df.shape:
        print("Not all types being displayed, missing:")
        print(
            changed_marked_df[~changed_marked_df[bar_col].isin(bar_types)][bar_col].unique()
        )

    percent_of_total = True
    changed_sum_df = agg.sum_abundance_by_change(
        agg.remove_month_17(
            filt_bar_types,
            timepoint_col
        ),
        timepoint_col=timepoint_col,
        percent_of_total=percent_of_total,
        change_col=bar_col,
    )
    changed_sum_df = changed_sum_df[changed_sum_df.cell_type.isin(['gr', 'b'])]

    changed_sum_with_gxd = changed_sum_df.merge(
        gfp_donor,
        how='inner',
        validate='m:1',
        on=['mouse_id', 'cell_type', timepoint_col]
    )
    timepoint_df = agg.get_clones_at_timepoint(
        changed_sum_with_gxd,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )

    order = ['Unchanged', 'Lymphoid', 'Myeloid']
    print('FILTERING FOR MICE FOUND IN FIRST AND LAST TIMEPOINT ABOVE GFP x DONOR THRESHOLD' )
    mice_left = agg.filter_gxd_first_last(
        changed_sum_with_gxd,
        timepoint_col,
        gfp_donor_thresh
    ) 
    

    if (bar_col == 'survived') and ('M2012' in mice_left):
        print('Excluding M2012')
        mice_left = [m for m in mice_left if m != 'M2012']
    print(mice_left)
    filt_gxd = timepoint_df[timepoint_df.mouse_id.isin(mice_left)]
    print('Pre-filt gxd mice:', timepoint_df.mouse_id.nunique())
    print('Post-filt gxd mice:', filt_gxd.mouse_id.nunique())

    hue_order = ['aging_phenotype', 'no_change']
    for ct, c_df in filt_gxd.groupby('cell_type'):
        fig, ax = plt.subplots(figsize=(8,8))
        filled = agg.fill_mouse_id_zeroes(
            c_df,
            info_cols=['cell_type', 'group'],
            fill_col='percent_engraftment',
            fill_cat_col=bar_col,
            fill_cats=bar_types,
            fill_val=0,
        )
        sns.barplot(
            data=filled,
            y='percent_engraftment',
            x=bar_col,
            order=bar_types,
            hue_order=hue_order,
            capsize=.1,
            errwidth=3,
            hue='group',
            zorder=10,
            palette=COLOR_PALETTES['group'],
            ax=ax,
        )
        sns.barplot(
            data=filled,
            y='percent_engraftment',
            x=bar_col,
            order=bar_types,
            hue='group',
            hue_order=hue_order,
            palette=COLOR_PALETTES['group'],
            ci=None,
            zorder=1,
            ax=ax,
        )
        ax.legend().remove()
        sns.despine()
        ax.set_title(ct.title() + ' ' + str(timepoint))
        ax.set_ylabel('Contribution (%)')
        ax.set_xlabel('')
        for k, spine in ax.spines.items():  #ax.spines is a dictionary
            spine.set_zorder(100)
        for g, g_df in filled.groupby('group'):
            stat_tests.one_way_ANOVArm(
                data=g_df,
                timepoint_col=bar_col,
                id_col='mouse_id',
                value_col='percent_engraftment',
                overall_context=g + ' ' + ct + ' at ' + str(timepoint),
                show_ns=True,
                match_cols=['mouse_id', 'group'],
                merge_type='outer',
                fill_na=0,
            )
        stat_tests.ind_ttest_between_groups_at_each_time(
            data=filled,
            test_col='percent_engraftment',
            timepoint_col=bar_col,
            overall_context=ct + ' at ' + str(timepoint),
            show_ns=True,
            group_col='group'
        )

        fname = save_path + os.sep + 'compare_contrib' \
            + '_' + bar_col \
            + '_' + ct \
            + '_gxd-' + str(gfp_donor_thresh) \
            + '_' + timepoint_col[0] + '-' + str(timepoint) \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_survival_line(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'medium',
        }

        )

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        False,
    )
    survival_df = agg.label_exhausted_clones(
        None,
        clonal_abundance_df,
        timepoint_col
    )
    survival_df = survival_df[survival_df.survived.isin([
        'Exhausted',
        'Survived'
    ])]
    if by_group:
        hue = 'group'
    else:
        hue = 'mouse_id'

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16,16))
    plt.subplots_adjust(hspace=.5)
    i=0
    for (s, ct), s_df in survival_df.groupby(['survived', 'cell_type']):
        if ct not in ['gr', 'b']:
            continue
        ax = axes.flatten()[i]
        i += 1
        if ct == 'gr':
            ax.set_yscale('symlog', linthreshy=10E-4)
        else:
            ax.set_yscale('symlog', linthreshy=10E-3)

        sns.lineplot(
            data=s_df,
            x=timepoint_col,
            y='percent_engraftment',
            estimator=None,
            units='code',
            hue=hue,
            palette=COLOR_PALETTES[hue],
            ax=ax,
            alpha=0.3
        )
        sns.lineplot(
            data=s_df,
            x=timepoint_col,
            y='percent_engraftment',
            hue=hue,
            palette=['white','white'],
            ax=ax,
            ci=None,
            lw=10,
        )
        sns.lineplot(
            data=s_df,
            x=timepoint_col,
            y='percent_engraftment',
            hue=hue,
            palette=COLOR_PALETTES[hue],
            ax=ax,
            ci=None,
            lw=6,
        )
        ax.legend().remove()
        sns.despine()
        ax.set_title(s + ' ' + ct)
    fname = os.path.join(
        save_path,
        'survival_line_abundance_'
            + hue
            + '.' + save_format
    ) 
    save_plot(fname, save, save_format)

def plot_count_by_change(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        by_group: bool,
        by_count: bool,
        hscs: bool,
        timepoint: Any,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 5,
            'axes.linewidth': 3,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.major.size': 4,
            'ytick.minor.size': 4,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'medium',
        }
        )
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False,
    )
    clonal_abundance_df = agg.remove_month_17(
        clonal_abundance_df,
        timepoint_col 
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        keep_hsc=False
    )
    change_marked_df = agg.mark_changed(
        clonal_abundance_df,
        bias_change_df,
        min_time_difference=mtd
    )
    if hscs:
        tp_data = change_marked_df[change_marked_df.cell_type == 'hsc']
        tp_data = tp_data[tp_data.percent_engraftment > 0]
    else:
        tp_data = agg.get_clones_at_timepoint(
            change_marked_df,
            timepoint_col,
            timepoint,
            by_mouse=False,
        )
        tp_data = tp_data[tp_data.percent_engraftment > 0.01]
    y_col = 'code'
    total = pd.DataFrame(
        tp_data.groupby('mouse_id').code.nunique()
    ).reset_index().rename(columns={'code':'total'})
    count_df = pd.DataFrame(
        tp_data.groupby([
            'mouse_id',
            'group',
            'change_type',
        ]).code.nunique()
    )[y_col].reset_index().merge(
        total
    )
    count_df['perc_unique'] = 100 * count_df[y_col]/count_df['total']
    y_col = 'perc_unique'
    desc_add = 'perc_unique'
    y_desc = '% Unique Clones'
    if by_count:
        y_col = 'code'
        desc_add='num_clones'
        y_desc='# of Unique Clones'
    if hscs:
        desc_add += '_hsc'
        y_desc+=' HSCs'

    count_df = agg.fill_mouse_id_zeroes(
        count_df,
        ['group'],
        fill_col=y_col,
        fill_cat_col='change_type',
        fill_cats=change_marked_df.change_type.unique(),
        fill_val=0,
    )
    print('SAVING DATA TO:', os.path.join(save_path, 'blood_by_change.csv'))
    count_df.to_csv(os.path.join(save_path, 'blood_by_change.csv'), index=False)
    mouse_markers = True

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Tests On: '+ desc_add.title() 
    )
    show_ns = True
    #match_cols = ['mouse_id']
    for group, g_df in count_df.groupby('group'):
        stat_tests.anova_oneway(
            data=g_df,
            category_col='change_type',
            value_col=y_col,
            overall_context=group + ' ' + y_desc,
            show_ns=show_ns
        )
    stat_tests.ind_ttest_between_groups_at_each_time(
        count_df,
        test_col=y_col,
        timepoint_col='change_type',
        overall_context=y_desc,
        show_ns=show_ns
    )

    fig, ax = plt.subplots(figsize=(7,5))
    sns.despine()
    if by_group:
        hue_col = 'group'
        desc_add += '_by-group'
        hue_order = ['aging_phenotype', 'no_change']
        palette = COLOR_PALETTES['group']
        dodge = True
    else:
        hue_col = 'change_type'
        hue_order = None
        palette = COLOR_PALETTES['change_type']
        dodge = False

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )
    stripplot_mouse_markers_with_mean(
        count_df,
        'change_type',
        y_col,
        ax,
        hue_col,
        hue_order,
        order=['Unchanged', 'Lymphoid', 'Myeloid']
    )

    plt.ylabel(y_desc)
    plt.xlabel('')
    plt.legend().remove()
    fname_prefix = save_path + os.sep \
        + 'count_by_bias_change' \
        + 't' + str(timepoint) \
        + '_mtd' + str(mtd)
    fname = fname_prefix \
        + '_' + desc_add \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_n_most_abundant_at_time(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        n: int, 
        timepoint: Any,
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
    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )

    time_clones = agg.get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        timepoint,
        by_mouse=True,
    ) 
    n_most_abund = []
    for vals, g_df in time_clones.groupby(['mouse_id', 'group', 'cell_type']):
        n_most_abund.append(
            g_df.nlargest(n=n, columns='percent_engraftment')
        )
    n_most_abund_df = pd.concat(n_most_abund)

    for cell_type, c_df in n_most_abund_df.groupby(['cell_type']):
        c_df['time_desc'] = timepoint
        context = cell_type.title() + ' ' + str(n) \
            + ' Most Abundant Clone Between Groups'
        stat_tests.ind_ttest_between_groups_at_each_time(
            c_df,
            test_col='percent_engraftment',
            timepoint_col='time_desc',
            overall_context=context,
            show_ns=True
        )
        _, ax = plt.subplots(figsize=(7,5))
        stripplot_mouse_markers_with_mean(
            c_df,
            'time_desc',
            'percent_engraftment',
            ax,
            'group',
            ['aging_phenotype', 'no_change']
        )
        ax.set(yscale='log')
        plt.title(
            timepoint_col.title() + ' ' + str(timepoint)
            + ' ' + cell_type.title()
            + ' Top ' + str(n) + ' Clones'
        )
        plt.ylabel('Abundance')
        plt.xlabel(timepoint_col.title())
        plt.legend().remove()
        file_name = save_path + os.sep \
            + 'top' + str(n) \
            + '_' + timepoint_col[0] + str(timepoint) \
            + '_' + cell_type \
            + '.' + save_format

        save_plot(file_name, save, save_format)
def plot_diversity_index(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: Any,
        save: bool,
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

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )

    time_clones = agg.get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        timepoint,
        by_mouse=True,
    ) 
    diversity_df = agg.calculate_shannon_diversity(
        time_clones,
        ['mouse_id', 'group', 'cell_type']
    )

    for cell_type, c_df in diversity_df.groupby(['cell_type']):
        c_df['time_desc'] = timepoint
        context = cell_type.title() \
            + ' Shannon Diversity Between Groups'
        stat_tests.ind_ttest_between_groups_at_each_time(
            c_df,
            test_col='Shannon Diversity',
            timepoint_col='time_desc',
            overall_context=context,
            show_ns=True
        )
        _, ax = plt.subplots(figsize=(7,5))
        stripplot_mouse_markers_with_mean(
            c_df,
            'time_desc',
            'Shannon Diversity',
            ax,
            'group',
            ['aging_phenotype', 'no_change']
        )
        plt.title(
            timepoint_col.title() + ' ' + str(timepoint)
            + ' ' + cell_type.title()
        )
        plt.xlabel(timepoint_col.title())
        plt.legend().remove()
        file_name = save_path + os.sep \
            + 'shannon-diversity' \
            + '_' + timepoint_col[0] + str(timepoint) \
            + '_' + cell_type \
            + '.' + save_format

        save_plot(file_name, save, save_format)

def plot_expanded_at_time_abundance(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: Any,
        by_group: bool,
        thresholds: Dict,
        n: int,
        flip_cell_type: bool,
        by_mouse: bool,
        group: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3.5,
            'axes.linewidth': 3.5,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'medium',
        }

    )
    desc = ''
    if group != 'all':
        desc += '_' + group
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.group == group]

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        False,
    )
    if timepoint_col == 'gen':
        clonal_abundance_df =agg.filter_mice_with_n_timepoints(
            clonal_abundance_df,
            8,
            timepoint_col,
        )
    elif timepoint_col == 'month':
        print('Filtering for mice with all 4 timepoints')
        clonal_abundance_df =agg.filter_mice_with_n_timepoints(
            clonal_abundance_df,
            4,
            timepoint_col,
        )
    if n:
        print(Fore.YELLOW + '\tplotting n most abundant: ' + str(n))
        expanded_at_time = agg.get_n_most_abundant_at_time(
            clonal_abundance_df,
            n,
            timepoint_col,
            timepoint,
            by_mouse=True,
        )
        desc += '_n-'+str(n)
    else:
        expanded_at_time = agg.combine_enriched_clones_at_time(
            clonal_abundance_df,
            timepoint,
            timepoint_col,
            thresholds,
            analyzed_cell_types=['gr', 'b'],
            by_mouse=True,
        )
        desc += '_gr-'+str(round(thresholds['gr'],2))\
             + '_b-'+str(round(thresholds['b'],2))
    if by_group:
        hue = 'group'
    else:
        hue = 'mouse_id'
    clone_hue = hue
    if by_mouse:
        clone_hue = 'mouse_id'
        desc += '_by_mouse'


    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18,8))
    plt.subplots_adjust(hspace=.2, wspace=.3)
    i=0
    for (ct), s_df in expanded_at_time.groupby(['cell_type']):
        if ct not in ['gr', 'b']:
            continue
        if flip_cell_type:
            alt_ct = 'b'
            if ct == 'b':
                alt_ct = 'gr'
                desc += '_flipped'
            
            s_df = clonal_abundance_df[
                clonal_abundance_df.cell_type == alt_ct
            ].merge(
                s_df[['code', 'mouse_id', timepoint_col]].drop_duplicates(),
                how='inner',
                validate='1:1'
            )
            if s_df.empty:
                print(Fore.RED + 'No clones in other cell type: ' + ct + ' to ' + alt_ct)
                continue
        ax = axes.flatten()[i]
        i += 1
        if ct == 'gr':
            ax.set_yscale('symlog', linthreshy=10e-4)
        else:
            ax.set_yscale('symlog', linthreshy=10e-3)
        if timepoint_col == 'gen':
            compare_df = s_df[s_df.gen <= 7]
            for g, g_df in compare_df.groupby('group'):
                stat_tests.friedman_wilcoxonSignedRank(
                    g_df,
                    timepoint_col,
                    id_col='code',
                    value_col='percent_engraftment',
                    overall_context=g + ' ' + desc.replace('_', ' ') + ' ' + ct + ' ' + str(timepoint),
                    show_ns=True,
                    match_cols=['mouse_id', 'group', 'code'],
                    merge_type='inner',
                    fill_na=0,
                    do_pairwise=False,
                )
                stat_tests.one_way_ANOVArm(
                    g_df,
                    timepoint_col,
                    id_col='code',
                    value_col='percent_engraftment',
                    overall_context=g + ' ' + desc.replace('_', ' ') + ' ' + ct + ' ' + str(timepoint),
                    show_ns=True,
                    match_cols=['mouse_id', 'group', 'code'],
                    merge_type='inner',
                    fill_na=0,
                    do_pairwise=False,
                )
        sns.lineplot(
            data=s_df,
            x=timepoint_col,
            y='percent_engraftment',
            estimator=None,
            units='code',
            hue=clone_hue,
            palette=COLOR_PALETTES[clone_hue],
            ax=ax,
            alpha=0.50,
            lw=2,
        )
        sns.lineplot(
            data=s_df,
            x=timepoint_col,
            y='percent_engraftment',
            hue=hue,
            palette=['white'] * s_df.group.nunique(),
            ax=ax,
            lw=10,
        )
        sns.lineplot(
            data=s_df,
            x=timepoint_col,
            y='percent_engraftment',
            hue=hue,
            palette=COLOR_PALETTES[hue],
            ax=ax,
            ci=95,
            lw=6,
            seed=32,
        )
        if timepoint_col == 'month':
            ax.set_xticks([4, 9, 12, 15])
        ax.legend().remove()
        sns.despine()
        title_str = str(timepoint) + ' ' + ct
        if n:
            title_str = str(timepoint) + ' ' + ct + ' n:' + str(n)
        if group != 'all':
            title_str = group + ' ' + title_str
        ax.set_title(title_str)
    fname = os.path.join(
        save_path,
        'expanded_line_abundance_'
        + hue + '_' + str(timepoint)
        + desc
        + '.' + save_format
    )
    save_plot(fname, save, save_format)

def plot_extreme_bias_percent_time(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: any,
        thresh: float,
        abundance_thresholds: Dict,
        save: bool,
        save_path: str,
        save_format: str,
    ) -> None:
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 24,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'medium',
        }

    )
    filt_df = agg.remove_month_17(
        lineage_bias_df,
        timepoint_col,
    )
    timepoint_df = agg.get_clones_at_timepoint(
        filt_df,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )
    timepoint_df['time'] = timepoint
    mice_time = timepoint_df[['mouse_id', 'group', 'time']].drop_duplicates()

    timepoint_df = agg.filter_lineage_bias_thresholds(
        timepoint_df,
        abundance_thresholds,
    )
    timepoint_df['abs_bias'] = timepoint_df.lineage_bias.abs()
    print(timepoint_df.groupby(timepoint_col)['mouse_id'].unique())
    lymph_bias = timepoint_df[
        timepoint_df.lineage_bias <= -thresh
    ]
    myel_bias = timepoint_df[
        timepoint_df.lineage_bias >= thresh
    ]
    either_bias = timepoint_df[
        timepoint_df.abs_bias >= thresh
    ]
    clone_count = pd.DataFrame(
        timepoint_df.groupby(['mouse_id', 'group']).code.nunique()
    ).reset_index().rename(columns={'code':'total'})

    lymph_count = pd.DataFrame(
        lymph_bias.groupby(['mouse_id', 'group']).code.nunique()
    ).reset_index().rename(columns={'code':'lymphoid_count'})

    myel_count = pd.DataFrame(
        myel_bias.groupby(['mouse_id', 'group']).code.nunique()
    ).reset_index().rename(columns={'code':'myeloid_count'})

    either_count = pd.DataFrame(
        either_bias.groupby(['mouse_id', 'group']).code.nunique()
    ).reset_index().rename(columns={'code':'either_count'})

    counts_df = mice_time.merge(clone_count, how='left').merge(
        lymph_count,
        how='left'
    ).merge(
        myel_count,
        how='left'
    ).merge(
        either_count,
        how='left'
    ).fillna(0)

    counts_df['lymph_perc'] = 100 * counts_df['lymphoid_count']/counts_df['total']
    counts_df['myel_perc'] = 100 * counts_df['myeloid_count']/counts_df['total']
    counts_df['either_perc'] = 100 * counts_df['either_count']/counts_df['total']
    counts_df['time'] = timepoint

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    plt.subplots_adjust(wspace=.3)
    cols = ['either_perc', 'myel_perc', 'lymph_perc']


    for i in range(3):
        y_col = cols[i]
        ax = axes.flatten()[i]
        ax.set_ylim([-5,105])
        stat_tests.ind_ttest_between_groups_at_each_time(
            data=counts_df,
            test_col=y_col,
            timepoint_col='time',
            overall_context= y_col + ' bias_thresh > ' + str(thresh),
            show_ns=True
        )
        stripplot_mouse_markers_with_mean(
            counts_df,
            timepoint_col='time',
            y_col=y_col,
            ax=ax,
        )
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(y_col.replace('_', ' ').title(), fontsize=10)
        labels = [l.get_text() for l in ax.get_xticklabels()]
        labels = [l.replace('aging_phenotype', 'E-MOLD').replace('no_change', 'D-MOLD') for l in labels]
        ax.set_xticklabels(labels)
    plt.suptitle(
        'Gr: ' + str(round(abundance_thresholds['gr'], 2)) +
        ' B: ' + str(round(abundance_thresholds['b'], 2)) +
        ' Bias Threshold: ' + str(thresh),
        fontsize=15
    )
    desc = 'biasthresh-' + str(thresh) + \
        '_' + timepoint_col[0] + str(timepoint) + \
        '_gr' + str(round(abundance_thresholds['gr'], 3)) +\
        '_b' + str(round(abundance_thresholds['b'], 3))
    fname = os.path.join(
        save_path,
        desc + '.' + save_format
    )

    save_plot(fname, save, save_format)


def stripplot_label_markers_with_mean(
    agg_df: pd.DataFrame,
    timepoint_col: str,
    y_col: str,
    ax,
    label_col: str,
    label_order: List = None,
    order: List = None,
    color: Any = None,
    ):

    if not label_order:
        label_order = agg_df[label_col].unique()
    if order:
        times = order
    else:
        times = agg_df.sort_values(by=timepoint_col)[timepoint_col].unique()
    agg_df['label_time'] = agg_df[label_col].astype(str).str[0:2] + '-' + agg_df[timepoint_col].astype(str).str[:2]
    order = []
    palette = {}
    for time in times:
        for label in label_order:
            order.append(str(label)[0:2] + '-' + str(time)[:2])
            if color:
                palette[order[-1]] = color
            else:
                palette[order[-1]] = COLOR_PALETTES[timepoint_col][time]
    print(palette)

    for (label), l_df in agg_df.groupby([label_col]):
        sns.stripplot(
            x='label_time',
            order=order,
            y=y_col,
            dodge=True,
            ax=ax,
            data=l_df,
            palette=palette,
            marker=MARKERS[label_col][label],
            size=12,
            linewidth=1,
            alpha=0.8,
            zorder=0,
        )
    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=3,
        color='black'
    )

    sns.boxplot(
        x='label_time',
        y=y_col,
        order=order,
        data=agg_df,
        ax=ax,
        fliersize=0,
        showbox=False,
        whiskerprops={
            "alpha": 0
        },
        showcaps=False,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
    )
    sns.despine()
    ax.legend().remove()
    ax.set_xlabel(label_col + '-' + timepoint_col)

def barplot_hatch_diff(
        data: pd.DataFrame,
        y:str,
        x:str,
        hatch_col: str,
        hatch_dict: Dict,
        color: str,
        ax,
        order: np.array = None,
        hatch_order: List = None,
    ):
    if hatch_order is None:
        hatch_order = data[hatch_col].unique()
    if order is None:
        order = data.sort_values(by=x)[x].unique()
    
    hue_order = ['first', 'last']
    for t, t_df in data.groupby(hatch_col):
        kws = {'hatch': hatch_dict[t]}
        sns.barplot(
            data=t_df,
            y=y,
            x=x,
            order=order,
            hue_order=hatch_order,
            capsize=.1,
            errwidth=3,
            ci=68,
            hue=hatch_col,
            saturation=1,
            zorder=10,
            palette=[color] * 2,
            ax=ax,
            **kws
        )

        sns.barplot(
            data=t_df,
            y=y,
            x=x,
            order=order,
            hue_order=hatch_order,
            hue=hatch_col,
            zorder=1,
            ci=None,
            saturation=1,
            palette=[color] * 2,
            ax=ax,
        )


def plot_blood_bias_abundance_time(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
):
    if not by_group:
        raise ValueError('--by-group flag must be set for this plot')

    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3.5,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'medium',
        }

    )

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        False,
    )
    n_time = clonal_abundance_df[timepoint_col].nunique()
    clonal_abundance_df =agg.filter_mice_with_n_timepoints(
        clonal_abundance_df,
        n_time,
        timepoint_col,
    )
    bias_cat_df = agg.add_bias_category(lineage_bias_df)
    bias_cat_df = bias_cat_df[[
        'mouse_id',
        'code',
        timepoint_col,
        'bias_category'
    ]].drop_duplicates()
    abundance_with_bias_cat = clonal_abundance_df.merge(
        bias_cat_df,
        how='inner',
        validate='m:1'
    )
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_time,
        figsize=(6*n_time, 10)
    )
    y_col = 'code'
    order = ['LB', 'B', 'MB']
    hue_order = ['aging_phenotype', 'no_change']
    time_order = clonal_abundance_df\
        .sort_values(by=timepoint_col)[timepoint_col]\
        .unique()
    mice = clonal_abundance_df.mouse_id.unique()
    mouse_group = clonal_abundance_df[['mouse_id', 'group']].drop_duplicates()
    for row, cell_type in enumerate(['gr', 'b']):
        ax_row = axes[row]
        ct_df = abundance_with_bias_cat[abundance_with_bias_cat.cell_type == cell_type]
        ct_df = ct_df[ct_df['percent_engraftment'] > 0]
        y_desc = cell_type.title() + ' Cell Count'
        agg_df = pd.DataFrame(
            ct_df.groupby(['mouse_id', 'group', 'bias_category', timepoint_col])\
                [y_col].nunique()
        ).reset_index()
        filled_dfs = []
        for t, t_df in agg_df.groupby(timepoint_col):
            filled = agg.fill_mouse_id_zeroes(
                    t_df,
                    info_cols=[timepoint_col, 'group'],
                    fill_col=y_col,
                    fill_cat_col='bias_category',
                    fill_cats=order,
                    fill_val=0,
                    mice_to_fill=mice,
            ).drop(columns='group').merge(
                mouse_group
            )
            if filled.mouse_id.nunique() != len(mice):
                print(filled.mouse_id.unique())
                print(mice)
                raise ValueError('All mice not represented')
            filled_dfs.append(filled)
        zero_filled = pd.concat(filled_dfs)
        for i, ax in enumerate(ax_row):
            t_df = zero_filled[zero_filled[timepoint_col] == time_order[i]]
            sns.barplot(
                y=y_col,
                x='bias_category',
                hue='group',
                hue_order=hue_order,
                order=order,
                data=t_df,
                palette=COLOR_PALETTES['group'],
                saturation=1,
                capsize=.2,
                ci=68,
                ax=ax,
                zorder=1,
            )
            sns.barplot(
                y=y_col,
                x='bias_category',
                hue='group',
                hue_order=hue_order,
                order=order,
                data=t_df,
                palette=COLOR_PALETTES['group'],
                saturation=1,
                ci=None,
                ax=ax,
                zorder=10
            )
            stat_tests.ind_ttest_between_groups_at_each_time(
                data=t_df,
                test_col=y_col,
                timepoint_col='bias_category',
                overall_context=timepoint_col.title() + ' ' + str(t) + ' ' + y_desc,
                show_ns=True
            )
            for g, g_df in t_df.groupby('group'):
                stat_tests.one_way_ANOVArm(
                    data=g_df,
                    timepoint_col='bias_category',
                    id_col='mouse_id',
                    value_col=y_col,
                    overall_context= g + ' ' + timepoint_col.title() + ' ' + str(t) + ' ' + y_desc,
                    show_ns=True,
                    match_cols=['mouse_id', 'group'],
                    merge_type='inner',
                    fill_na=None
                )
            for k, spine in ax.spines.items():  #ax.spines is a dictionary
                spine.set_zorder(100)
            ax.legend().remove()
            ax.set_xlabel('')
            ax.set_ylabel('')
            if i == 0:
                ax.set_ylabel(y_desc.title())
            ax.set_title(
                cell_type.title() + ' ' + timepoint_col[0].title() + ' ' + str(time_order[i])
            )
            sns.despine()

    desc = 'blood_count_by-group'
    fname = os.path.join(
        save_path,
        desc + '.' + save_format
    )

    save_plot(fname, save, save_format)

def plot_abundance_change_stable_group_grid(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        change_type: str,
        timepoint: str,
        by_mouse:bool,
        mtd: int,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    """ Plot a scatter/stripplot of abundance change between first 
    and last time point per mouse.
    
    Arguments:
        lineage_bias_df {pd.DataFrame}
        timepoint_col {str} 
        change_type {str} -- direction of change to analyze
        by_mouse {bool} -- set to plot avg change per mouse
        by_clone {bool} -- set to plot absolute difference instead of log2xChange
    
    Keyword Arguments:
        save {bool} 
        save_path {str}
        save_format {str}
    
    Returns:
        None
    """
    change_param = 'change_type'
    y_cols =  ['myeloid_change', 'lymphoid_change']
    group_cols = ['mouse_id', 'group', 'bias_category']

    if by_mouse:
        group_desc = 'by-mouse'
    else:
        group_desc = 'by-clone'

    # NOTE By_Clone here is used as a flag for log2 fold vs linear change
    math_desc = 'absolute_change'

    sns.set_context(
        'paper',
        font_scale=1,
        rc={
            'lines.linewidth': 1,
            'axes.linewidth': 3,
            'axes.labelsize': 5,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'figure.titlesize': 'small',
        }
    )

    lineage_bias_df = agg.remove_month_17_and_6(
        lineage_bias_df,
        timepoint_col
    )
    lineage_bias_df = agg.remove_gen_8_5(
        lineage_bias_df,
        timepoint_col,
        keep_hsc=False,
    )
    clone_bias = agg.add_bias_category(
        agg.get_clones_at_timepoint(
            lineage_bias_df,
            timepoint_col,
            timepoint,
            by_mouse,
        )
    ).reset_index()
    clone_bias = clone_bias[['code', 'mouse_id', 'bias_category']].drop_duplicates()

    

    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False,
    )
    marked_bias_change_df = agg.mark_changed(
        bias_change_df,
        bias_change_df,
        min_time_difference=mtd,
    )
    marked_bias_change_df= marked_bias_change_df.assign(
        myeloid_change=lambda x: x.myeloid_percent_abundance_last - x.myeloid_percent_abundance_first,
        lymphoid_change=lambda x: x.lymphoid_percent_abundance_last - x.lymphoid_percent_abundance_first,
    )
    if timepoint_col == 'gen':
        marked_bias_change_df = marked_bias_change_df[marked_bias_change_df.gen != 8.5]

    stable_only = marked_bias_change_df[marked_bias_change_df[change_param] == 'Unchanged'] 
    stable_only = stable_only.merge(clone_bias, how='inner', validate='1:1')


    if by_mouse:
        avg_abund_per_mouse = pd.DataFrame(stable_only.groupby(
            group_cols
            )[y_cols].mean()).reset_index()
        
        melt_df = pd.melt(
            avg_abund_per_mouse,
            id_vars=group_cols,
            value_vars=y_cols
        )
    else:
        melt_df = pd.melt(
            stable_only,
            id_vars=['code'] + group_cols,
            value_vars=y_cols
        )
    melt_df['group_first'] = melt_df['group'].str[0]
    melt_df['bias-group'] = melt_df['bias_category'].str.cat(melt_df['group_first'], sep='-')
    col_order = ['LB-a', 'LB-n', 'B-a', 'B-n', 'MB-a', 'MB-n']

    # Find sig difference
    nan_inf_handle = 'propagate'

    show_ns=True
    stat_tests.ttest_1samp(
        data=melt_df,
        group_vars=['bias-group', 'variable'],
        value_var='value',
        null_mean=0,
        overall_context='Phenotype-Bias Clone Abundance Change ' + math_desc + ' ' + group_desc,
        show_ns=show_ns,
        handle_nan=nan_inf_handle,
        handle_inf=nan_inf_handle,
    )

    for var, v_df in melt_df.groupby('variable'):
        stat_tests.ranksums_test_group_time(
            data=v_df,
            test_col='value',
            timepoint_col='bias_category',
            overall_context=var,
            group_col='group',
            show_ns=show_ns,
        )
    print(Fore.RED + 'FACET STARTING') 
    g = sns.FacetGrid(
        melt_df,
        row='variable',
        hue='group',
        palette=COLOR_PALETTES['group'],
        sharey='row',
        aspect=3
    )
    print(Fore.RED + 'FORMATTING AXES') 
    for ax in g.axes.flat:
        ax.tick_params(axis='y', labelleft=True)
        ax.axhline(y=0, color='gray', linestyle='dashed', linewidth=1.5, zorder=0)
        ax.set_yscale('symlog', linthreshy=10e-2)

    medianprops = dict(
        linewidth=0,
    )
    meanprops = dict(
        linestyle='solid',
        linewidth=2,
        color='#2f3640'
    )
    def violin_mean(x, y, **kwargs):
        ax = sns.boxenplot(
            x=x,
            y=y,
            lw=0,
            **kwargs,
        )
        sns.boxplot(
            x=x,
            y=y,
            whiskerprops={
                "alpha": 0
            },
            order=col_order,
            showcaps=False,
            showmeans=True,
            meanline=True,
            meanprops=meanprops,
            medianprops=medianprops,
            ax=ax,
            fliersize=0,
            showbox=False,
        )
    if by_mouse:
        g.map(
            sns.swarmplot,
            "bias-group",
            "value",
            order=col_order,
            zorder=0,
        )
    else:
        print(Fore.RED + 'MAPPING TO FACETGRID') 
        g.map(
            violin_mean,
            "bias-group",
            "value",
            order=col_order,
        )


    fname = save_path + os.sep \
        + 'face_grid_abundance_change_' \
        + str(timepoint) \
        + '_' + change_type \
        + '_' + math_desc \
        + '_' + group_desc \
        + '.' + save_format
    save_plot(fname, save, save_format)

def mouse_marker_legend(
        clonal_abundance_df: pd.DataFrame,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> None:
    sns.set_context(
        'paper',
        font_scale=1,
        rc={
            'lines.linewidth': 1,
            'axes.linewidth': 3,
            'axes.labelsize': 5,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'figure.titlesize': 'small',
        }
    ) 

    clonal_abundance_df = clonal_abundance_df.sort_values(by='group')
    mice = clonal_abundance_df.mouse_id.unique()
    n_mice = len(mice)
    vals = [1] * n_mice
    fig, ax = plt.subplots(figsize=(n_mice, 2))

    ax.scatter(
        mice,
        vals,
        c='white',
    )
    for mouse in mice:
        ax.scatter(
            [mouse],
            [1],
            c=COLOR_PALETTES['mouse_id'][mouse],
            marker=MARKERS['mouse_id'][mouse],
            s=300,
            edgecolors='gray',
            linewidths=1,
        )
    ax.tick_params(axis='x', labelrotation=90)
    sns.despine()

    fname = save_path + os.sep \
        + 'mouse_marker_legend.' + save_format
    save_plot(fname, save, save_format)

def plot_change_status_bias_at_time(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        timepoint: Any,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 5,
            'axes.linewidth': 3,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.major.size': 4,
            'ytick.minor.size': 4,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'medium',
        }
        )
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False,
    )
    change_marked_df = agg.mark_changed(
        lineage_bias_df,
        bias_change_df,
        min_time_difference=mtd,
    )
    bias_cat = agg.add_bias_category(
        change_marked_df,
    )
    first_last_clones = agg.get_clones_exist_first_and_last_per_mouse(
        bias_cat,
        timepoint_col
    )
    time_df = agg.get_clones_at_timepoint(
        first_last_clones,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )
    if timepoint_col == 'month':
        print(Fore.YELLOW + 'Only including mice from contribution analysis')
        time_df = time_df[
            time_df.mouse_id.isin([
            'M2012', 'M2059', 'M3010', 'M3013', 'M3016', 'M190', 'M2061', 'M3000', 'M3012', 'M3001', 'M3009', 'M3018', 'M3028'
            ])
        ]
    y_col = 'code'
    total = pd.DataFrame(
        time_df.groupby('mouse_id').code.nunique()
    ).reset_index().rename(columns={'code':'total'})
    agg_df = pd.DataFrame(
        time_df.groupby([
            'mouse_id',
            'group',
            'bias_category',
            'change_type'
        ])[y_col].nunique()
    ).reset_index()
    filled = []
    change_order = ['Lymphoid', 'Unchanged', 'Myeloid']
    bias_order = ['LB', 'B', 'MB']
    for _, b_df in agg_df.groupby('bias_category'):
        filled.append(
            agg.fill_mouse_id_zeroes(
                b_df,
                info_cols=['group', 'bias_category'],
                fill_col=y_col,
                fill_cat_col='change_type',
                fill_cats=change_order,
                fill_val=0,
            )
        )
    zero_filled_df = pd.concat(filled).merge(total)
    zero_filled_df['perc'] = 100 * zero_filled_df[y_col]/zero_filled_df['total']
    print(zero_filled_df)
    y_col = 'perc'

    if not by_group:
        zero_filled_df['group'] = 'all'

    for g, g_df in zero_filled_df.groupby('group'):
        piv = g_df.pivot_table(
            values=y_col,
            index='bias_category',
            columns='change_type',
            aggfunc=np.median,
        )
        piv.iloc[0] = 100*piv.iloc[0]/piv.iloc[0].sum()
        piv.iloc[1] = 100*piv.iloc[1]/piv.iloc[1].sum()
        piv.iloc[2] = 100*piv.iloc[2]/piv.iloc[2].sum()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(
            piv[change_order].reindex(bias_order),
            cmap='magma',
            vmin=0,
            vmax=100,
            ax=ax
        )
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        hlines = np.linspace(ylims[0], ylims[1], 4)
        ax.hlines(hlines, *xlims, colors='white')

        vlines = np.linspace(*xlims, 4)
        ax.vlines(vlines, *ylims, colors='white')
        ax.set_title(g)


        fname_prefix = save_path + os.sep + 'median_perc_bias_at_change_'
        fname = fname_prefix + g  \
            + '_' + timepoint_col[0] +  str(timepoint) \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_bias_change_cutoff_hist(
        lineage_bias_df: pd.DataFrame,
        thresholds: Dict[str, float],
        timepoint_col: str,
        timepoint: float = None,
        abundance_cutoff: float = 0.0,
        group: str = 'all',
        min_time_difference: int = 0,
        save: bool = False,
        save_path: str = 'output',
        save_format: str = 'png',
    ) -> None:
    """ Plots histogram of bias change annotated with line to cut "change" vs "non change" clones

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

    sns.set_context(
        'paper',
        font_scale=2,
        rc={
            'lines.linewidth': 3,
            'lines.markersize': 6,
            'axes.linewidth': 3,
            'axes.labelsize': 20,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 24,
            'ytick.labelsize': 24,
            'figure.titlesize': 'small',
        }
    )


    filt_lineage_bias_df = agg.filter_lineage_bias_anytime(
        lineage_bias_df,
        thresholds=thresholds
    )
    bias_change_df = agg.calculate_first_last_bias_change(
        filt_lineage_bias_df,
        timepoint_col,
        by_mouse=False,
        )
        
    timepoint_text = ''
    if timepoint is not None:
        bias_change_df = agg.filter_bias_change_timepoint(
            bias_change_df,
            timepoint
        )
        timepoint_text = ' - Clones Must have First or Last Time at: ' +str(timepoint)
    bias_change_df = bias_change_df[bias_change_df.time_change >= min_time_difference]
    if group != 'all':
        bias_change_df = bias_change_df.loc[bias_change_df.group == group]

    fig, ax = plt.subplots(figsize=(6,5))
    _, _, _, _, x_c, _, _ = agg.calculate_bias_change_cutoff(
        bias_change_df,
        min_time_difference=min_time_difference,
        timepoint=timepoint,
    )
    alpha = 0.5
    c_u = mpl.colors.to_rgba(COLOR_PALETTES['change_status']['Unchanged'], alpha)
    ax.hist(
        bias_change_df[bias_change_df.bias_change.abs() < x_c[0]].bias_change,
        range=(-2,2),
        facecolor=c_u,
        edgecolor=COLOR_PALETTES['change_status']['Unchanged'],
        lw=2
    ) 
    c_c = mpl.colors.to_rgba(COLOR_PALETTES['change_status']['Changed'], alpha)
    ax.hist(
        bias_change_df[bias_change_df.bias_change.abs() >= x_c[0]].bias_change,
        range=(-2,2),
        facecolor=c_c,
        edgecolor=COLOR_PALETTES['change_status']['Changed'],
        lw=2
    ) 
    ax.set_xlabel('Lineage Bias Change')
    ax.set_ylabel('Clone Count')
    #ax.axvline(x_c[0], c='gray', ls='--')
    #ax.axvline(-x_c[0], c='gray', ls='--')
    ax.set_xticks([-2, -1, 0, 1, 2])
    sns.despine()

    fname = save_path + os.sep \
        + 'bias_change_cutoff_hist_a' + str(abundance_cutoff).replace('.', '-') \
        + '_' + group \
        + '_mtd' + str(min_time_difference) \
        + '_' + timepoint_col[0] + str(timepoint) \
        + '.' + save_format
    save_plot(fname, save, save_format)


def plot_change_status_bias_at_time_abundance(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        mtd: int,
        timepoint: Any,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        font_scale=2.0,
        rc={
            'lines.linewidth': 5,
            'axes.linewidth': 3,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.major.size': 4,
            'ytick.minor.size': 0,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'medium',
        }
        )
    agg_func = 'sum'
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False,
    )
    bias_change_df = bias_change_df.assign(
        myeloid_change=lambda x: x.myeloid_percent_abundance_last - x.myeloid_percent_abundance_first,
        lymphoid_change=lambda x: x.lymphoid_percent_abundance_last - x.lymphoid_percent_abundance_first,
    )
    change_marked_df = agg.mark_changed(
        lineage_bias_df,
        bias_change_df,
        min_time_difference=mtd,
    ).merge(
        bias_change_df[
            ['code', 'mouse_id','myeloid_change', 'lymphoid_change']
        ].drop_duplicates()
    )
    bias_cat = agg.add_bias_category(
        change_marked_df,
    )
    time_df = agg.get_clones_at_timepoint(
        bias_cat,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )
    for abundance in ['myeloid_change', 'lymphoid_change']:
        agg_df = pd.DataFrame(
            time_df.groupby([
                'mouse_id',
                'group',
                'bias_category',
                'change_type'
            ])[abundance].mean()
        ).reset_index()
        filled = []
        change_order = ['Lymphoid', 'Unchanged', 'Myeloid']
        bias_order = ['LB', 'B', 'MB']
        for _, b_df in agg_df.groupby('bias_category'):
            filled.append(
                agg.fill_mouse_id_zeroes(
                    b_df,
                    info_cols=['group', 'bias_category'],
                    fill_col=abundance,
                    fill_cat_col='change_type',
                    fill_cats=change_order,
                    fill_val=0,
                )
            )
        zero_filled_df = pd.concat(filled)

        if not by_group:
            zero_filled_df['group'] = 'all'

        ticks = [-1, -.1, 1e02, 0, 1e-2, .1, 1]
        if agg_func == 'sum':
            ticks = [-10, -1, -.1, 0, .1, 1, 10]
        for g, g_df in zero_filled_df.groupby('group'):
            piv = g_df.pivot_table(
                values=abundance,
                index='bias_category',
                columns='change_type',
                aggfunc=agg_func,
            )
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(
                piv[change_order].reindex(bias_order),
                cmap='magma',
                ax=ax,
                norm=co.SymLogNorm(linthresh=1e-2),
                vmin=ticks[0],
                vmax=ticks[-1]
            )
            cbar = ax.collections[-1].colorbar
            cbar.ax.set_yticks(ticks)
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()

            hlines = np.linspace(ylims[0], ylims[1], 4)
            ax.hlines(hlines, *xlims, colors='white')

            vlines = np.linspace(*xlims, 4)
            ax.vlines(vlines, *ylims, colors='white')
            ax.set_title(g + ' ' + abundance)


            fname_prefix = save_path + os.sep \
                + agg_func + '_perc_bias_at_change_' \
                + abundance  \
                + '_' + str(timepoint) + '_'
            fname = fname_prefix + g  \
                + '.' + save_format
            save_plot(fname, save, save_format)


def plot_change_contributions_refactor(
        changed_marked_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: Any,
        gfp_donor: pd.DataFrame,
        gfp_donor_thresh: float,
        force_order: bool,
        save: bool = False,
        bar_col: str = 'change_type',
        bar_types: List = ['Lymphoid', 'Unchanged', 'Myeloid', 'Unknown'],
        present_thresh: float = 0.01,
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


    if timepoint_col == 'gen':
        changed_marked_df = changed_marked_df[changed_marked_df.gen != 8.5]

    filt_bar_types = changed_marked_df[changed_marked_df[bar_col].isin(bar_types)]
    if filt_bar_types.shape != changed_marked_df.shape:
        print("Not all types being displayed, missing:")
        print(
            changed_marked_df[~changed_marked_df[bar_col].isin(bar_types)][bar_col].unique()
        )

    percent_of_total = True
    changed_sum_df = agg.sum_abundance_by_change(
        agg.remove_month_17(
            filt_bar_types,
            timepoint_col
        ),
        timepoint_col=timepoint_col,
        percent_of_total=percent_of_total,
        change_col=bar_col,
    )
    changed_sum_df = changed_sum_df[changed_sum_df.cell_type.isin(['gr', 'b'])]
    changed_sum_with_gxd = changed_sum_df.merge(
        gfp_donor,
        how='inner',
        validate='m:1',
        on=['mouse_id', 'cell_type', timepoint_col]
    )

    timepoint_df = agg.get_clones_at_timepoint(
        changed_sum_with_gxd,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )

    print('FILTERING FOR MICE FOUND IN FIRST AND LAST TIMEPOINT ABOVE GFP x DONOR THRESHOLD' )
    mice_left = agg.filter_gxd_first_last(
        changed_sum_with_gxd,
        timepoint_col,
        threshold=gfp_donor_thresh
    )


    if (bar_col == 'survived') and ('M2012' in mice_left):
        print('Excluding M2012')
        mice_left = [m for m in mice_left if m != 'M2012']
    print(mice_left)

    filt_gxd = timepoint_df[timepoint_df.mouse_id.isin(mice_left)]
    print('Pre-filt gxd mice:', timepoint_df.mouse_id.nunique())
    print('Post-filt gxd mice:', filt_gxd.mouse_id.nunique())

    filt_gxd = filt_gxd.assign(total=100)
    filt_gxd = filt_gxd.sort_values(by='percent_engraftment', ascending=False)
    print(filt_gxd.groupby('mouse_id')[timepoint_col].unique())


    for group, g_df in filt_gxd.groupby('group'):
        avg_per_group = pd.DataFrame(
            g_df.groupby(
            [bar_col, 'cell_type']
            ).percent_engraftment.sum()/g_df.mouse_id.nunique()
        ).reset_index()
        avg_per_group['total'] = 100
        avg_per_group['mouse_id'] = 'Average'
        group_df = g_df.append(avg_per_group, sort=False)
        ''' Save data to CSV
        group_df[[
            'mouse_id',
            'month',
            'cell_type',
            'group',
            'survived',
            'percent_engraftment',
        ]].to_csv(group + '_contrib_' + bar_col + '.csv', index=False )
        '''
        for (m, cell_type), m_df in group_df.groupby(['mouse_id', 'cell_type']):
            b_dfs = []
            for i, b_type in enumerate(bar_types):
                t_df = m_df[m_df[bar_col] == b_type]
                if t_df.empty:
                    temp_row = pd.DataFrame()
                    temp_row['mouse_id'] = [m]
                    temp_row['cell_type'] = [cell_type]
                    temp_row[bar_col] = b_type
                    temp_row['total'] = 100
                    temp_row['percent_engraftment'] = 0
                    t_df = temp_row
                    group_df = group_df.append(temp_row, sort=False)
                elif len(t_df) > 1:
                    print(b_type, '> 1')
                    print(m, cell_type, len(t_df))
                b_dfs.append(t_df)

            for i, b_type in enumerate(bar_types):
                type_bool = (
                    (group_df.mouse_id == m) &
                    (group_df[bar_col] == b_type) &
                    (group_df.cell_type == cell_type)
                )
                group_df.loc[
                    type_bool,
                    'percent_engraftment'
                ] = b_dfs[i].percent_engraftment.values[0]
                if i > 0:
                    for j in range(i):
                        group_df.loc[
                            type_bool,
                            'percent_engraftment'
                        ] = b_dfs[j].percent_engraftment.values[0]\
                            + group_df[type_bool].percent_engraftment.values[0]


        if force_order:
            print(Fore.YELLOW + 'FORCING ORDER BASED ON HARD CODED VALUES')
            if group == 'aging_phenotype':
                force_order = ['M3003', 'M3007', 'M3010', 'M3013', 'M3015', 'M3011', 'M2012', 'M2059', 'M3012', 'M3025', 'M3016']
                if timepoint_col == 'gen':
                    force_order = ['M5', 'M12', 'M3', 'M1', 'M10']
            elif group == 'no_change':
                force_order = ['M3022', 'M3008', 'M3019', 'M3023', 'M2061', 'M3009', 'M190', 'M3028', 'M3018', 'M3001', 'M3000', 'M3017']
                if timepoint_col == 'gen':
                    force_order = ['M7', 'M16', 'M6']
            else:
                raise ValueError('Group not identified, order not forcable')
                
            order = [m for m in force_order if m in group_df.mouse_id.unique()]
        else: 
            order = list(
                pd.DataFrame(
                    group_df[
                        (group_df[bar_col] != 'Unknown') &
                        (group_df.mouse_id != 'Average')
                    ].groupby(
                        ['mouse_id']
                    ).percent_engraftment.sum()
                ).reset_index().sort_values(
                by='percent_engraftment',
                ascending=False,
                ).mouse_id.unique()
            )
            print(group, ':', order)

        order.append('Average')
        plt.figure(figsize=(7,10))
        print(y_col_to_title(group) + ' Mice: ' + str(group_df.mouse_id.nunique()))
        bar_types_plot_order = bar_types[::-1]
        ax = sns.barplot(
            x='percent_engraftment',
            y='mouse_id',
            hue='cell_type',
            order=order,
            hue_order=['gr', 'b'],
            data=group_df[group_df[bar_col] == bar_types_plot_order[0]],
            palette=[COLOR_PALETTES[bar_col][bar_types_plot_order[0]]]*2,
            saturation=1,
        )
        for b_type in bar_types_plot_order[1:]:
            sns.barplot(
                x='percent_engraftment',
                y='mouse_id',
                hue='cell_type',
                order=order,
                hue_order=['gr', 'b'],
                data=group_df[group_df[bar_col]== b_type],
                palette=[COLOR_PALETTES[bar_col][b_type]]*2,
                saturation=1,
                ax=ax,
            )
        plt.xlabel('Contribution of Changed Cells')
        plt.ylabel('')
        plt.suptitle(' Cumulative Abundance of Clones by Bias Change')
        plt.title('Group: ' + group.replace('_', ' ').title() + ' ' + timepoint_col.title() + ' ' + str(timepoint))
        plt.gca().legend(title='Change Direction', loc='lower right').remove()
        sns.despine(left=True, bottom=True)

        fname = save_path + os.sep + 'contribution_bar' \
            + '_' + bar_col \
            + '_' + group \
            + '_gxd-' + str(gfp_donor_thresh) \
            + '_' + timepoint_col[0] + '-' + str(timepoint) \
            + '_' + str(present_thresh) \
            + '.' + save_format
        save_plot(fname, save, save_format)

def plot_hsc_vs_cell_type_lb_change(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        alpha: float,
        thresholds: Dict[str, float],
        cell_type: str,
        by_group: bool,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png'
    ) -> pd.DataFrame:
    sns.set_context(
        'paper',
        font_scale=1.5,
        rc={
            'lines.linewidth': 3,
            'axes.linewidth': 3,
            'axes.labelsize': 25,
            'xtick.major.width': 3,
            'ytick.major.width': 3,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }
    )
    tenx_mice_only = False # Plot only 10X sequenced mice
    plot_b = False # Plots only B abundance on Y-axis, if cell_type gr, plots B abundance of isolated clones
    save_labels = False # Save code-hsc abundance isolated labels
    plot_boundary = False # Plot decision boundary
    plot_upper = False # Plot decision boundary upper
    mark_decision = False # Wether to create bootstrap confidence intervals or na
    do_stats = False 
    if timepoint_col != 'month':
        raise ValueError('Plot only works for aging over time')

    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col,
        by_mouse=False,
    )
    change_marked = agg.mark_changed(
        clonal_abundance_df,
        bias_change_df,
        min_time_difference=3
    )    
    clonal_abundance_df = agg.remove_month_17_and_6(
        change_marked,
        timepoint_col
    )
    clonal_abundance_df = clonal_abundance_df.merge(
        agg.add_bias_category(lineage_bias_df)[
            ['code', 'mouse_id', timepoint_col, 'bias_category']
        ],
        how='left'
    )


    filt_df = clonal_abundance_df[
        clonal_abundance_df.cell_type.isin([cell_type, 'hsc'])
    ]
    y_col = cell_type + '_percent_engraftment'
    x_col = 'hsc_percent_engraftment'
    wide_ct_abundance_df = agg.abundance_to_long_by_cell_type(
        filt_df,
        timepoint_col
    ).dropna(subset=[x_col, y_col])
    wide_ct_abundance_df = wide_ct_abundance_df[
        wide_ct_abundance_df.hsc_percent_engraftment > 0
    ]

    sub_sample_amount = .95
    sub_sample_count = round(len(wide_ct_abundance_df) * sub_sample_amount)
    n_sub_samples = 100
    # Log Transform Data:
    wide_ct_abundance_df[x_col] = np.log10(1+(wide_ct_abundance_df[x_col] * 1000))
    wide_ct_abundance_df[y_col] = np.log10(1+(wide_ct_abundance_df[y_col] * 1000))
    

    if mark_decision:
        for i in progressbar.progressbar(range(n_sub_samples)):
            sub_sample_df = wide_ct_abundance_df.sample(
                n=sub_sample_count,
                random_state=i
            )
            reg_model = sm.OLS(
                sub_sample_df[y_col],
                sub_sample_df[x_col]
            )
            res = reg_model.fit()
            prstd, iv_l, iv_u = wls_prediction_std(res, alpha=alpha)
            x, y = agg.sort_xy_lists(sub_sample_df[x_col], iv_l)
            xu, yu = agg.sort_xy_lists(sub_sample_df[x_col], iv_u)


            if i == 0:
                lower_lims = pd.DataFrame.from_dict({'hsc_percent_engraftment': x, str(i): y}).drop_duplicates()
                upper_lims = pd.DataFrame.from_dict({'hsc_percent_engraftment': xu, str(i): yu}).drop_duplicates()
            else:
                lower_lims = lower_lims.merge(
                    pd.DataFrame.from_dict({'hsc_percent_engraftment': x, str(i): y}).drop_duplicates(),
                    how='outer',
                    validate='1:1'
                )
                upper_lims = upper_lims.merge(
                    pd.DataFrame.from_dict({'hsc_percent_engraftment': xu, str(i): yu}).drop_duplicates(),
                    how='outer',
                    validate='1:1'
                )

        boundary = lower_lims.set_index('hsc_percent_engraftment').min(axis=1)
        boundary = boundary.reset_index().rename(columns={0: 'boundary'})
        upper_boundary = upper_lims.set_index('hsc_percent_engraftment').min(axis=1)
        upper_boundary = upper_boundary.reset_index().rename(columns={0: 'boundary'})
        with_boundary_df = wide_ct_abundance_df.merge(
            boundary,
            how='inner',
            validate='m:1'
        )
        with_boundary_df['in_boundary'] = with_boundary_df[y_col] < with_boundary_df['boundary']
    else:
        with_boundary_df = wide_ct_abundance_df
        with_boundary_df['in_boundary'] = False

    if by_group:
        hue_col = 'group'
    else:
        hue_col = 'mouse_id'
    
    '''
    _, ax = plt.subplots(figsize=(5, 5))
    c_df = with_boundary_df[with_boundary_df.change_type == 'Unchanged']
    ax.scatter(
        c_df[x_col],
        c_df[y_col],
        edgecolors=(.176, .203, .176,0.5),
        s=80,
        linewidths=1,
        c=[(0, 0, 0, 0)] * len(c_df[x_col])
    )
    for change_type in ['Lymphoid', 'Myeloid']:
        c_df = with_boundary_df[with_boundary_df.change_type == change_type]
        ax.scatter(
            c_df[x_col],
            c_df[y_col],
            edgecolors=COLOR_PALETTES['change_type'][change_type],
            s=100,
            linewidths=2,
            c=[(0, 0, 0, 0)] * len(c_df[x_col])
        )

    ticks = ticker.FuncFormatter(
        lambda x, pos: r'$10^{' + r'{0:g}'.format(x - 3) + r'}$'
    )
    ax.xaxis.set_major_formatter(ticks)
    ax.yaxis.set_major_formatter(ticks)
    sns.despine()
    ax.set_xlabel('HSC Abundance')
    ax.set_ylabel(cell_type.title() + ' Abundance')
    ax.set_title(
        cell_type.title()
        + ' sub-samples: ' + str(n_sub_samples)
        + ' sample-perc: ' + str(100 * sub_sample_amount)
        + ' alpha: ' + str(alpha)
        + ' ' + change_type,
        fontsize=10
        )
    inv_desc = ''
    '''
    inv_desc = ''
    _, axes = plt.subplots(ncols=3, nrows=1, figsize=(18,5))
    plt.subplots_adjust(wspace=.35, hspace=.35)
    for i, change_type in enumerate(['Lymphoid', 'Unchanged', 'Myeloid']):
        ax = axes[i]
        for h_val, h_df in with_boundary_df.groupby(hue_col):
            c_df = h_df[
                (h_df.change_type == change_type)
            ]
            not_c_df = h_df[
                (h_df.change_type != change_type)
            ]

            ax.scatter(
                not_c_df[x_col],
                not_c_df[y_col],
                edgecolors=(.176, .203, .176,0.5),
                s=80,
                linewidths=1,
                c=[(0, 0, 0, 0)] * len(not_c_df[x_col])
            )
            ax.scatter(
                c_df[x_col],
                c_df[y_col],
                edgecolors=COLOR_PALETTES[hue_col][h_val],
                s=100,
                linewidths=2,
                c=[(0, 0, 0, 0)] * len(c_df[x_col])
            )

            ticks = ticker.FuncFormatter(
                lambda x, pos: r'$10^{' + r'{0:g}'.format(x - 3) + r'}$'
            )
            ax.xaxis.set_major_formatter(ticks)
            ax.yaxis.set_major_formatter(ticks)
            sns.despine()
            ax.set_xlabel('HSC Abundance')
            ax.set_ylabel(cell_type.title() + ' Abundance')
            ax.set_title(
                cell_type.title()
                + ' ' + change_type,
                fontsize=15
                )
    fname = save_path + os.sep \
        + 'hsc_bootstrap_' \
        + cell_type  \
        + inv_desc \
        + '_a' + str(thresholds[cell_type]).replace('.','-') \
        + '_alpha' + str(alpha).replace('.','-') \
        + '_nregs' + str(n_sub_samples) \
        + '_sampratio' + str(sub_sample_amount).replace('.', '-') \
        + '_color-' + hue_col \
        + '.' + save_format

    save_plot(fname, save, save_format)

def plot_hsc_abund_at_exh_lb(
        marked_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        bar_col: str,
        bar_types: List,
        gxd_mice: List,
        save: bool,
        save_path: str,
        save_format: str='png',
    ) -> None:
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 2,
            'axes.linewidth': 4,
            'axes.labelsize': 20,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'ytick.major.size': 8,
            'ytick.minor.width': 0,
            'ytick.minor.size': 0,
            'xtick.labelsize': 15,
            'ytick.labelsize': 20,
            'figure.titlesize': 'small',
        }

        )
    if clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].empty:
        raise ValueError('No HSC Cells in Clonal Abundance Data')
    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']
    labeled_hsc_data = hsc_data.merge(
        marked_df[['code', 'mouse_id', bar_col]].drop_duplicates(),
        how='left',
        validate='m:1'
    )
    print('Showing un labeled hscs')
    print(labeled_hsc_data[labeled_hsc_data[bar_col].isna()])
    labeled_hsc_data.loc[labeled_hsc_data[bar_col].isna(), bar_col] = 'Unknown'

    file_add = ''
    y_desc = 'abundance'
    y_col = 'percent_engraftment'

    if gxd_mice is not None:
        labeled_hsc_data = labeled_hsc_data[labeled_hsc_data.mouse_id.isin(gxd_mice)]
    print(labeled_hsc_data.groupby(['mouse_id', bar_col]).code.nunique())

    hue_order = ['aging_phenotype', 'no_change']
    _, ax = plt.subplots(figsize=(7,5))
    sns.barplot(
        y=y_col,
        x=bar_col,
        hue='group',
        hue_order=hue_order,
        order=bar_types,
        data=labeled_hsc_data,
        palette=COLOR_PALETTES['group'],
        saturation=1,
        capsize=.2,
        ci=68,
        ax=ax,
        zorder=1,
    )
    sns.barplot(
        y=y_col,
        x=bar_col,
        hue='group',
        hue_order=hue_order,
        order=bar_types,
        data=labeled_hsc_data,
        palette=COLOR_PALETTES['group'],
        saturation=1,
        ci=None,
        ax=ax,
        zorder=10
    )
    for k, spine in ax.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(100)
    stat_tests.ranksums_test_group_time(
        data=labeled_hsc_data,
        test_col='percent_engraftment',
        timepoint_col=bar_col,
        overall_context='HSC Abundance',
        show_ns=True,
    )
    for group, g_df in labeled_hsc_data.groupby('group'):
        stat_tests.ranksums_test_group(
            data=g_df,
            test_col='percent_engraftment',
            group_col=bar_col,
            overall_context='HSC Abundance ' + group,
            show_ns=True,
        )
    ax.set(yscale='log')
    ax.set_yticks([10E-1, 10E-2, 10E-3,])
    ax.legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('HSC ' + y_desc.title())
    sns.despine()

    fname = save_path + os.sep \
        + y_desc \
        + '_hsc_abundance-group' \
        + '_' + bar_col \
        + file_add \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_clones_at_time_total_abundance(
    clonal_abundance_df: pd.DataFrame,
    timepoint_col: str, 
    log_scale: bool,
    save: bool = False,
    save_path: str = './output',
    save_format: str = 'png',
):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3.5,
            'axes.linewidth': 3.5,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'medium',
        }

    )

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        False,
    )
    times = clonal_abundance_df.sort_values(by=timepoint_col)[timepoint_col].unique()
    print(times)
    fig, axes = plt.subplots(ncols=len(times), nrows=2, figsize=(9*len(times),8*2))
    plt.subplots_adjust(hspace=.25, wspace=.3)
    ct_row = {'gr': 0, 'b': 1}
    for cell_type in ['gr', 'b']:
        for i, timepoint in enumerate(times):
            ax = axes[ct_row[cell_type]][i]
            ct_df = clonal_abundance_df[clonal_abundance_df.cell_type == cell_type]
            tp_df = agg.get_clones_at_timepoint(
                ct_df,
                timepoint_col,
                timepoint,
                by_mouse=False,
            )
            tp_df = tp_df[tp_df.percent_engraftment > 0.01][[
                'code',
                'mouse_id',
            ]].drop_duplicates()
            only_at_tp = ct_df.merge(
                tp_df,
                how='inner',
                validate='m:1'
            )
            sum_df = pd.DataFrame(
                only_at_tp.groupby(
                    ['mouse_id', timepoint_col, 'group', 'cell_type']
                ).percent_engraftment.sum()
            ).reset_index()
            zero_filled = agg.fill_mouse_id_zeroes(
                sum_df,
                ['group'],
                fill_col='percent_engraftment',
                fill_cat_col=timepoint_col,
                fill_cats=times,
                fill_val=0
            )
            if sum_df.cell_type.nunique() != 1:
                raise ValueError('Cell type not unique')
            if log_scale:
                if cell_type == 'gr':
                    ax.set_yscale('symlog', linthreshy=10e-4)
                else:
                    ax.set_yscale('symlog', linthreshy=10e-3)
            
            hue='group'
            sns.lineplot(
                data=zero_filled,
                x=timepoint_col,
                y='percent_engraftment',
                estimator=None,
                units='mouse_id',
                hue=hue,
                palette=COLOR_PALETTES['group'],
                ax=ax,
                alpha=0.50,
                lw=2,
            )
            sns.lineplot(
                data=zero_filled,
                x=timepoint_col,
                y='percent_engraftment',
                hue=hue,
                palette=['white'] * zero_filled.group.nunique(),
                ax=ax,
                lw=10,
            )
            sns.lineplot(
                data=zero_filled,
                x=timepoint_col,
                y='percent_engraftment',
                hue=hue,
                palette=COLOR_PALETTES[hue],
                ax=ax,
                ci=95,
                lw=6,
                seed=32,
            )
            if timepoint_col == 'month':
                ax.set_xticks([4, 9, 12, 15])

            ax.legend().remove()
            title_str = str(timepoint) + ' ' + cell_type
            ax.set_title(title_str, fontsize=20)
            sns.despine()

    fname = os.path.join(
        save_path,
        'abundance_at_times'
        + 'log_' + str(log_scale)
        + '.' + save_format
    )
    save_plot(fname, save, save_format)

def plot_change_contributions_pie(
        changed_marked_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: Any,
        gfp_donor: pd.DataFrame,
        gfp_donor_thresh: float,
        force_order: bool,
        save: bool = False,
        bar_col: str = 'change_type',
        bar_types: List = ['Lymphoid', 'Unchanged', 'Myeloid', 'Unknown'],
        present_thresh: float = 0.01,
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


    if timepoint_col == 'gen':
        changed_marked_df = changed_marked_df[changed_marked_df.gen != 8.5]

    filt_bar_types = changed_marked_df[changed_marked_df[bar_col].isin(bar_types)]
    if filt_bar_types.shape != changed_marked_df.shape:
        print("Not all types being displayed, missing:")
        print(
            changed_marked_df[~changed_marked_df[bar_col].isin(bar_types)][bar_col].unique()
        )

    percent_of_total = True
    changed_sum_df = agg.sum_abundance_by_change(
        agg.remove_month_17(
            filt_bar_types,
            timepoint_col
        ),
        timepoint_col=timepoint_col,
        percent_of_total=percent_of_total,
        change_col=bar_col,
    )
    changed_sum_df = changed_sum_df[changed_sum_df.cell_type.isin(['gr', 'b'])]
    changed_sum_with_gxd = changed_sum_df.merge(
        gfp_donor,
        how='inner',
        validate='m:1',
        on=['mouse_id', 'cell_type', timepoint_col]
    )

    timepoint_df = agg.get_clones_at_timepoint(
        changed_sum_with_gxd,
        timepoint_col,
        timepoint,
        by_mouse=True,
    )

    print('FILTERING FOR MICE FOUND IN FIRST AND LAST TIMEPOINT ABOVE GFP x DONOR THRESHOLD' )
    mice_left = agg.filter_gxd_first_last(
        changed_sum_with_gxd,
        timepoint_col,
        threshold=gfp_donor_thresh
    )


    if (bar_col == 'survived') and ('M2012' in mice_left):
        print('Excluding M2012')
        mice_left = [m for m in mice_left if m != 'M2012']
    print(mice_left)

    filt_gxd = timepoint_df[timepoint_df.mouse_id.isin(mice_left)]
    print('Pre-filt gxd mice:', timepoint_df.mouse_id.nunique())
    print('Post-filt gxd mice:', filt_gxd.mouse_id.nunique())

    filt_gxd = filt_gxd.assign(total=100)
    filt_gxd = filt_gxd.sort_values(by='percent_engraftment', ascending=False)
    print(filt_gxd.groupby('mouse_id')[timepoint_col].unique())
    agg_df = pd.DataFrame(
        filt_gxd.groupby(['group', 'mouse_id', 'cell_type', bar_col]).percent_engraftment.sum()
    ).reset_index()
    j = 0
    _, axes = plt.subplots(ncols=2, nrows=2, figsize=(16,8))
    for cell_type, ct_df in agg_df.groupby('cell_type'):
        zero_filled = agg.fill_mouse_id_zeroes(
            ct_df,
            ['group'],
            fill_col='percent_engraftment',
            fill_cat_col=bar_col,
            fill_cats=bar_types,
            fill_val=0
        )
        i=0
        for group, g_df in zero_filled.groupby('group'):
            ax = axes[j][i]
            i+=1

            ct_agg = g_df.groupby(bar_col).percent_engraftment.median()
            ct_agg = 100 * ct_agg/ct_agg.sum()
            ct_agg = pd.DataFrame(
                ct_agg
            ).reset_index()
            copy_palette = COLOR_PALETTES
            copy_palette['survived']['Survived'] = 'white'
            colors = [copy_palette[bar_col][bt] for bt in ct_agg[bar_col].values]
            ax.pie(
                ct_agg.percent_engraftment,
                colors=colors,
                wedgeprops = {'linewidth': 1, 'edgecolor':'black'},
            )
            print(cell_type, group)
            print(ct_agg)
            ax.set_title(cell_type + ' ' + group + ' ' + timepoint)

        j+=1
    fname = save_path + os.sep + 'pie_contrib' \
        + '_' + bar_col \
        + '_' + cell_type \
        + '_gxd-' + str(gfp_donor_thresh) \
        + '_' + timepoint_col[0] + '-' + str(timepoint) \
        + '_' + str(present_thresh) \
        + '.' + save_format
    save_plot(fname, save, save_format)

def plot_clones_at_time_total_abundance_heatmap(
    clonal_abundance_df: pd.DataFrame,
    timepoint_col: str, 
    by_average: bool,
    save: bool = False,
    save_path: str = './output',
    save_format: str = 'png',
):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3.5,
            'axes.linewidth': 3.5,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'medium',
        }

    )

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        False,
    )
    agg_func='median'
    if by_average:
        agg_func='mean'
    times = clonal_abundance_df.sort_values(by=timepoint_col)[timepoint_col].unique()
    print(times)
    agg_dfs = []
    for (ct), ct_df in clonal_abundance_df.groupby(['cell_type']):
        if ct not in ['gr', 'b']:
            continue
        for i, timepoint in enumerate(times):
            tp_df = agg.get_clones_at_timepoint(
                ct_df,
                timepoint_col,
                timepoint,
                by_mouse=False,
            )
            tp_df = tp_df[tp_df.percent_engraftment > 0.01][[
                'code',
                'mouse_id',
            ]].drop_duplicates()
            only_at_tp = ct_df.merge(
                tp_df,
                how='inner',
                validate='m:1'
            )
            sum_df = pd.DataFrame(
                only_at_tp.groupby(
                    ['mouse_id', timepoint_col, 'group', 'cell_type']
                ).percent_engraftment.sum()
            ).reset_index()
            zero_filled = agg.fill_mouse_id_zeroes(
                sum_df,
                ['group', 'cell_type'],
                fill_col='percent_engraftment',
                fill_cat_col=timepoint_col,
                fill_cats=times,
                fill_val=0
            )
            zero_filled['time_at'] = timepoint
            agg_dfs.append(zero_filled)

    agg_dfs = pd.concat(agg_dfs)
    for (group, cell_type), a_df in agg_dfs.groupby(['group', 'cell_type']):
        vmin=0
        if cell_type == 'b':
            vmax=28
        else:
            vmax=7
        piv = a_df.pivot_table(
            columns=timepoint_col,
            index='time_at',
            aggfunc=agg_func,
            values='percent_engraftment'
        )
        _, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(
            piv,
            vmin=vmin,
            vmax=vmax,
            cmap='magma_r'
        )
        ax.set_title(
            group + ' ' + cell_type + ' ' + agg_func,
            fontsize=20,
        )

        fname = os.path.join(
            save_path,
            'abundance_at_times'
            + 'by_' + str(agg_func)
            + '_' + group
            + '_' + cell_type
            + '.' + save_format
        )
        save_plot(fname, save, save_format)

def plot_abundance_lineplot(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3.5,
            'axes.linewidth': 3.5,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'medium',
        }

    )
    bias_change_df = agg.calculate_first_last_bias_change(
        lineage_bias_df,
        timepoint_col=timepoint_col,
        by_mouse=False,
    )
    mtd=3

    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = agg.remove_gen_8_5(
        clonal_abundance_df,
        timepoint_col,
        False,
    )
    if timepoint_col == 'gen':
        clonal_abundance_df =agg.filter_mice_with_n_timepoints(
            clonal_abundance_df,
            8,
            timepoint_col,
        )
    elif timepoint_col == 'month':
        print('Filtering for mice with all 4 timepoints')
        clonal_abundance_df =agg.filter_mice_with_n_timepoints(
            clonal_abundance_df,
            4,
            timepoint_col,
        )
    clonal_abundance_df = agg.mark_changed(
        clonal_abundance_df,
        bias_change_df,
        merge_type='left',
        min_time_difference=mtd,
    ).merge(
        lineage_bias_df[
            ['mouse_id', 'code', timepoint_col, 'lineage_bias']
        ].drop_duplicates()
    )
    clonal_abundance_df = agg.add_bias_category(clonal_abundance_df)
    last_myeloid = agg.get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        'last',
        by_mouse=True
    )
    last_myeloid = last_myeloid[
        (last_myeloid.bias_category == 'MB') &
        (last_myeloid.change_type == 'Unchanged')
    ][['code', 'mouse_id']].drop_duplicates()
    

    filts = [
        (
            'anti-aging',
            clonal_abundance_df[
                (clonal_abundance_df.change_type == 'Lymphoid')
            ]
        ),
        (
            'stable myeloid',
            clonal_abundance_df.merge(
                last_myeloid
            )

        )
    ]
    clone_hue = 'group'
    hue = 'group'
    for filt_type, filt_df in filts:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18,8))
        plt.subplots_adjust(hspace=.2, wspace=.3)
        i=0
        for (ct), s_df in filt_df.groupby(['cell_type']):
            if ct not in ['gr', 'b']:
                continue
            print(filt_type, ct)
            print(s_df.groupby(['change_type', 'bias_category']).code.nunique())
            ax = axes.flatten()[i]
            i += 1
            if ct == 'gr':
                ax.set_yscale('symlog', linthreshy=10e-4)
            else:
                ax.set_yscale('symlog', linthreshy=10e-3)
            sns.lineplot(
                data=s_df,
                x=timepoint_col,
                y='percent_engraftment',
                estimator=None,
                units='code',
                hue=clone_hue,
                palette=COLOR_PALETTES[clone_hue],
                ax=ax,
                alpha=0.50,
                lw=2,
            )
            sns.lineplot(
                data=s_df,
                x=timepoint_col,
                y='percent_engraftment',
                hue=hue,
                palette=['white'] * s_df.group.nunique(),
                ax=ax,
                lw=10,
            )
            sns.lineplot(
                data=s_df,
                x=timepoint_col,
                y='percent_engraftment',
                hue=hue,
                palette=COLOR_PALETTES[hue],
                ax=ax,
                ci=95,
                lw=6,
                seed=32,
            )
            if timepoint_col == 'month':
                ax.set_xticks([4, 9, 12, 15])
            ax.legend().remove()
            sns.despine()
            title_str = filt_type + ' ' + ct
            ax.set_title(title_str)
        fname = os.path.join(
            save_path,
            'lineplot_abundance-'
            + filt_type 
            + '.' + save_format
        )
        save_plot(fname, save, save_format)


def plot_activated_clone_bias(
        clonal_abundance_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3.5,
            'axes.linewidth': 3.5,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'medium',
        }

    )
    exhaust_labeled = agg.label_exhausted_clones(
        None,
        agg.remove_gen_8_5(
            agg.remove_month_17_and_6(
                clonal_abundance_df,
                timepoint_col
            ),
            timepoint_col,
            keep_hsc=False
        ),
        timepoint_col,
        present_thresh=0.01,
    )
    last_bias = agg.get_clones_at_timepoint(
        lineage_bias_df,
        timepoint_col,
        'last',
        by_mouse=True,
    )
    marked_df = agg.label_activated_clones(
        exhaust_labeled,
        timepoint_col,
        present_thresh=0.01,
    )
    with_bias_cats = marked_df.merge(
        agg.add_bias_category(
            last_bias
        )[['code','bias_category']].drop_duplicates()
    )
    if timepoint_col == 'month':
        gxd_mice = ['M2059', 'M3010', 'M3013', 'M3016', 'M190', 'M2061', 'M3000', 'M3012', 'M3001', 'M3009', 'M3018', 'M3028']
        print(Fore.YELLOW + 'KEEPING GXD MICE ONLY', gxd_mice)
        with_bias_cats = with_bias_cats[with_bias_cats.mouse_id.isin(gxd_mice)]

    agg_df = with_bias_cats[
        with_bias_cats.survived == 'Activated'
    ].groupby([
        'mouse_id',
        'group',
        'bias_category'
    ]).code.nunique()
    agg_df = pd.DataFrame(agg_df).reset_index()
    agg_df = agg.fill_mouse_id_zeroes(
        agg_df,
        ['group'],
        'code',
        'bias_category',
        ['MB', 'B', 'LB'],
        0,
    )
    fig, ax = plt.subplots(figsize=(8,8))
    stripplot_mouse_markers_with_mean(
        agg_df,
        'bias_category',
        'code',
        ax,
        'group',
        ['aging_phenotype', 'no_change'],
        size=20
    )
    stat_tests.ind_ttest_group_time(
        data=agg_df,
        test_col='code',
        timepoint_col='bias_category',
        overall_context='Activated Clones',
        show_ns=True
    )
    stat_tests.rel_ttest_group_time(
        data=agg_df,
        merge_type='inner',
        match_cols=['mouse_id', 'group'],
        fill_na=0,
        test_col='code',
        timepoint_col='bias_category',
        overall_context='Activated Clones',
        show_ns=True
    )

    fname = os.path.join(
        save_path,
        'activated_clone_bias'
        + '.' + save_format
    )
    save_plot(fname, save, save_format)

def plot_not_first_last_abundance(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        save: bool = False,
        save_path: str = './output',
        save_format: str = 'png',
    ):
    sns.set_context(
        'paper',
        rc={
            'lines.linewidth': 3.5,
            'axes.linewidth': 3.5,
            'axes.labelsize': 30,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'figure.titlesize': 'medium',
        }

    )
    clonal_abundance_df = agg.remove_month_17_and_6(
        clonal_abundance_df,
        timepoint_col
    )
    clonal_abundance_df = clonal_abundance_df[
        clonal_abundance_df.cell_type.isin(['gr','b'])
    ]
    n_timepoints = clonal_abundance_df[timepoint_col].nunique()
    clonal_abundance_df = agg.filter_mice_with_n_timepoints(
        clonal_abundance_df,
        n_timepoints=n_timepoints,
        time_col=timepoint_col
    )
    first = agg.get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        'first',
        by_mouse=True
    )
    last = agg.get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        'last',
        by_mouse=True
    )
    first['time'] = 'first'
    last['time'] = 'last'
    fl_marked = clonal_abundance_df.merge(
        pd.concat([first, last])[['code', 'cell_type', 'mouse_id', 'time', timepoint_col]].drop_duplicates(),
        how='left',
        validate='1:1'
    )
    fl_marked.loc[fl_marked.time.isna(), 'time'] = 'Other'
    piv = fl_marked.pivot_table(
        columns='time',
        index=['mouse_id', 'group', 'code'],
        values='percent_engraftment',
        aggfunc='max'
    ).fillna(0)
    not_fl = piv[
        (piv['first'] < 0.01) & 
        (piv['last'] < 0.01)
    ].reset_index()[['mouse_id', 'group', 'code']]
    not_fl_abund = pd.DataFrame(
        clonal_abundance_df.merge(
            not_fl,
            how='inner',
            validate='m:1'
        ).groupby(['mouse_id', 'cell_type', 'group', timepoint_col])\
            .percent_engraftment.sum()
    ).reset_index()
    total_clones = pd.DataFrame(
        clonal_abundance_df.groupby(['mouse_id', 'cell_type', 
        timepoint_col]).percent_engraftment.sum()
    ).reset_index().rename(columns={'percent_engraftment': 'total'})
    not_fl_abund = not_fl_abund.merge(
        total_clones,
        how='right',
    )
    not_fl_abund['perc_abund'] = 100 * not_fl_abund['percent_engraftment'] / not_fl_abund['total']
    fig, axes = plt.subplots(ncols=2, figsize=(1.5*n_timepoints*2.2, 5))
    plt.subplots_adjust(wspace=0.35)
    hue_col='group'
    hue_order=['aging_phenotype', 'no_change']
    for i, (ct, ct_df) in enumerate(not_fl_abund.groupby('cell_type')):
        ax = axes.flatten()[i]
        stripplot_mouse_markers_with_mean(
            ct_df,
            timepoint_col,
            'perc_abund',
            ax,
        )
        ax.set_ylabel('Abundance (%)')
        ax.set_title(ct, fontsize=15)
        for group, g_df in ct_df.groupby('group'):
            stat_tests.one_way_ANOVArm(
                data=g_df,
                timepoint_col=timepoint_col,
                id_col='mouse_id',
                value_col='perc_abund',
                overall_context=ct + ' ' + group,
                show_ns=True,
                match_cols=['mouse_id'],
                merge_type='inner',
                fill_na=0,
            )
        stat_tests.ind_ttest_between_groups_at_each_time(
            ct_df,
            'perc_abund',
            timepoint_col,
            ct,
            show_ns=True,
            group_col='group'
        )

    fname = save_path + os.sep \
        + 'not_first_last_abundance' \
        + '.' + save_format
    save_plot(fname, save, save_format)



