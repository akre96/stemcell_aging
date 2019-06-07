""" Create plots from step 7 output data

Example:
python plot_data.py \
    -i ~/Data/serial_transplant_data/M_allAniaAnia\ serial\ transpl_percent-engraftment_121018_long.csv \
    -o ~/Data/serial_transplant_data/Graphs \
    -l ~/Data/serial_transplant_data/lineage_bias/lineage_bias_t0-0_from-counts.csv \
    -c ~/Data/serial_transplant_data/lineage_bias/bias_change_t0-0_from-counts.csv \
    -g clone_count_bar \
    -d --line
Returns:
    None - Shows plots
"""

import argparse
import json
import glob
import os
import sys
from colorama import init, Fore, Back, Style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_types import y_col_type, timepoint_type, change_type
from lineage_bias import get_bias_change, parse_wbc_count_file
from aggregate_functions import filter_threshold, \
    clones_enriched_at_last_timepoint, percentile_sum_engraftment, \
    find_top_percentile_threshold, find_clones_bias_range_at_time, \
    filter_cell_type_threshold, combine_enriched_clones_at_time, \
    mark_changed, sum_abundance_by_change, between_gen_bias_change, \
    calculate_thresholds_sum_abundance, filter_lineage_bias_threshold, \
    across_gen_bias_change, day_to_month, \
    day_to_gen, calculate_bias_change
from plotting_functions import *

     


    
def main():
    """ Create plots set options via command line arguments

    Available graph types:

    default:                Subject to change based on what is being actively developed
    cluster:                Clustered heatmap of present clone engraftment
    venn:                   Venn Diagram of clone existance at timepoint
    clone_count_bar:        Bar charts of clone counts by cell type at different thresholds
    clone_count_line:       Lineplots of clone counts by cell type, mice, and average at different thresholds
    lineage_bias_line:      lineplots of lineage bias over time at different abundance from last timepoint
    top_perc_bias:          line plot of lineage bias over time with top percentile of clones by abundance during last time point
                            options -- value [0-1] indicating percentile to filter for (.95 => 95th percentile)
    engraftment_time:       lineplot/swarmplot of abundance of clones with high values at 4, 12, and 14 months
                            options -- value [0-100] percent cumulative abundance to use as threshold. If not set. uses 99.5th percentile
    counts_at_perc:         line or barplot of clone counts where cell-types are filtered at 90th percentile of abundance
    perc_bias_month:        lineplot of clones in top abundandance percentile at a specific month. Plots lineage bias.
    bias_time_abund:        3d plot of lineage bias vs time vs abundance in b and gr cells
    max_engraftment:        point plot of maximum engraftment averaged across mice by phenotype groups and all mice
    max_eng_mouse:          lineplot of maximum engraftment per mouse
    max_eng_group:          average of maximum engraftment in mice per group for each cell type
    bias_change_dist:       distribution (histogram + rugplot + kde) of change in lineage bias across thresholds
    bias_change_cutoff:     KDE distribution of change in lineage bias across thresholds annotated with recommended cutoff for change
    bias_violin:            Violin plot of lineage bias default split by group
                            options -- group 'all' (default), 'no_change', or 'aging_phenotype'
    range_bias_month:       plots a 3d plot (lineage bias vs abundance vs time) and violin plot (lineage bias vs time)
                            for cells within a specified lineage bias range at a specific month
    sum_abundance:          Cumulative abundance of cell types at increasing percentile of cell population ranked by percent_engraftment
    contrib_change_cell:    Contribution of changed cells vs not changed cells grouped by cell type i.e can great 'all', 'aging_phenotype', and 'no_change' graphs
                            options -- group 'all' (default), 'no_change', or 'aging_phenotype'
    contrib_change_group:   Contribution of changed cells as percent of total cell type. One graph per cell type, grouped by phenotype group
                            options -- 'line', 'bar'
    bias_eng_hist:          Weighted Histogram of lineage bias, weights by count and abundance. Done in Gr and B at specified timepoint
                            options -- integer, month number --> one of 4, 9, 12, or 14
    sum_abund_counts:       Count clones passing thresholds calculated by cumulative abundance based thresholds. Manually set Line to true of false for barchart or lineplot
                            options -- abundance_cutoff, defaults to 50%, range [0-100]
    avg_abund_by_sum:       Lineplot of average abundance of clones above threshold selected based on cumulative abundance
                            options -- abundance_cutoff, defaults to 50%, range [0-100]
        

    """
    parser = argparse.ArgumentParser(description="Plot input data")
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing long format step7 output', default='~/Data/stemcell_aging/Ania_M_allAnia_percent-engraftment_052219_long.csv')
    parser.add_argument('-r', '--rest', dest='rest_of_clones', help='Path to folder containing data on "rest of clones" abundnace and bias', default='~/Data/stemcell_aging/rest_of_clones')
    parser.add_argument('-l', '--lineage-bias', dest='lineage_bias', help='Path to csv containing lineage bias data', default='/home/sakre/Data/stemcell_aging/lineage_bias/lineage_bias_t0-0_from-counts.csv')
    parser.add_argument('-c', '--count', dest='cell_count', help='Path to txt containing FACS cell count data', default='/home/sakre/Data/stemcell_aging/OT_2.0_WBCs_051818-modified.txt')
    parser.add_argument('--bias-change', dest='bias_change', help='Path to csv containing lineage bias change', default='/home/sakre/Data/stemcell_aging/lineage_bias/bias_change_t0-0_from-counts.csv')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='/home/sakre/Data/stemcell_aging/Graphs')
    parser.add_argument('-s', '--save', dest='save', help='Set flag if you want to save output graphs', action="store_true")
    parser.add_argument('-g', '--graph', dest='graph_type', help='Type of graph to output', default='default')
    parser.add_argument('-p', '--options', dest='options', help='Graph Options', default='default')
    parser.add_argument('-d', '--by-day', dest='by_day', help='Plotting done on a day by day basis', action="store_true")
    parser.add_argument('-t', '--threshold', dest='threshold', help='Set threshold for filtering', type=float, required=False)
    parser.add_argument('-a', '--abundance-cutoff', dest='abundance_cutoff', help='Set threshold based on abundance cutoff', type=float, required=False)
    parser.add_argument('-b', '--bias-cutoff', dest='bias_cutoff', help='Cutoff for extreme bias', type=float, required=False)
    parser.add_argument('--invert', dest='invert', help='Invert the selection being done while filtering', action='store_true')
    parser.add_argument('-f', '--filter-bias-abund', dest='filter_bias_abund', help='Abundance threshold to filter lineage bias data', type=float, required=False, default=0.05)
    parser.add_argument('--group', dest='group', help='Set group to inspect', type=str, required=False, default='all')
    parser.add_argument('--time-change', dest='time_change', help='Set time change to across or between for certain graphs', type=str, required=False, default='between')
    parser.add_argument('--timepoint', dest='timepoint', help='Set timepoint to inspect for certain graphs', type=timepoint_type, required=False)
    parser.add_argument('--change-type', dest='change_type', help='Set type of lineage bias changed clones to inspect for certain graphs', type=change_type, required=False)
    parser.add_argument('--line', dest='line', help='Wether to use lineplot for certain graphs', action="store_true")
    parser.add_argument('--by-group', dest='by_group', help='Whether to plot vs group istead of vs cell_type for certain graphs', action="store_true")
    parser.add_argument('--sum', dest='sum', help='Whether to plot sum abundance vs average abundance for certain graphs', action="store_true")
    parser.add_argument('--by-clone', dest='by_clone', help='Whether to plot clone color instead of group for certain graphs', action="store_true")
    parser.add_argument('--by-mouse', dest='by_mouse', help='Whether to plot mouse color instead of group for certain graphs', action="store_true")
    parser.add_argument('--plot-rest', dest='plot_rest', help='Whether to plot rest of clones instead of tracked clones', action="store_true")
    parser.add_argument('--by-gen', dest='by_gen', help='Plotting done on a generation by generation basis', action="store_true")
    parser.add_argument('--limit-gen', dest='limit_gen', help='Limit Serial Transplant data to first 3 generations', action="store_true")
    parser.add_argument('--magnitude', dest='magnitude', help='Plot change in magnitude', action="store_true")
    parser.add_argument('--cache', dest='cache', help='Use Cached Data', action="store_true")
    parser.add_argument('--cache-dir', dest='cache_dir', help='Where cache data is stored', default='/home/sakre/Data/cache')
    parser.add_argument('-y', '--y-col', dest='y_col', help='Which column to plot as y-axis for certain plots', required=False, default='lineage_bias', type=y_col_type)

    # Init colorama
    init(autoreset=True)

    args = parser.parse_args()
    options = args.options
    input_df = pd.read_csv(args.input)
    lineage_bias_df = pd.read_csv(args.lineage_bias)
    bias_change_df = pd.read_csv(args.bias_change)

    analysed_cell_types = ['gr', 'b']
    phenotypic_groups = ['aging_phenotype', 'no_change']
    cell_count_df = parse_wbc_count_file(args.cell_count, ['gr', 'b', 'wbc'])


    if not input_df[~input_df.group.isin(phenotypic_groups)].empty:
        print(Style.BRIGHT + Fore.YELLOW+ '\n !! Warning: Following Mice not in a phenotypic group !!')
        print(Fore.YELLOW+ '  Mouse ID(s): ' + ', '.join(input_df[~input_df.group.isin(phenotypic_groups)].mouse_id.unique()))

    present_clones_df = input_df

    graph_type = args.graph_type
    if graph_type == 'default':
        print(Style.BRIGHT + '\n -- Plotting Default Plot(s) -- \n')
    else:
        print(Style.BRIGHT + '\n -- Graph Type: ' + graph_type + ' -- \n')
    
    if options != 'default':
        print(Style.BRIGHT + ' -- Extra Options Set: ' + options + ' -- \n')


    if args.group != 'all':
        print(' - Group Filtering Set to: ' + args.group)
    if args.by_group:
        print(' - Plotting Each Phenotypic Group Individually')
    
    if args.y_col:
        print(' - Plotting y_axis as: ' + args.y_col)

    if args.invert:
        print(' - Invert Bias Filtering set')

    if args.by_clone:
        print(' - Plotting by clone set')

    if args.magnitude:
        print(' - Plot Magnitude set')

    if args.threshold:
        print(' - Threshold set to: ' + str(args.threshold))
    
    if args.filter_bias_abund:
        print(' - Lineage Bias Min Abundance set to: ' + str(args.filter_bias_abund))
        lineage_bias_df = lineage_bias_df[
            (lineage_bias_df.gr_percent_engraftment >= args.filter_bias_abund) \
            | (lineage_bias_df.b_percent_engraftment >= args.filter_bias_abund)
        ]

    if args.cache:
        print(' - Using Cached Data')
    

    rest_of_clones_abundance_df = pd.read_csv(args.rest_of_clones + os.sep + 'rest_of_clones_abundance_long.csv')

    rest_of_clones_bias_df = pd.read_csv(args.rest_of_clones + os.sep + 'rest_of_clones_lineage_bias.csv')

    color_palettes = json.load(open('color_palettes.json', 'r'))

    if args.by_day:
        print(' - Time By Day Set \n')
        first_timepoint = present_clones_df.day.min()
        timepoint_col = 'day'
    # Adds generation calculation for serial transplant data
    elif args.by_gen:
        print(' - Time By Generation Set \n')


        first_timepoint = 1
        timepoint_col = 'gen'
        lineage_bias_df = lineage_bias_df.assign(gen=lambda x: day_to_gen(x.day))
        input_df = input_df.assign(gen=lambda x: day_to_gen(x.day))
        present_clones_df = present_clones_df.assign(gen=lambda x: day_to_gen(x.day))
        cell_count_df = cell_count_df.assign(gen=lambda x: day_to_gen(x.day))
        rest_of_clones_abundance_df = rest_of_clones_abundance_df.assign(gen=lambda x: day_to_gen(x.day))
        rest_of_clones_bias_df = rest_of_clones_bias_df.assign(gen=lambda x: day_to_gen(x.day))
        if args.limit_gen:
            print(' - ONLY CONSIDERING GENERATIONS 1-3 \n')
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen <= 3]
            input_df = input_df[input_df.gen <= 3]
            present_clones_df = present_clones_df[present_clones_df.gen <= 3]
            cell_count_df = cell_count_df[cell_count_df.gen <= 3]
            input_df = input_df[input_df.gen <= 3]
            rest_of_clones_abundance_df = rest_of_clones_abundance_df[rest_of_clones_abundance_df.gen <= 3]
            rest_of_clones_bias_df = rest_of_clones_bias_df[rest_of_clones_bias_df.gen <= 3]
    else:
        print(' - Time By Month Set \n')
        first_timepoint = 4
        timepoint_col = 'month'
        lineage_bias_df = lineage_bias_df.assign(month=lambda x: day_to_month(x.day))
        input_df = input_df.assign(month=lambda x: day_to_month(x.day))
        present_clones_df = present_clones_df.assign(month=lambda x: day_to_month(x.day))
        cell_count_df = cell_count_df.assign(month=lambda x: day_to_month(x.day))
        rest_of_clones_abundance_df = rest_of_clones_abundance_df.assign(month=lambda x: day_to_month(x.day))
        rest_of_clones_bias_df = rest_of_clones_bias_df.assign(month=lambda x: day_to_month(x.day))
    print('Aging Phenotype Mice: ' + str(input_df[input_df.group == 'aging_phenotype'].mouse_id.nunique()))
    print('No Change Mice: ' + str(input_df[input_df.group == 'no_change'].mouse_id.nunique())  + '\n')

    if graph_type in ['exhausted_clone_hsc_abund']:
        save_path = args.output_dir + os.sep + 'exhausted_clone_hsc_abund'
        exhausted_clone_hsc_abund(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            group=args.group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['hsc_to_ct_compare_svm']:
        save_path = args.output_dir + os.sep + 'hsc_to_ct_compare'

        abundance_cutoff = .01
        thresholds = {'gr': 0.01, 'b': 0.01, 'hsc': 0.01}

        if options != 'default':
            n_clusters = int(options)
        else:
            n_clusters = 2

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=['gr', 'b', 'hsc']
            )
        for cell_type in ['gr', 'b']:
            hsc_to_ct_compare_svm(
                present_clones_df,
                timepoint_col,
                thresholds,
                abundance_cutoff=abundance_cutoff,
                invert=args.invert,
                n_clusters=n_clusters,
                cell_type=cell_type,
                by_mouse=args.by_mouse,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
    if graph_type in ['dist_bias_time_vs_group']:
        save_path = args.output_dir + os.sep + 'dist_bias_time_vs_group'
        bins = 20
        if options != 'default':
            bins = int(options)

        for cell_type in ['gr', 'b', 'sum']:
            plot_dist_bias_at_time_vs_group(
                lineage_bias_df,
                timepoint_col,
                bins=bins,
                cell_type=cell_type,
                change_type=args.change_type,
                timepoint=args.timepoint,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )

    if graph_type in ['exhaust_persist_hsc_abund']:
        save_path = args.output_dir + os.sep + 'exhaust_persist_hsc_abund' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        exhaust_persist_hsc_abund(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['hsc_blood_heatmap_over_time']:
        save_path = args.output_dir + os.sep + 'hsc_blood_prod_over_time'
        heatmap_correlation_hsc_ct(
            present_clones_df,
            timepoint_col,
            by_mouse=args.by_mouse,
            group=args.group,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['hsc_blood_prod_over_time']:
        save_path = args.output_dir + os.sep + 'hsc_blood_prod_over_time'

        hsc_blood_prod_over_time(
            present_clones_df,
            timepoint_col,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['hsc_to_ct_compare_pyod']:
        save_path = args.output_dir + os.sep + 'hsc_to_ct_compare'

        abundance_cutoff = .01
        thresholds = {'gr': 0.01, 'b': 0.01, 'hsc': 0.01}

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=['gr', 'b', 'hsc']
            )
        for cell_type in ['gr', 'b']:
            hsc_to_ct_compare_pyod(
                present_clones_df,
                timepoint_col,
                thresholds,
                abundance_cutoff=abundance_cutoff,
                invert=args.invert,
                cell_type=cell_type,
                by_mouse=args.by_mouse,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
    if graph_type in ['hsc_to_ct_compare_outlier']:
        save_path = args.output_dir + os.sep + 'hsc_to_ct_compare'

        abundance_cutoff = .01
        thresholds = {'gr': 0.01, 'b': 0.01, 'hsc': 0.01}

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=['gr', 'b', 'hsc']
            )
        for cell_type in ['gr', 'b']:
            hsc_to_ct_compare_outlier(
                present_clones_df,
                timepoint_col,
                thresholds,
                abundance_cutoff=abundance_cutoff,
                invert=args.invert,
                cell_type=cell_type,
                by_mouse=args.by_mouse,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )

    if graph_type in ['hsc_to_ct_compare']:
        save_path = args.output_dir + os.sep + 'hsc_to_ct_compare'

        abundance_cutoff = .01
        thresholds = {'gr': 0.01, 'b': 0.01, 'hsc': 0.01}

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=['gr', 'b', 'hsc']
            )
        hsc_to_ct_compare(
            present_clones_df,
            timepoint_col,
            thresholds,
            abundance_cutoff=abundance_cutoff,
            invert=args.invert,
            cell_type='gr',
            by_mouse=args.by_mouse,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
        hsc_to_ct_compare(
            present_clones_df,
            timepoint_col,
            thresholds,
            abundance_cutoff=abundance_cutoff,
            invert=args.invert,
            by_mouse=args.by_mouse,
            cell_type='b',
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['hsc_mouse_pie']:
        save_path = args.output_dir + os.sep + 'hsc_mouse_pie'
        plot_hsc_pie_mouse(
            present_clones_df,
            timepoint_col,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['contrib_by_bias_cat']:
        save_path = args.output_dir + os.sep + 'contrib_by_bias_cat' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        for cell_type in ['gr', 'b']:
            plot_contribution_by_bias_cat(
                lineage_bias_df,
                timepoint_col,
                cell_type,
                by_sum=args.sum,
                by_group=args.by_group,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
        

    if graph_type in ['exhausted_lymphoid_at_time']:
        save_path = args.output_dir + os.sep + 'exhausted_lymphoid_at_time' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        timepoint = first_timepoint
        if args.timepoint:
            timepoint = args.timepoint

        plot_exhausted_lymphoid_at_time(
            lineage_bias_df,
            clonal_abundance_df=present_clones_df,
            timepoint_col=timepoint_col,
            timepoint=timepoint,
            y_col=args.y_col,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )

    if graph_type in ['exhausted_abund_at_time']:
        save_path = args.output_dir + os.sep + 'exhausted_abundance_at_time' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]
        plot_not_survived_abundance_at_time(
            lineage_bias_df,
            timepoint_col,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['bias_first_last_abund']:
        save_path = args.output_dir + os.sep + 'bias_first_last_abund'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        abundance_cutoff = 0.01
        thresholds = {'gr': 0.01, 'b': 0.01}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
            
        plot_swarm_violin_first_last_bias(
            lineage_bias_df,
            timepoint_col,
            thresholds,
            abundance_cutoff=abundance_cutoff,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['clone_count_first_last']:
        save_path = args.output_dir + os.sep + 'clone_count_bar'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        abundance_cutoff = 0.01
        thresholds = {'gr': 0.01, 'b': 0.01}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
            
        plot_clone_count_bar_first_last(
            present_clones_df,
            timepoint_col,
            thresholds,
            abundance_cutoff=abundance_cutoff,
            analyzed_cell_types=list(thresholds.keys()),
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['clone_count_swarm_vs_ct']:
        save_path = args.output_dir + os.sep + 'clone_count_swarm_vs_cell-type'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        abundance_cutoff = 0.01
        thresholds = {'gr': 0.01, 'b': 0.01}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
            
        plot_clone_count_swarm_vs_cell_type(
            present_clones_df,
            timepoint_col,
            thresholds,
            abundance_cutoff=abundance_cutoff,
            analyzed_cell_types=['gr', 'b'],
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['clone_count_swarm']:
        save_path = args.output_dir + os.sep + 'clone_count_swarm'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        abundance_cutoff = 0.01
        thresholds = {'gr': 0.01, 'b': 0.01}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
            
        plot_clone_count_swarm(
            present_clones_df,
            timepoint_col,
            thresholds,
            abundance_cutoff=abundance_cutoff,
            analyzed_cell_types=list(thresholds.keys()),
            line=args.line,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['n_most_abund']:
        save_path = args.output_dir + os.sep + 'n_most_abund_contrib'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        n = 5
        if options != 'default':
            n = int(options)

        plot_n_most_abundant(
            present_clones_df,
            timepoint_col,
            n,
            save=args.save,
            save_path=save_path
        )
        

    if graph_type in ['bias_dist_contrib_over_time']:
        save_path = args.output_dir + os.sep + 'bias_distribution_contribution_over_time'
        for cell_type in ['gr', 'b']:
            plot_bias_dist_contribution_over_time(
                lineage_bias_df,
                timepoint_col,
                cell_type,
                by_group=args.by_group,
                save=args.save,
                save_path=save_path,
            )

    if graph_type in ['abundance_by_change']:
        save_path = args.output_dir + os.sep + 'abundance_by_change' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)

        plot_abundance_by_change(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            mtd,
            args.timepoint,
            sum=args.sum,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['bias_dist_by_change']:
        save_path = args.output_dir + os.sep + 'bias_dist_by_change' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)

        plot_bias_dist_by_change(
            lineage_bias_df,
            timepoint_col,
            mtd,
            args.group,
            args.timepoint,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['perc_survival_bias_heatmap']:
        save_path = args.output_dir + os.sep + 'perc_survival_bias_type' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        plot_perc_survival_bias_heatmap(
            lineage_bias_df,
            timepoint_col,
            by_clone=args.by_clone,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['perc_survival_bias_type']:
        save_path = args.output_dir + os.sep + 'perc_survival_bias_type' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        plot_perc_survival_bias(
            lineage_bias_df,
            timepoint_col,
            by_clone=args.by_clone,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['bias_stable_abundant_at_time']:
        save_path = args.output_dir + os.sep + 'stable_abund_time' \
            + os.sep + args.y_col + '_lbf' + str(args.filter_bias_abund).replace('.', '-')

        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        timepoint = 'last'
        if args.timepoint:
            timepoint = args.timepoint
    
        for cell_type in ['gr', 'b']:
            plot_stable_abund_time_clones(
                lineage_bias_df,
                present_clones_df,
                args.bias_cutoff,
                abund_timepoint=timepoint,
                t1=first_timepoint,
                timepoint_col=timepoint_col,
                thresholds=thresholds,
                cell_type=cell_type,
                y_col=args.y_col,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )

    if graph_type in ['changed_status_overtime']:
        save_path = args.output_dir + os.sep + 'change_status' \
            + os.sep + args.y_col + '_lbf' + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)

        plot_change_marked(
            lineage_bias_df,
            input_df,
            timepoint_col,
            mtd,
            args.y_col,
            args.by_clone,
            save=args.save,
            save_path=save_path
        )
        


    if graph_type in ['hsc_abund_bias_last']:
        save_path = args.output_dir + os.sep + 'hsc_abund_bias_last' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        plot_hsc_abund_bias_at_last(
            lineage_bias_df,
            input_df,
            timepoint_col,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['not_survived_count_box']:
        save_path = args.output_dir + os.sep + 'not_survived_count_box' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]
        plot_not_survived_count_box(
            lineage_bias_df,
            timepoint_col,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['not_survived_abund']:
        save_path = args.output_dir + os.sep + 'not_survived_abundance' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]
        plot_not_survived_abundance(
            lineage_bias_df,
            timepoint_col,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['not_survived_bias']:
        save_path = args.output_dir + os.sep + 'not_survived_bias'
        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]
        if args.by_mouse:
            plot_not_survived_count_mouse(
                lineage_bias_df,
                timepoint_col,
                save=args.save,
                save_path=save_path
            )
        else:
            for group in phenotypic_groups + ['all']:
                plot_not_survived_by_bias(
                    lineage_bias_df,
                    timepoint_col,
                    group=group,
                    save=args.save,
                    save_path=save_path
                )

    if graph_type in ['survival_time_change']:
        save_path = args.output_dir + os.sep + 'survival_time_change'
        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
        if args.time_change == 'across':
            cumulative = True
        elif args.time_change == 'between':
            cumulative = False
        for cell_type in ['gr', 'b']:
            plot_abundant_clone_survival(
                present_clones_df,
                timepoint_col,
                thresholds,
                cell_type,
                cumulative,
                by_mouse=args.by_mouse,
                save=args.save,
                save_path=save_path
            )

    if graph_type in ['bias_first_last']:
        save_path = args.output_dir + os.sep + 'bias_first-last'
        if args.threshold:
            threshold = args.threshold
        else:
            threshold = 0.0
        if args.y_col == 'lineage_bias':
            raise ValueError("Y-Col must be 'sum_abundance', 'gr_percent_engraftment', or 'b_percent_engraftment")
        plot_bias_first_last(
            lineage_bias_df,
            timepoint_col,
            filter_col=args.y_col,
            cutoff=threshold,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['bias_dist_at_time']:
        save_path = args.output_dir + os.sep + 'bias_distribution_at_time'
        timepoint = first_timepoint
        if args.timepoint:
            timepoint = args.timepoint

        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
        plot_dist_bias_at_time(
            lineage_bias_df,
            timepoint_col,
            timepoint,
            thresholds=thresholds,
            abundance_cutoff=abundance_cutoff,
            by_mouse=args.by_mouse,
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['bias_dist_over_time']:
        save_path = args.output_dir + os.sep + 'bias_distribution_over_time'
        plot_dist_bias_over_time(
            lineage_bias_df,
            timepoint_col,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['bias_change_mean_scatter']:
        save_path = args.output_dir + os.sep + 'bias_distribution_mean_scatter'
        plot_bias_change_mean_scatter(
            lineage_bias_df,
            timepoint_col,
            y_col='gr_percent_engraftment',
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
        )
        plot_bias_change_mean_scatter(
            lineage_bias_df,
            timepoint_col,
            y_col='sum_abundance',
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
        )
        plot_bias_change_mean_scatter(
            lineage_bias_df,
            timepoint_col,
            by_group=args.by_group,
            y_col='b_percent_engraftment',
            save=args.save,
            save_path=save_path,
        )
    if graph_type in ['bias_change_mean_dist_vs_group']:
        save_path = args.output_dir + os.sep + 'bias_distribution_mean_abund_vs_g'
        if args.threshold:
            threshold = args.threshold
        else:
            threshold = 0.01

        if args.options != 'default':
            mtd = int(args.options)
        else:
            mtd = 1
        ## PLots unused, uncomment to filter based on b and gr average engraftment
        #plot_bias_dist_mean_abund_group_vs(
            #lineage_bias_df,
            #timepoint_col,
            #cutoff=threshold,
            #y_col='b_percent_engraftment',
            #save=args.save,
            #save_path=save_path,
        #)
        #plot_bias_dist_mean_abund_group_vs(
            #lineage_bias_df,
            #timepoint_col,
            #cutoff=threshold,
            #y_col='gr_percent_engraftment',
            #save=args.save,
            #save_path=save_path,
        #)

        plot_bias_dist_mean_abund_group_vs(
            lineage_bias_df,
            timepoint_col,
            cutoff=threshold,
            mtd=mtd,
            timepoint=args.timepoint,
            y_col='sum_abundance',
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['abund_swarm_time']:
        save_path = args.output_dir + os.sep + 'abund_at_first_timepoint'
        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
        first_timepoint_df = present_clones_df[present_clones_df[timepoint_col] == first_timepoint]
        filt_df = filter_cell_type_threshold(
            first_timepoint_df,
            thresholds=thresholds,
            analyzed_cell_types=['gr', 'b'],
        )
        plot_abund_swarm_box(
            filt_df,
            thresholds,
            save=args.save,
            save_path=save_path,
        )

        
    # Plots distribution of change in bias from a clones first to last timepoint
    #    Each line is the result of filtering the above based on cutoffs of abundance
    #    Plots 1 figure for filters on b, gr, and combined abundance
    if graph_type in ['bias_change_mean_abund_dist']:
        save_path = args.output_dir + os.sep + 'bias_distribution_mean_abund'
        abundance_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1]
        plot_bias_dist_mean_abund(
            lineage_bias_df,
            timepoint_col,
            cutoffs=abundance_thresholds,
            y_col='b_percent_engraftment',
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
        )
        abundance_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
        plot_bias_dist_mean_abund(
            lineage_bias_df,
            timepoint_col,
            cutoffs=abundance_thresholds,
            y_col='gr_percent_engraftment',
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
        )
        abundance_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1]
        plot_bias_dist_mean_abund(
            lineage_bias_df,
            timepoint_col,
            cutoffs=abundance_thresholds,
            y_col='sum_abundance',
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
        )




    if graph_type in ['bias_dist_abund']:
        save_path = args.output_dir + os.sep + 'bias_distribution_time'
        abundance_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        plot_bias_dist_at_time(
            lineage_bias_df,
            abundance_thresholds,
            timepoint_col,
            group=args.group,
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['extreme_bias_time']:
        save_path = args.output_dir + os.sep \
            + 'extreme_bias_time' + os.sep \
            + str(args.filter_bias_abund).replace('.', '-')

        bias_cutoff = .9
        if args.bias_cutoff:
            bias_cutoff = args.bias_cutoff
        
        
        # Timepoint defaults to 4 months
        timepoint = first_timepoint
        if args.timepoint:
            timepoint = args.timepoint
        
        y_col = 'lineage_bias'
        if args.y_col:
            y_col = args.y_col

        if args.invert:
            plot_extreme_bias_time(
                lineage_bias_df,
                present_clones_df,
                timepoint_col,
                timepoint,
                y_col,
                bias_cutoff,
                invert_selection=args.invert,
                by_clone=args.by_clone,
                save=args.save,
                save_path=save_path,
            )
        else:
            for cutoff in [bias_cutoff, -1 * bias_cutoff]:
                plot_extreme_bias_time(
                    lineage_bias_df,
                    present_clones_df,
                    timepoint_col,
                    timepoint,
                    y_col,
                    cutoff,
                    by_clone=args.by_clone,
                    save=args.save,
                    save_path=save_path,
                )

    if graph_type in ['extreme_bias_abund']:
        save_path = args.output_dir + os.sep + 'extreme_bias_abundance'
        plot_extreme_bias_abundance(
            lineage_bias_df,
            timepoint_col,
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['stable_bias']:
        bias_change_cutoff = args.bias_cutoff
        timepoints = lineage_bias_df[timepoint_col].unique()
        timepoints.sort()
        print(' - Bias Change Cutoff Set to: ' + str(bias_change_cutoff))

        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
        plot_stable_clones(
            lineage_bias_df,
            bias_change_cutoff,
            t1=first_timepoint,
            timepoint_col=timepoint_col,
            clonal_abundance_df=present_clones_df,
            thresholds=thresholds,
            y_col=args.y_col,
            save=args.save,
            save_path=args.output_dir + os.sep + 'Stable_Lineage_Bias_Line_Plot',
            save_format='png',
        )

    if graph_type in ['rest_vs_tracked']:
        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        plot_col = 'lineage_bias'
        if args.y_col:
           plot_col = args.y_col

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
        plot_rest_vs_tracked(
                lineage_bias_df,
                rest_of_clones_bias_df,
                cell_count_df,
                y_col=plot_col,
                abundance_cutoff=abundance_cutoff,
                thresholds=thresholds,
                timepoint_col=timepoint_col,
                save=args.save,
                save_path=args.output_dir + os.sep + 'rest_vs_tracked',
        )

    if graph_type in ['bias_change_rest']:
        if not args.plot_rest:
            print("\n !! plot-rest not set --> EXITING !! \n")
            sys.exit(1)
        if args.time_change == 'across':
            cumulative = True
            cache_file = args.cache_dir + os.sep + 'across_abundance_change_df.csv'
        elif args.time_change == 'between':
            cumulative = False
            cache_file = args.cache_dir + os.sep + 'between_abundance_change_df.csv'

        if args.by_day:
            first_timepoint = present_clones_df.day.min()
        if args.by_gen:
            first_timepoint = 1
        else:
            first_timepoint = 4

        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )

        cached_change = None
        if args.cache:
            cached_change = pd.read_csv(cache_file)
        plot_bias_change_rest(
            rest_of_clones_bias_df,
            cumulative=cumulative,
            by_day=args.by_day,
            by_gen=args.by_gen,
            first_timepoint=first_timepoint,
            save=args.save,
            save_path=args.output_dir + os.sep + 'bias_change_rest',
            cached_change=cached_change,
            cache_dir=args.cache_dir,
        )

    if graph_type in ['rest_by_phenotype']:
        for phenotype, group in rest_of_clones_bias_df.groupby('group'):
            plt.figure()
            sns.lineplot(
                x='day',
                y='lineage_bias',
                hue='mouse_id',
                style='code',
                data=group,
            )
            plt.title('Lineage Bias' + ' ' + phenotype.title())
        for phenotype, group in rest_of_clones_abundance_df.groupby('group'):
            for cell_type, cell_group in group.groupby('cell_type'):
                plt.figure()
                sns.lineplot(
                    x='day',
                    y='percent_engraftment',
                    hue='mouse_id',
                    style='code',
                    data=cell_group,
                )
                plt.title(phenotype.title() + ' ' + cell_type.title())

    if graph_type in ['bias_change_time_kde']:
        if args.time_change == 'across':
            cumulative = True
        elif args.time_change == 'between':
            cumulative = False
        cache_file = args.cache_dir \
            + os.sep + args.time_change + '_' \
            + args.group + '_bias_change_df.csv'

        if args.by_day:
            first_timepoint = present_clones_df.day.min()
            timepoint_col = 'day'
        elif args.by_gen:
            first_timepoint = 1
            timepoint_col = 'gen'
        else:
            first_timepoint = 4
            timepoint_col = 'month'

        print('Plotting ' + args.time_change + ' ' + timepoint_col + 's')
        cached_change = None
        if args.cache:
            if os.path.exists(cache_file):
                cached_change = pd.read_csv(cache_file)
            else:
                print('\n --- Warning: Cache does not exist, new cache will be generated --- \n')

        if args.plot_rest:
            print('Plotting Rest of Clones (untracked)')
            print(args.rest_of_clones)
            print(rest_of_clones_bias_df)
            print(rest_of_clones_bias_df.iloc[3:5]['mouse_id'].values.tolist())
            plot_bias_change_time_kdes(
                rest_of_clones_bias_df,
                first_timepoint=first_timepoint,
                absolute_value=args.magnitude,
                group=args.group,
                cumulative=cumulative,
                timepoint_col=timepoint_col,
                save=args.save,
                save_path=args.output_dir + os.sep + 'bias_change_rest',
                cached_change=cached_change,
                cache_dir=args.cache_dir,
            )
        else:
            plot_bias_change_time_kdes(
                lineage_bias_df,
                first_timepoint=first_timepoint,
                absolute_value=args.magnitude,
                group=args.group,
                cumulative=cumulative,
                timepoint_col=timepoint_col,
                save=args.save,
                save_path=args.output_dir + os.sep + 'bias_change',
                cached_change=cached_change,
                cache_dir=args.cache_dir,
            )

    if graph_type in ['abundance_change']:
        if args.time_change == 'across':
            cumulative = True
            cache_file = args.cache_dir + os.sep + 'across_abundance_change_df.csv'
        elif args.time_change == 'between':
            cumulative = False
            cache_file = args.cache_dir + os.sep + 'between_abundance_change_df.csv'

        if args.by_day:
            first_timepoint = present_clones_df.day.min()
        if args.by_gen:
            first_timepoint = 1
        else:
            first_timepoint = 4
        
        if options in ['default', 'first']:
            filter_end_time = False
        elif options in ['last']:
            filter_end_time = True
        else:
            print('Error: Filter time point option (-p) must be "first" or "last"')
            sys.exit(1)

        abundance_cutoff = 0
        thresholds = {'gr': 0, 'b': 0}
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )

        cached_change = None
        if args.cache:
            cached_change = pd.read_csv(cache_file)
        plot_abundance_change(
            present_clones_df,
            magnitude=args.magnitude,
            cumulative=cumulative,
            filter_end_time=filter_end_time,
            abundance_cutoff=abundance_cutoff,
            thresholds=thresholds,
            by_day=args.by_day,
            by_gen=args.by_gen,
            first_timepoint=first_timepoint,
            group=args.group,
            analyzed_cell_types=['gr', 'b'],
            save=args.save,
            save_path=args.output_dir + os.sep + 'abundance_change',
            cached_change=cached_change,
            cache_dir=args.cache_dir,
        )

    if graph_type in ['bias_change_gen_between_kde']:
        save_path = args.output_dir + os.sep + 'bias_change_kde_between'
        thresholds = {
            'gr': 0.0,
            'b': 0.0
            }
        abundance_cutoff = 0.0


        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )

        bias_change_df = between_gen_bias_change(
            lineage_bias_df,
            absolute=args.magnitude
        )
        
        plot_bias_change_time_kdes(
            bias_change_df,
            first_timepoint=first_timepoint,
            absolute_value=args.magnitude,
            group=args.group,
            save=args.save,
            save_path=save_path,
        )
    if graph_type in ['bias_change_gen_across_kde']:
        save_path = args.output_dir + os.sep + 'bias_change_kde_across'
        thresholds = {
            'gr': 0.0,
            'b': 0.0
            }
        abundance_cutoff = 0.0


        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )

        bias_change_df = across_gen_bias_change(
            lineage_bias_df,
            absolute=args.magnitude
        )
        
        plot_bias_change_time_kdes(
            bias_change_df,
            first_timepoint=first_timepoint,
            absolute_value=args.magnitude,
            group=args.group,
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['change_across_gens']:
        save_path = args.output_dir + os.sep + 'bias_across'
        thresholds = {
            'gr': 0.0,
            'b': 0.0
            }
        abundance_cutoff = 0.0


        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
        if args.magnitude:
            plot_bias_change_across_gens(
                lineage_bias_df,
                abundance_cutoff=abundance_cutoff,
                thresholds=thresholds,
                magnitude=True,
                group=args.group,
                by_clone=args.by_clone,
                save=args.save,
                save_path=save_path,
            )
        plot_bias_change_across_gens(
            lineage_bias_df,
            abundance_cutoff=abundance_cutoff,
            thresholds=thresholds,
            magnitude=False,
            group=args.group,
            by_clone=args.by_clone,
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['change_between_gens']:
        save_path = args.output_dir + os.sep + 'bias_between'
        thresholds = {
            'gr': 0.0,
            'b': 0.0
            }
        abundance_cutoff = 0.0

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col
            )

        if args.plot_rest:
            print('Plotting Rest of Clones (untracked)')
            print(rest_of_clones_bias_df.group)
            plot_bias_change_between_gen(
                rest_of_clones_bias_df,
                abundance_cutoff=abundance_cutoff,
                thresholds=thresholds,
                magnitude=True,
                group=args.group,
                by_clone=args.by_clone,
                legend='brief',
                style='code',
                save=args.save,
                save_path=save_path + '_rest',
            )
            plot_bias_change_between_gen(
                rest_of_clones_bias_df,
                abundance_cutoff=abundance_cutoff,
                thresholds=thresholds,
                legend='brief',
                style='code',
                magnitude=False,
                group=args.group,
                by_clone=args.by_clone,
                save=args.save,
                save_path=save_path + '_rest',
            )
        else:
            plot_bias_change_between_gen(
                lineage_bias_df,
                abundance_cutoff=abundance_cutoff,
                thresholds=thresholds,
                magnitude=True,
                group=args.group,
                by_clone=args.by_clone,
                save=args.save,
                save_path=save_path,
            )
            plot_bias_change_between_gen(
                lineage_bias_df,
                abundance_cutoff=abundance_cutoff,
                thresholds=thresholds,
                magnitude=False,
                group=args.group,
                by_clone=args.by_clone,
                save=args.save,
                save_path=save_path,
            )

    if graph_type in ['swarm_abund_cut']:
        abundance_cutoff = 0
        thresholds = {
            'gr': 0,
            'b': 0
        }

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff

            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col
            )
        group = 'all'
        if args.group:
            group = args.group

        cell_type = 'gr'
        for cell_type in ['gr', 'b']:
            swamplot_abundance_cutoff(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                thresholds=thresholds,
                by_group=args.by_group,
                timepoint_col=timepoint_col,
                cell_type=cell_type,
                save=args.save,
                save_path=args.output_dir + os.sep + 'swarmplot_abundance',
            )

    if graph_type in ['avg_abund_by_sum']:
        save_path = args.output_dir + os.sep + 'Average_Abundance'
        abundance_cutoff = 0
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
        analysed_cell_types = ['gr', 'b']
        print('Thresholds calculated based on cumulative abundance')
        _, thresholds = calculate_thresholds_sum_abundance(
            present_clones_df,
            abundance_cutoff=abundance_cutoff,
            timepoint_col=timepoint_col,
        )
        filter_df = filter_cell_type_threshold(
            present_clones_df,
            thresholds,
            analysed_cell_types,
            )

        cell_type = 'gr'
        plot_average_abundance(
            filter_df,
            thresholds=thresholds,
            cell_type=cell_type,
            save=args.save,
            save_path=save_path,
        )
        cell_type = 'b'
        plot_average_abundance(
            filter_df,
            thresholds=thresholds,
            cell_type=cell_type,
            save=args.save,
            save_path=save_path,
        )
        
    if graph_type in ['sum_abund_counts']:
        save_path = args.output_dir + os.sep + 'Clone_Count_at_Thresholds_Over_Time'
        abundance_cutoff = 50
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff

        plot_counts_at_abundance(present_clones_df,
                                 abundance_cutoff=abundance_cutoff,
                                 timepoint_col=timepoint_col,
                                 analyzed_cell_types=analysed_cell_types,
                                 save=args.save,
                                 line=args.line,
                                 save_path=save_path,
                                 group=args.group,
                                 )

    if graph_type in ['bias_eng_hist']:
        save_path = args.output_dir + os.sep + 'hist_bias_weighted_by_abundance'
        bins = 30
        month = 4
        if options != 'default':
            month = int(options)
        cell_type = 'gr'
        plot_weighted_bias_hist(
            lineage_bias_df,
            cell_type=cell_type,
            month=month,
            by_group=True,
            bins=bins,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
        cell_type = 'b'
        plot_weighted_bias_hist(
            lineage_bias_df,
            cell_type=cell_type,
            month=month,
            by_group=True,
            bins=bins,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )

        
    if graph_type in ['contrib_change_group']:
        save_path = args.output_dir + os.sep + 'changed_contribution'
        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)
        if args.cache:
            change_marked_df = pd.read_csv(args.cache_dir + os.sep + 'mtd' + str(mtd) + '_change_marked_df.csv')
        else:
            bias_change_df = get_bias_change(
                lineage_bias_df,
                timepoint_col,
            )
            change_marked_df = mark_changed(
                present_clones_df,
                bias_change_df,
                min_time_difference=mtd
            )
            change_marked_df.to_csv(args.cache_dir + os.sep + 'mtd' + str(mtd) + '_change_marked_df.csv', index=False)

        percent_of_total = True
        cell_type = 'gr'
        plot_change_contributions_by_group(
            change_marked_df,
            cell_type=cell_type,
            percent_of_total=percent_of_total,
            line=args.line,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
        cell_type = 'b'
        plot_change_contributions_by_group(
            change_marked_df,
            cell_type=cell_type,
            percent_of_total=percent_of_total,
            line=args.line,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )

    if graph_type in ['contrib_change_cell']:
        save_path = args.output_dir + os.sep + 'changed_contribution'
        if args.options != 'default':
            mtd = int(args.options)
        if args.cache:
            change_marked_df = pd.read_csv(args.cache_dir + os.sep + 'mtd' + str(mtd) + '_change_marked_df.csv')
        else:
            bias_change_df = get_bias_change(
                lineage_bias_df,
                timepoint_col=timepoint_col,
            )
            change_marked_df = mark_changed(
                present_clones_df,
                bias_change_df,
                min_time_difference=mtd,
                timepoint=args.timepoint,
            )
            change_marked_df.to_csv(args.cache_dir + os.sep + 'mtd' + str(mtd) + '_change_marked_df.csv', index=False)
        group = args.group
        percent_of_total = False
        print('Change Cutoff:')
        print(change_marked_df.change_cutoff.unique())
        plot_change_contributions(change_marked_df,
            timepoint_col=timepoint_col,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )

    if graph_type in ['sum_abundance']:
        save_path = args.output_dir + os.sep + 'abundance_at_percentile'
        num_points = 100
        cell_type = 'gr'
        contributions = percentile_sum_engraftment(present_clones_df, cell_type=cell_type, num_points=num_points, by_day=args.by_day)
        if args.by_day:
            time_point_col = 'day'
        else:
            time_point_col = 'month'
        plot_contributions(contributions,
            cell_type=cell_type,
            save=args.save,
            save_path=save_path,
            save_format='png',
            by_day=args.by_day
        )

        cell_type = 'b'
        contributions = percentile_sum_engraftment(present_clones_df, cell_type=cell_type, num_points=num_points, by_day=args.by_day)
        plot_contributions(contributions,
            cell_type=cell_type,
            save=args.save,
            save_path=save_path,
            save_format='png',
            by_day=args.by_day
        )

    if graph_type in ['range_bias_month']:
        save_path = args.output_dir + os.sep + 'Lineage_Bias_Line_Plot'
        filt_df = find_clones_bias_range_at_time(
            lineage_bias_df,
            month=4,
            min_bias=.45,
            max_bias=.75,
        )
        group = 'no_change'
        plot_lineage_bias_violin(
            filt_df,
            timepoint_col,
            group=group,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
        plot_lineage_bias_abundance_3d(filt_df, group=group)
        group = 'aging_phenotype'
        plot_lineage_bias_violin(
            filt_df,
            timepoint_col,
            group=group,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
        plot_lineage_bias_abundance_3d(filt_df, group=group)
        

    if graph_type in ['bias_violin']:
        if args.options == 'default':
            group = 'all'
        else:
            group = args.options
        save_path = args.output_dir
        plot_lineage_bias_violin(
            lineage_bias_df,
            group=group,
            save=args.save,
            save_path=save_path,
            save_format='png',
            timepoint_col=timepoint_col,
        )

    if graph_type in ['max_eng_mouse']:
        save_path = args.output_dir + os.sep + 'Max_Engraftment'
        percentile = .95
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)
        filtered_df = filter_cell_type_threshold(present_clones_df, thresholds=dominant_thresholds, analyzed_cell_types=['gr', 'b'])
        if args.options == 'default':
            group = 'all'
        else:
            group = args.options
        cell_type = 'b'
        plot_max_engraftment_by_mouse(
            filtered_df,
            title='Abundance > '
            + str(round(dominant_thresholds[cell_type], 2))
            + ' % WBC, Percentile: '
            + str(round(100*percentile, 2)),
            group=group,
            cell_type=cell_type,
            percentile=percentile,
            save=args.save,
            save_path=save_path
        )
        cell_type = 'gr'
        plot_max_engraftment_by_mouse(
            filtered_df,
            title='Abundance > '
            + str(round(dominant_thresholds[cell_type], 2))
            + ' % WBC, Percentile: '
            + str(round(100*percentile, 2)),
            group=group,
            cell_type=cell_type,
            percentile=percentile,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['max_eng_group']:
        save_path = args.output_dir + os.sep + 'Max_Engraftment'
        percentile = .95
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)
        filtered_df = filter_cell_type_threshold(present_clones_df, thresholds=dominant_thresholds, analyzed_cell_types=['gr', 'b'])

        cell_type = 'gr'
        plot_max_engraftment_by_group(
            filtered_df,
            title='Abundance > '
            + str(round(dominant_thresholds[cell_type], 2))
            + ' % WBC, Percentile: '
            + str(round(100*percentile, 2)),
            cell_type=cell_type,
            percentile=percentile,
            save=args.save,
            save_path=save_path
        )
        cell_type = 'b'
        plot_max_engraftment_by_group(
            filtered_df,
            title='Abundance > '
            + str(round(dominant_thresholds[cell_type], 2))
            + ' % WBC, Percentile: '
            + str(round(100*percentile, 2)),
            cell_type=cell_type,
            percentile=percentile,
            save=args.save,
            save_path=save_path
        )


    if graph_type in ['bias_change_cutoff']:
        abundance_cutoff = 0
        thresholds = {'gr':0, 'b':0}
        bias_change_df = None
        if args.abundance_cutoff:
            threshold = 'a'+str(args.abundance_cutoff).replace('.', '-')

            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=['gr','b'],
            )
        if args.cache:
            cache_file = args.cache_dir + os.sep + 'bias_change_df_a'+str(round(abundance_cutoff, 2)) + '.csv'
            bias_change_df = pd.read_csv(cache_file)

        min_time_difference = 0
        if options not in ['default']:
            min_time_difference = int(options)

        save_path = args.output_dir + os.sep + 'bias_change_cutoff'
        
        plot_bias_change_cutoff(
            lineage_bias_df=lineage_bias_df,
            thresholds=thresholds,
            timepoint_col=timepoint_col,
            timepoint=args.timepoint,
            abundance_cutoff=abundance_cutoff,
            absolute_value=True,
            group=args.group,
            min_time_difference=min_time_difference,
            save=args.save,
            save_path=save_path,
            cache_dir=args.cache_dir,
            cached_change=bias_change_df
        )
            


    if graph_type in ['bias_change_dist']:
        thresholds = [0.01]
        bias_data_dir = os.path.dirname(args.bias_change)
        save_path = args.output_dir + os.sep + 'bias_distribution'
        for threshold in thresholds:
            bias_change_file = glob.glob(bias_data_dir + os.sep + 'bias_change_t'+str(threshold).replace('.', '-')+'_*.csv')
            if len(bias_change_file) != 1:
                print('\nMissing file for threshold: ' + str(threshold))
                print('Results when searching for bias change file:')
                print(bias_change_file)
                continue
                
            th_change_df = pd.read_csv(bias_change_file[0])
            plot_bias_change_hist(th_change_df,
                threshold=threshold,
                absolute_value=True,
                group='all',
                save=args.save,
                save_path=save_path
            )
            plot_bias_change_hist(th_change_df,
                threshold=threshold,
                absolute_value=True,
                group='no_change',
                save=args.save,
                save_path=save_path
            )
            plot_bias_change_hist(th_change_df,
                threshold=threshold,
                absolute_value=True,
                group='aging_phenotype',
                save=args.save,
                save_path=save_path
            )
                    
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
                             save_path=args.output_dir + os.sep + 'Max_Engraftment',
        )


    if graph_type == 'bias_time_abund':
        plot_lineage_bias_abundance_3d(lineage_bias_df, by_day=args.by_day)
        plot_lineage_bias_abundance_3d(lineage_bias_df, by_day=args.by_day, group='aging_phenotype')
        plot_lineage_bias_abundance_3d(lineage_bias_df, by_day=args.by_day, group='no_change')

    if graph_type == 'counts_at_perc':
        percentile = .95
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)

        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')
        line = args.by_clone
        plot_counts_at_percentile(present_clones_df,
                                  percentile=percentile,
                                  thresholds=dominant_thresholds,
                                  analyzed_cell_types=analysed_cell_types,
                                  save=args.save,
                                  line=line,
                                  save_path=args.output_dir + os.sep + 'Clone_Count_at_Thresholds_Over_Time',
                                  group=args.group,
                                 )

    # Venn diagram of present clones
    if graph_type == 'venn':
        venn_barcode_in_time(present_clones_df,
                             analysed_cell_types,
                             save=args.save,
                             save_path=args.output_dir + os.sep + 'Venn_Presence_At_Time',
                             save_format='png',
                             group='no_change'
                            )
        venn_barcode_in_time(present_clones_df,
                             analysed_cell_types,
                             save=args.save,
                             save_path=args.output_dir + os.sep + 'Venn_Presence_At_Time',
                             save_format='png',
                             group='aging_phenotype'
                            )
        #venn_barcode_in_time(present_clones_df,
                             #analysed_cell_types,
                             #save=args.save,
                             #save_path=args.output_dir + os.sep + 'Venn_Presence_At_Time',
                             #save_format='png',
                             #group='all'
                            #)
    # heatmap present clones
    if graph_type == 'cluster':
        clustermap_clone_abundance(present_clones_df,
                                   analysed_cell_types,
                                   normalize=True,
                                   save=args.save,
                                   save_path=args.output_dir + os.sep + 'Heatmap_Clone_Abundance',
                                   save_format='png',
                                   group='aging_phenotype',
                                  )
        clustermap_clone_abundance(present_clones_df,
                                   analysed_cell_types,
                                   normalize=True,
                                   save=args.save,
                                   save_path=args.output_dir + os.sep + 'Heatmap_Clone_Abundance',
                                   save_format='png',
                                   group='no_change',
                                  )

    # Count clones by threshold
    if graph_type == 'clone_count_bar':

        abundance_cutoff = 0.01
        thresholds = {
            'gr': 0.01,
            'b': 0.01
        }

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff

            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col
            )

        line = args.line
        plot_clone_count_by_thresholds(
            present_clones_df,
            thresholds,
            analysed_cell_types,
            timepoint_col,
            abundance_cutoff=abundance_cutoff,
            line=line,
            save=args.save,
            save_path=args.output_dir + os.sep + 'Clone_Count_at_Thresholds_Over_Time',
            group=args.group
        )

    if graph_type == 'abund_bias_time':
        abundance_cutoff = 0
        thresholds = {
            'gr': 0,
            'b': 0
        }
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            print('Thresholds calculated based on cumulative abundance')
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                timepoint_col=timepoint_col,
                abundance_cutoff=abundance_cutoff,
            )

        timepoint = first_timepoint
        if args.timepoint:
            timepoint = args.timepoint


        analysed_cell_types = ['gr', 'b']

        print('Abundance cutoff set to: ' + str(abundance_cutoff))

        cell_type = 'gr'
        filt_lineage_bias_gr_df = combine_enriched_clones_at_time(
            input_df=lineage_bias_df,
            enrichment_time=timepoint,
            timepoint_col=timepoint_col,
            thresholds=thresholds,
            analyzed_cell_types=[cell_type],
            lineage_bias=True,
        )
        cell_type = 'b'
        filt_lineage_bias_b_df = combine_enriched_clones_at_time(
            input_df=lineage_bias_df,
            enrichment_time=timepoint,
            timepoint_col=timepoint_col,
            thresholds=thresholds,
            analyzed_cell_types=[cell_type],
            lineage_bias=True,
        )
        save_path=args.output_dir + os.sep \
            + args.y_col.title() + '_Line_Plot' \
            + os.sep + 'min_abund' \
            + str(args.filter_bias_abund).replace('.', '-') \
            + os.sep
        plot_lineage_average(
            filt_lineage_bias_gr_df,
            present_clones_df,
            title_addon='Gr > ' + str(round(thresholds['gr'], 2)) + '% WBC at ' + timepoint_col.title() + ': ' + str(timepoint),
            save=args.save,
            timepoint=timepoint,
            timepoint_col=timepoint_col,
            y_col=args.y_col,
            by_clone=args.by_clone,
            save_path=save_path+'gr',
            save_format='png',
            abundance=abundance_cutoff,
        )
        plot_lineage_average(
            filt_lineage_bias_b_df,
            present_clones_df,
            title_addon='B > ' + str(round(thresholds['b'], 2)) + '% WBC at ' + timepoint_col.title() + ': ' + str(timepoint),
            timepoint=timepoint,
            timepoint_col=timepoint_col,
            by_clone=args.by_clone,
            abundance=abundance_cutoff,
            y_col=args.y_col,
            save_format='png',
            save=args.save,
            save_path=save_path+'b'
        )
    if graph_type == 'perc_bias_month':
        percentile = .995
        month = 4
        if args.options == 'default':
            percentile = .995
        else:
            percentile = float(args.options)

        print('Percentile set to: ' + str(percentile))
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)

        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')

        cell_type = 'gr'
        filt_lineage_bias_gr_df = combine_enriched_clones_at_time(
                                                                 input_df=lineage_bias_df,
                                                                 enrichment_time=month,
                                                                 thresholds=dominant_thresholds,
                                                                 timepoint_col='month',
                                                                 lineage_bias=True,
                                                                 analyzed_cell_types=[cell_type],
        )
        cell_type = 'b'
        filt_lineage_bias_b_df = combine_enriched_clones_at_time(
                                                                 input_df=lineage_bias_df,
                                                                 enrichment_time=month,
                                                                 thresholds=dominant_thresholds,
                                                                 timepoint_col='month',
                                                                 lineage_bias=True,
                                                                 analyzed_cell_types=[cell_type],
        )
        plot_lineage_average(filt_lineage_bias_gr_df,
                             clonal_abundance_df=present_clones_df,
                             title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['gr'], 2)) + '% WBC abundance in GR at Month ' + str(month),
                             save=args.save,
                             timepoint=month,
                             timepoint_col=timepoint,
                             save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/gr',
                             save_format='png',
                             percentile=percentile
                            )
        plot_lineage_average(filt_lineage_bias_b_df,
                             clonal_abundance_df=present_clones_df,
                             title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['b'], 2)) + '% WBC abundance in b at Month ' + str(month),
                             timepoint=month,
                             timepoint_col=timepoint,
                             save=args.save,
                             save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/b',
                             save_format='png',
                             percentile=percentile
                            )

    if graph_type == 'abundant_at_last':
        abundance_cutoff = 0
        thresholds = {
            'gr': 0,
            'b': 0
        }
        analyzed_cell_types = ['gr', 'b']
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            print('Thresholds calculated based on cumulative abundance')
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=analyzed_cell_types,
            )

        for cell_type in analyzed_cell_types:
            filt_lineage_bias_df = clones_enriched_at_last_timepoint(
                input_df=input_df,
                lineage_bias_df=lineage_bias_df,
                thresholds=thresholds,
                lineage_bias=True,
                cell_type=cell_type,
                timepoint_col=timepoint_col
            )
            plot_lineage_bias_line(
                filt_lineage_bias_df,
                clonal_abundance_df=present_clones_df,
                title_addon='Filtered by clones with > ' + str(round(thresholds[cell_type], 2)) + '% WBC abundance in ' + cell_type.title() + ' at last timepoint',
                y_col=args.y_col,
                save=args.save,
                save_path=args.output_dir + os.sep + 'abundance_at_last_' + args.y_col + os.sep + cell_type,
                save_format='png',
                timepoint_col=timepoint_col,
                abundance=abundance_cutoff
            )

        
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

        filt_lineage_bias_b_df = clones_enriched_at_last_timepoint(
            input_df=input_df,
            timepoint_col=timepoint_col,
            lineage_bias_df=lineage_bias_df,
            thresholds=dominant_thresholds,
            lineage_bias=True,
            cell_type='gr',
        )
        filt_lineage_bias_gr_df = clones_enriched_at_last_timepoint(
            input_df=input_df,
            timepoint_col=timepoint_col,
            lineage_bias_df=lineage_bias_df,
            thresholds=dominant_thresholds,
            lineage_bias=True,
            cell_type='b',
        )
        plot_lineage_bias_line(
            filt_lineage_bias_gr_df,
            title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['gr'], 2)) + '% WBC abundance in GR at last timepoint',
            save=args.save,
            save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/gr',
            save_format='png',
            percentile=percentile
        )
        plot_lineage_bias_line(
            filt_lineage_bias_b_df,
            title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['b'], 2)) + '% WBC abundance in B at last timepoint',
            save=args.save,
            save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/b',
            save_format='png',
            percentile=percentile
        )

    # Lineage Bias Line Plots by threshold
    if graph_type == 'lineage_bias_line':
        threshold = 1
        filt_lineage_bias_df = clones_enriched_at_last_timepoint(input_df=input_df,
                                                                 timepoint_col=timepoint_col,
                                                                 lineage_bias_df=lineage_bias_df,
                                                                 thresholds={'any': threshold},
                                                                 lineage_bias=True,
                                                                 cell_type='any')
        plot_lineage_bias_line(filt_lineage_bias_df,
                               title_addon='Filtered by clones with Abundance >' + str(round(threshold, 2)) + '% WBC at last timepoint',
                               save=args.save,
                               by_day=args.by_day,
                               save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot',
                               save_format='png',
                              )

    # Abundant clones at specific time
    if graph_type == 'engraftment_time':
        if args.abundance_cutoff:
            print('Thresholds calculated based on cumulative abundance')
            abundance_cutoff = args.abundance_cutoff
            _, dominant_thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
            )
        else:
            percentile = 0.995
            print('Percentile Set To: ' + str(percentile))
            present_at_month_4 = present_clones_df.loc[present_clones_df[timepoint_col] == first_timepoint]
            dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)

        print(dominant_thresholds)
        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')

        timepoint = first_timepoint
        if args.timepoint:
            timepoint = args.timepoint
        plot_clone_enriched_at_time(present_clones_df,
                                    [timepoint],
                                    dominant_thresholds,
                                    timepoint_col=timepoint_col,
                                    save=args.save,
                                    save_path=args.output_dir + os.sep + 'Dominant_Clone_Abundance_Over_Time',
                                    save_format='png',
                                    by_clone=args.by_clone,
                                   )
    
    if plt.get_fignums():
        if not args.save:
            plt.show()
        else:
            print(Style.BRIGHT + Fore.GREEN + '\n*** All Plots Saved ***\n')
    else:
        print(Style.BRIGHT + Fore.RED + '\n !! ERROR: No Figures Drawn !! \n')


if __name__ == "__main__":
    main()
