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
import glob
import os
import sys
from colorama import init, Fore, Style
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_types import y_col_type, timepoint_type, change_type, change_status
from lineage_bias import parse_wbc_count_file
from aggregate_functions import (
    clones_enriched_at_last_timepoint,
    filter_cell_type_threshold,
    combine_enriched_clones_at_time,
    mark_changed,
    calculate_thresholds_sum_abundance,
    calculate_first_last_bias_change,
    day_to_month,
    day_to_gen
)
import aggregate_functions as agg
from plotting_functions import *

     


    
def main():
    """ Create plots set options via command line arguments

    Available graph types: -- Out of Date, reference comments in
        google drive slides for how each graph is generated.

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
    root_data_dir = '/Users/akre96/Data/HSC_aging_project/aging_10x_2020'
    parser = argparse.ArgumentParser(description="Plot input data")
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing long format step7 output', default=root_data_dir+'/Ania_M_allAnia OT2.0 rerun_percent-engraftment_NO filter_013120_long.csv')
    parser.add_argument('-r', '--rest', dest='rest_of_clones', help='Path to folder containing data on "rest of clones" abundnace and bias', default=root_data_dir+'/rest_of_clones')
    parser.add_argument('-l', '--lineage-bias', dest='lineage_bias', help='Path to csv containing lineage bias data', default=root_data_dir+'/lineage_bias/lineage_bias.csv')
    parser.add_argument('-c', '--count', dest='cell_count', help='Path to txt containing FACS cell count data', default=root_data_dir+'/WBC_combined.txt')
    parser.add_argument('--gfp', dest='gfp', help='Path to txt containing FACS GFP data', default=root_data_dir+'/GFP_combined.txt')
    parser.add_argument('--donor', dest='donor', help='Path to txt containing FACS Donor Chimerism data', default=root_data_dir+'/donor_combined.txt')
    parser.add_argument('--group-file', dest='group_file', help='Path to csv containing mouse to group mapping data', default=root_data_dir+'/mouse_id_group.csv')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default=root_data_dir+'/Graphs')
    parser.add_argument('-s', '--save', dest='save', help='Set flag if you want to save output graphs', action="store_true")
    parser.add_argument('-g', '--graph', dest='graph_type', help='Type of graph to output', default='default')
    parser.add_argument('-p', '--options', dest='options', help='Graph Options', default='default')
    parser.add_argument('-d', '--by-day', dest='by_day', help='Plotting done on a day by day basis', action="store_true")
    parser.add_argument('-t', '--threshold', dest='threshold', help='Set threshold for filtering', type=float, required=False, default=0.0)
    parser.add_argument('-a', '--abundance-cutoff', dest='abundance_cutoff', help='Set threshold based on abundance cutoff', type=float, required=False)
    parser.add_argument('-b', '--bias-cutoff', dest='bias_cutoff', help='Cutoff for extreme bias', type=float, required=False)
    parser.add_argument('--invert', dest='invert', help='Invert the selection being done while filtering', action='store_true')
    parser.add_argument('-f', '--filter-bias-abund', dest='filter_bias_abund', help='Abundance threshold to filter lineage bias data', type=float, required=False, default=0.05)
    parser.add_argument('--group', dest='group', help='Set group to inspect', type=str, required=False, default='all')
    parser.add_argument('--myeloid-cell', dest='myeloid_cell', help='cell used for myeloid lineage', type=str, required=False, default='gr')
    parser.add_argument('--lymphoid-cell', dest='lymphoid_cell', help='cell used for lymphoid lineage', type=str, required=False, default='b')
    parser.add_argument('--time-change', dest='time_change', help='Set time change to across or between for certain graphs', type=str, required=False, default='between')
    parser.add_argument('--timepoint', dest='timepoint', help='Set timepoint to inspect for certain graphs', type=timepoint_type, required=False)
    parser.add_argument('--change-status', dest='change_status', help='Set status (Changed/Unchanged) of lineage bias changed clones to inspect for certain graphs', type=change_status, required=False)
    parser.add_argument('--change-type', dest='change_type', help='Set type (Lymphoid/Myeloid) of lineage bias changed clones to inspect for certain graphs', type=change_type, required=False)
    parser.add_argument('--bias-type', dest='bias_type', help='Set type [LC, LB, B, MB, MC] of lineage bias to inspect for certain graphs', required=False)
    parser.add_argument('--line', dest='line', help='Wether to use lineplot for certain graphs', action="store_true")
    parser.add_argument('--pie', dest='pie', help='Wether to use pie chart for certain graphs', action="store_true")
    parser.add_argument('--by-group', dest='by_group', help='Whether to plot vs group istead of vs cell_type for certain graphs', action="store_true")
    parser.add_argument('--sum', dest='sum', help='Whether to plot sum abundance vs average abundance for certain graphs', action="store_true")
    parser.add_argument('--by-clone', dest='by_clone', help='Whether to plot clone color instead of group for certain graphs', action="store_true")
    parser.add_argument('--by-count', dest='by_count', help='Whether to plot count of clones for certain graphs', action="store_true")
    parser.add_argument('--by-average', dest='by_average', help='Whether to plot average until time point for certain graphs', action="store_true")
    parser.add_argument('--by-mouse', dest='by_mouse', help='Whether to plot mouse color instead of group for certain graphs', action="store_true")
    parser.add_argument('--plot-rest', dest='plot_rest', help='Whether to plot rest of clones instead of tracked clones', action="store_true")
    parser.add_argument('--by-gen', dest='by_gen', help='Plotting done on a generation by generation basis', action="store_true")
    parser.add_argument('--limit-gen', dest='limit_gen', help='Limit Serial Transplant data to first 3 generations', action="store_true")
    parser.add_argument('--magnitude', dest='magnitude', help='Plot change in magnitude', action="store_true")
    parser.add_argument('--cache', dest='cache', help='Use Cached Data', action="store_true")
    parser.add_argument('--cache-dir', dest='cache_dir', help='Where cache data is stored', default='/home/sakre/Data/cache')
    parser.add_argument('-y', '--y-col', dest='y_col', help='Which column to plot as y-axis for certain plots', required=False, default='lineage_bias', type=y_col_type)
    parser.add_argument('-n', dest='n', help='Integer value', required=False, type=int)

    # Init colorama
    init(autoreset=True)

    args = parser.parse_args()
    options = args.options
    input_df = pd.read_csv(args.input)
    raw_lineage_bias_df = pd.read_csv(args.lineage_bias)
    group_map = pd.read_csv(args.group_file)

    if 'group' in input_df.columns:
        input_df = input_df.drop(columns='group')
    if 'group' in raw_lineage_bias_df.columns:
        raw_lineage_bias_df = raw_lineage_bias_df.drop(columns='group')
    input_df = input_df.merge(
        group_map,
        how='left',
        validate='m:1'
    )
    raw_lineage_bias_df = raw_lineage_bias_df.merge(
        group_map,
        how='left',
        validate='m:1'
    )

    analyzed_cell_types = [args.myeloid_cell, args.lymphoid_cell]
    cell_count_df = parse_wbc_count_file(args.cell_count, [args.myeloid_cell, 'hsc', args.lymphoid_cell, 'wbc'])

    presence_thresholds = {
        'any': 0.01,
    }


    if 'group' not in input_df.columns:
        print(Style.BRIGHT + Fore.YELLOW+ '\n !! Warning: No Groups !!')
        input_df['group'] = 'None'
        raw_lineage_bias_df['group'] = 'None'
    else:
        phenotypic_groups = input_df.group.unique()
        if not input_df[~input_df.group.isin(phenotypic_groups)].empty:
            print(Style.BRIGHT + Fore.YELLOW+ '\n !! Warning: Following Mice not in a phenotypic group !!')
            print(Fore.YELLOW+ '  Mouse ID(s): ' + ', '.join(input_df[~input_df.group.isin(phenotypic_groups)].mouse_id.unique()))
        for group in phenotypic_groups:
            print(str(group).title() + ' Mice: ' + str(input_df[input_df.group == group].mouse_id.nunique()))
    





    present_clones_df = agg.filter_clones_threshold_anytime(
        input_df,
        presence_thresholds,
        analyzed_cell_types=input_df.cell_type.unique(),
        filter_exempt_cell_types=['hsc'],
        filt_0_out_exempt=False
    )

    lineage_bias_df = raw_lineage_bias_df
    # EXLCUDE MICE
    excluded_mice = pd.read_csv('~/Data/HSC_aging_project/exclude_mice.csv')
    for mouse_id in excluded_mice.mouse_id.unique():
        if not present_clones_df[present_clones_df.mouse_id == mouse_id].empty:
            print(Fore.YELLOW + ' Excluding mouse: ' + mouse_id)
            lineage_bias_df = lineage_bias_df[lineage_bias_df.mouse_id != mouse_id]
            present_clones_df = present_clones_df[present_clones_df.mouse_id != mouse_id]
        if not lineage_bias_df[lineage_bias_df.mouse_id == mouse_id].empty:
            print(Fore.YELLOW + ' Excluding mouse: ' + mouse_id)
            lineage_bias_df = lineage_bias_df[lineage_bias_df.mouse_id != mouse_id]
            present_clones_df = present_clones_df[present_clones_df.mouse_id != mouse_id]

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


    if args.cache:
        print(' - Using Cached Data')

    rest_of_clones_abundance_df = pd.read_csv(args.rest_of_clones + os.sep + 'rest_of_clones_abundance_long.csv')
    rest_of_clones_bias_df = pd.read_csv(args.rest_of_clones + os.sep + 'rest_of_clones_lineage_bias.csv')


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
        present_clones_df = present_clones_df.assign(gen=lambda x: day_to_gen(x.day))
        cell_count_df = cell_count_df.assign(gen=lambda x: day_to_gen(x.day))
        rest_of_clones_abundance_df = rest_of_clones_abundance_df.assign(gen=lambda x: day_to_gen(x.day))
        rest_of_clones_bias_df = rest_of_clones_bias_df.assign(gen=lambda x: day_to_gen(x.day))
        if args.limit_gen:
            print(' - ONLY CONSIDERING GENERATIONS 1-3 \n')
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen <= 3]
            present_clones_df = present_clones_df[present_clones_df.gen <= 3]
            cell_count_df = cell_count_df[cell_count_df.gen <= 3]
            rest_of_clones_abundance_df = rest_of_clones_abundance_df[rest_of_clones_abundance_df.gen <= 3]
            rest_of_clones_bias_df = rest_of_clones_bias_df[rest_of_clones_bias_df.gen <= 3]
    else:
        print(' - Time By Month Set \n')
        first_timepoint = 4
        timepoint_col = 'month'
        lineage_bias_df = lineage_bias_df.assign(month=lambda x: day_to_month(x.day))
        present_clones_df = present_clones_df.assign(month=lambda x: day_to_month(x.day))
        cell_count_df = cell_count_df.assign(month=lambda x: day_to_month(x.day))
        rest_of_clones_abundance_df = rest_of_clones_abundance_df.assign(month=lambda x: day_to_month(x.day))
        rest_of_clones_bias_df = rest_of_clones_bias_df.assign(month=lambda x: day_to_month(x.day))

    if args.filter_bias_abund:
        print(' - Lineage Bias Min Abundance set to: ' + str(args.filter_bias_abund))
        lineage_bias_df = agg.filter_lineage_bias_cell_type_ratio_per_mouse(
            lineage_bias_df,
            timepoint_col,
            cell_count_df,
            args.filter_bias_abund,
            myeloid_cell=args.myeloid_cell,
            lymphoid_cell=args.lymphoid_cell,)
        print('Mice found in lineage bias data:', ', '.join(lineage_bias_df.mouse_id.unique()))
    print('Mice found in abundance data:', ', '.join(present_clones_df.mouse_id.unique()))

    if graph_type in ['mouse_marker_legend']:
        save_path = args.output_dir + os.sep + 'legends'
        mouse_marker_legend(
            present_clones_df,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )


    if graph_type in ['clones_at_time_total_abundance']:
        save_path = args.output_dir + os.sep + 'clones_at_time_total_abundance'
        log_scale = False
        if args.options == 'log':
            log_scale = True

        if args.pie:
            plot_clones_at_time_total_abundance_heatmap(
                present_clones_df,
                timepoint_col,
                by_average=args.by_average,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
        else:
            plot_clones_at_time_total_abundance(
                present_clones_df,
                timepoint_col,
                log_scale,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
    if graph_type in ['blood_bias_abundance_time']:
        save_path = args.output_dir + os.sep + 'blood_bias_abundance_time'
        plot_blood_bias_abundance_time(
            present_clones_df,
            lineage_bias_df,
            timepoint_col,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['extreme_bias_percent_time']:
        save_path = args.output_dir + os.sep + 'extreme_bias_perc'

        timepoint = 'first'
        if args.timepoint:
            timepoint = args.timepoint
        
        extreme_bias_threshold = 0.9
        if args.bias_cutoff:
            extreme_bias_threshold = args.bias_cutoff

        thresholds = {'gr': 0.0, 'b': 0.0}

        if args.abundance_cutoff:
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=args.abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=[args.myeloid_cell, args.lymphoid_cell]
            )
        
        plot_extreme_bias_percent_time(
            lineage_bias_df,
            timepoint_col,
            timepoint,
            extreme_bias_threshold,
            abundance_thresholds=thresholds,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )


    if graph_type in ['abundance_lineplot']:
        save_path = args.output_dir + os.sep + 'abundance_lineplot'

        plot_abundance_lineplot(
            present_clones_df,
            lineage_bias_df,
            timepoint_col,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['expanded_at_time_abundance']:
        save_path = args.output_dir + os.sep + 'expanded_at_time_abundance'

        thresholds = {'gr': 0.0, 'b': 0.0}
        flip = False
        if args.options == 'flip':
            flip = True


        if args.abundance_cutoff:
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=args.abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=[args.myeloid_cell, args.lymphoid_cell]
            )

        plot_expanded_at_time_abundance(
            present_clones_df,
            timepoint_col,
            args.timepoint,
            args.by_group,
            thresholds,
            n=args.n,
            flip_cell_type=flip,
            by_mouse=args.by_mouse,
            group=args.group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['survival_line']:
        save_path = args.output_dir + os.sep + 'survival_line'
        plot_survival_line(
            present_clones_df,
            timepoint_col,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['compare_change_contrib']:
        save_path = args.output_dir + os.sep + 'changed_contribution'
        mtd = 1
        if timepoint_col == 'month':
            mtd = 3

        bar_col = 'change_type'
        bar_types = ['Lymphoid', 'Unchanged', 'Myeloid', 'Unknown']
        # Remove a type to not include it from plotting
        bar_types = ['Unchanged', 'Lymphoid', 'Myeloid']

        if args.options != 'default':
            if args.options in ['survived', 'change_type']:
                bar_col = str(args.options)
            else:
                raise ValueError('Bar Col must be change_type or survived')

        if bar_col == 'change_type':
            bias_change_df = calculate_first_last_bias_change(
                lineage_bias_df,
                timepoint_col=timepoint_col,
                by_mouse=False,
            )
            marked_df = mark_changed(
                present_clones_df,
                bias_change_df,
                merge_type='left',
                min_time_difference=mtd,
            )
            print('Change Cutoff:', marked_df.change_cutoff.unique()[0])
        elif bar_col == 'survived':
            bar_types = ['Exhausted', 'Survived', 'Unknown', 'Activated']
            bar_types = ['Exhausted', 'Survived', 'Activated']
            exhaust_labeled = agg.label_exhausted_clones(
                None,
                agg.remove_gen_8_5(
                    agg.remove_month_17_and_6(
                        present_clones_df,
                        timepoint_col
                    ),
                    timepoint_col,
                    keep_hsc=False
                ),
                timepoint_col
            )
            marked_df = agg.label_activated_clones(
                exhaust_labeled,
                timepoint_col
            )
            print('Exhaustion Labeling Results (unique clones):')
            print(marked_df.groupby('survived').code.nunique())

        timepoint = 'last'
        if args.timepoint:
            timepoint = args.timepoint

        donor_data = parse_wbc_count_file(
            args.donor,
            [args.myeloid_cell, args.lymphoid_cell],
            sep='\t',
            data_type='donor_perc'
        )
        gfp_data = parse_wbc_count_file(
            args.gfp,
            [args.myeloid_cell, args.lymphoid_cell],
            sep='\t',
            data_type='gfp_perc'
        )
        if timepoint_col == 'month':
            gfp_data[timepoint_col] = day_to_month(gfp_data['day'])
            donor_data[timepoint_col] = day_to_month(donor_data['day'])
        elif timepoint_col == 'gen':
            gfp_data[timepoint_col] = day_to_gen(gfp_data['day'])
            donor_data[timepoint_col] = day_to_gen(donor_data['day'])

        gfp_donor = donor_data.merge(
            gfp_data,
            how='inner',
            on=['mouse_id', timepoint_col, 'cell_type'],
            validate='1:1'
        )
        gfp_donor['gfp_x_donor'] = gfp_donor['gfp_perc'] * gfp_donor['donor_perc'] / (100 * 100)
        gfp_donor = gfp_donor[['mouse_id', timepoint_col, 'cell_type', 'gfp_x_donor']].drop_duplicates()

        force_order = False
        if timepoint_col == 'month':
            force_order = True
        plot_compare_change_contrib(
            marked_df,
            timepoint_col=timepoint_col,
            timepoint=timepoint,
            bar_col=bar_col,
            bar_types=bar_types,
            gfp_donor=gfp_donor,
            gfp_donor_thresh=args.threshold,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
    if graph_type in ['cell_type_expanded_hsc_vs_group']:
        save_path = args.output_dir + os.sep + 'cell_type_expanded_hsc_vs_group'

        abundance_cutoff = args.abundance_cutoff
        _, thresholds = calculate_thresholds_sum_abundance(
            present_clones_df,
            abundance_cutoff=50,
            timepoint_col=timepoint_col,
            analyzed_cell_types=['gr', 'b']
        )
        timepoint = 'last'
        if args.timepoint:
            timepoint = args.timepoint
        for cell_type in ['gr', 'b']:
            cell_type_expanded_hsc_vs_group(
                present_clones_df,
                timepoint_col,
                thresholds,
                by_mouse=args.by_mouse,
                by_count=args.by_count,
                cell_type=cell_type,
                timepoint=timepoint,
                save=args.save,
                save_path=save_path,
                save_format='png',
            )

    if graph_type in ['count_biased_changing_at_time']:
        save_path = args.output_dir + os.sep + 'count_biased_changing_at_time' 

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)
        
        timepoint = lineage_bias_df[timepoint_col].min()
        if args.timepoint:
            timepoint = args.timepoint

        plot_count_biased_changing_at_time(
            lineage_bias_df,
            timepoint_col,
            by_group=args.by_group,
            mtd=mtd,
            timepoint=int(timepoint),
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['hsc_abund_bias_last_LC']:
        save_path = args.output_dir + os.sep + 'hsc_abund_bias_lat'
        max_myeloid_abundance = 0.001
        if args.options != 'default':
            max_myeloid_abundance = float(args.options)

        plot_lymphoid_committed_vs_bias_hsc(
            present_clones_df,
            lineage_bias_df,
            timepoint_col,
            max_myeloid_abundance=max_myeloid_abundance,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )        
    if graph_type in ['percent_balanced_expanded']:
        save_path = args.output_dir + os.sep + 'percent_balanced_expanded'

        thresholds = {'gr': 0.0, 'b': 0.0}

        if args.abundance_cutoff:
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=args.abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=[args.myeloid_cell, args.lymphoid_cell]
            )
        if not args.bias_type:
            bias_type = 'B'
        else:
            bias_type = args.bias_type
        plot_percent_balanced_expanded(
            present_clones_df,
            lineage_bias_df,
            timepoint_col,
            thresholds,
            args.by_group,
            bias_type,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )        
    if graph_type in ['balanced_clone_abundance']:
        save_path = args.output_dir + os.sep + 'balanced_clone_abundance'

        thresholds = {'gr': 0.0, 'b': 0.0}

        if args.abundance_cutoff:
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=args.abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=[args.myeloid_cell, args.lymphoid_cell]
            )

        plot_balanced_clone_abundance(
            present_clones_df,
            lineage_bias_df,
            timepoint_col,
            thresholds,
            args.group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )        
        
    if graph_type in ['balanced_at_second_to_last']:
        save_path = args.output_dir + os.sep + 'balanced_s2l'
        plot_balanced_at_second_to_last(
                present_clones_df,
                lineage_bias_df,
                timepoint_col,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
    if graph_type in ['abundance_bias_change_type_heatmap']:
        save_path = args.output_dir + os.sep + 'abundance_by_change_heatmap' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)

        plot_abundance_bias_change_type_heatmap(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            mtd,
            plot_average=args.by_average,
            merge_type='inner',
            change_type=args.change_type,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['abundance_changed_bygroup']:
        save_path = args.output_dir + os.sep + 'abundance_by_change' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)

        plot_abundance_changed_bygroup(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            mtd,
            by_mouse=args.by_mouse,
            timepoint=args.timepoint,
            merge_type='inner',
            sum=args.sum,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['palette']:
        save_path = args.output_dir + os.sep + 'palette'
        if options != 'default':
            palette = options
        else:
            palette = 'mouse_id'
        plot_palette(
            palette,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['hsc_to_ct_lb_change']:
        save_path = args.output_dir + os.sep + 'hsc_to_ct_lb_change'

        abundance_cutoff = .01
        thresholds = {'gr': 0.0, 'b': 0.0, 'hsc': 0.0}

        if options != 'default':
            alpha = float(options)
        else:
            alpha = 0.1

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=[args.myeloid_cell, args.lymphoid_cell, 'hsc']
            )
        for cell_type in ['gr', 'b']:
            plot_hsc_vs_cell_type_lb_change(
                present_clones_df,
                lineage_bias_df,
                timepoint_col,
                alpha=alpha,
                by_group=args.by_group,
                thresholds=thresholds,
                cell_type=cell_type,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
    if graph_type in ['hsc_to_ct_bootstrap']:
        save_path = args.output_dir + os.sep + 'hsc_to_ct_bootstrap'

        abundance_cutoff = .01
        thresholds = {'gr': 0.0, 'b': 0.0, 'hsc': 0.0}

        if options != 'default':
            alpha = float(options)
        else:
            alpha = 0.1

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=[args.myeloid_cell, args.lymphoid_cell, 'hsc']
            )
        for cell_type in ['gr', 'b']:
            plot_hsc_vs_cell_type_abundance_bootstrapped(
                present_clones_df,
                lineage_bias_df,
                timepoint_col,
                alpha=alpha,
                by_group=args.by_group,
                thresholds=thresholds,
                cell_type=cell_type,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )

    if graph_type in ['hsc_vs_blood_count']:
        save_path = args.output_dir + os.sep + 'hsc-blood_count'
        if args.timepoint:
            exclude_timepoints = [int(args.timepoint)]
        else:
            exclude_timepoints = []
        
        plot_hsc_and_blood_clone_count(
            present_clones_df,
            timepoint_col,
            exclude_timepoints=exclude_timepoints,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['abundance_change_stable_group_grid']:
        save_path = args.output_dir + os.sep + 'abundance_change_stable_group_grid'
        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)
        if args.change_type is None:
            ct = 'Unchanged'
        else:
            ct = args.change_type
        timepoint = 'first'
        if args.timepoint is not None:
            timepoint = args.timepoint

        plot_abundance_change_stable_group_grid(
            lineage_bias_df,
            timepoint_col,
            timepoint=timepoint,
            change_type=ct,
            mtd=mtd,
            by_mouse=args.by_mouse,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['abundance_change_changed_group_grid']:
        save_path = args.output_dir + os.sep + 'abundance_change_change-type_group_grid'
        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)

        plot_abundance_change_changed_group_grid(
            lineage_bias_df,
            timepoint_col,
            mtd=mtd,
            by_mouse=args.by_mouse,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['abundance_changed_group_grid']:
        save_path = args.output_dir + os.sep + 'abundance_change-type_group_grid'

        plot_abundance_changed_group_grid(
            lineage_bias_df,
            timepoint_col,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['dist_bias_time_vs_group_facet_grid']:
        save_path = args.output_dir + os.sep + 'dist_bias_time_vs_group'
        bins = 20
        if options != 'default':
            bins = int(options)

        plot_dist_bias_at_time_vs_group_facet_grid(
            lineage_bias_df,
            timepoint_col,
            bins=bins,
            change_type=args.change_type,
            change_status=args.change_status,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )
    if graph_type in ['expanded_abundance_per_mouse']:
        save_path = args.output_dir + os.sep + 'expanded_abundance_per_mouse'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        abundance_cutoff = 0.0
        thresholds = {'gr': 0.0, 'b': 0.0, 'hsc': 0.0}

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff,
                timepoint_col=timepoint_col,
                analyzed_cell_types=['gr', 'b', 'hsc']
            )
        plot_expanded_abundance_per_mouse(
            present_clones_df,
            timepoint_col,
            abundance_cutoff,
            thresholds,
            by_sum=args.sum,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['exhausted_clone_abund']:
        save_path = args.output_dir + os.sep + 'exhausted_clone_abund'
        if args.by_clone:
            exhausted_clone_abund(
                lineage_bias_df,
                present_clones_df,
                timepoint_col,
                cell_type='any',
                by_sum=args.sum,
                by_count=args.by_clone,
                group=args.group,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
        else:
            for cell_type in ['gr', 'b']:
                exhausted_clone_abund(
                    lineage_bias_df,
                    present_clones_df,
                    timepoint_col,
                    cell_type=cell_type,
                    by_sum=args.sum,
                    by_count=args.by_clone,
                    group=args.group,
                    save=args.save,
                    save_path=save_path,
                    save_format='png'
                )
    if graph_type in ['exhaust_persist_abund']:
        save_path = args.output_dir + os.sep + 'exhausted_persist_abund'
        if args.by_count:
            exhaust_persist_abund(
                present_clones_df,
                timepoint_col,
                cell_type='any',
                by_sum=args.sum,
                by_count=args.by_count,
                by_group=args.by_group,
                plot_average=args.by_average,
                save=args.save,
                save_path=save_path,
                save_format='png'
            )
        else:
            for cell_type in ['gr', 'b']:
                exhaust_persist_abund(
                    present_clones_df,
                    timepoint_col,
                    cell_type=cell_type,
                    by_sum=args.sum,
                    by_count=args.by_count,
                    by_group=args.by_group,
                    plot_average=args.by_average,
                    save=args.save,
                    save_path=save_path,
                    save_format='png'
                )
    if graph_type in ['exhausted_clone_hsc_abund']:
        save_path = args.output_dir + os.sep + 'exhausted_clone_hsc_abund'
        exhausted_clone_hsc_abund(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            min_hsc_per_mouse,
            by_count=args.by_count,
            by_sum=args.sum,
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
            present_clones_df,
            timepoint_col,
            by_sum=args.sum,
            by_clone=args.by_clone,
            by_count=args.by_count,
            by_group=args.by_group,
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
            group=args.group,
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
        for cell_type in ['gr', 'b']:
            hsc_to_ct_compare(
                present_clones_df,
                timepoint_col,
                thresholds,
                by_group=args.by_group,
                abundance_cutoff=abundance_cutoff,
                invert=args.invert,
                cell_type=cell_type,
                by_mouse=args.by_mouse,
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
            
        #plot_clone_count_swarm_mean_first_last(
            #present_clones_df,
            #timepoint_col,
            #thresholds,
            #abundance_cutoff=abundance_cutoff,
            #analyzed_cell_types=list(thresholds.keys()),
            #save=args.save,
            #save_path=save_path,
            #save_format='png'
        #)
    if graph_type in ['clone_count_swarm_vs_ct']:
        save_path = args.output_dir + os.sep + 'clone_count_swarm_vs_cell-type'

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
        abundance_cutoff = 0.0
        thresholds = {'gr': 0.0, 'b': 0.0}
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
            by_group=args.by_group,
            abundance_cutoff=abundance_cutoff,
            analyzed_cell_types=list(thresholds.keys()),
            line=args.line,
            save=args.save,
            save_path=save_path,
            save_format='png'
        )

    if graph_type in ['shannon_time']:
        save_path = args.output_dir + os.sep + 'shannon_at_time'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        plot_diversity_index(
            present_clones_df,
            timepoint_col,
            timepoint=args.timepoint,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['n_most_abund_time']:
        save_path = args.output_dir + os.sep + 'n_most_abund_time'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        n = 3
        if args.n:
            n = int(args.n)

        plot_n_most_abundant_at_time(
            present_clones_df,
            timepoint_col,
            n,
            timepoint=args.timepoint,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['n_most_abund']:
        save_path = args.output_dir + os.sep + 'n_most_abund_contrib'

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        n = 5
        if args.n:
            n = int(args.n)

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

    if graph_type in ['count_by_change']:
        save_path = args.output_dir + os.sep + 'count_by_change' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)
            

        if timepoint_col == 'month':
            gxd_mice = ['M2012', 'M2059', 'M3010', 'M3013', 'M3016', 'M190', 'M2061', 'M3000', 'M3012', 'M3001', 'M3009', 'M3018', 'M3028']
            print(Fore.YELLOW + 'Only using mice ' + str(gxd_mice))
        plot_count_by_change(
            lineage_bias_df,
            present_clones_df,#[present_clones_df.mouse_id.isin(gxd_mice)],
            timepoint_col,
            mtd,
            hscs=args.invert,
            by_count=args.by_count,
            by_group=args.by_group,
            timepoint=args.timepoint,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['hsc_abundance_by_change']:
        save_path = args.output_dir + os.sep + 'hsc_abundance_by_change' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]

        mtd = 0
        if args.options != 'default':
            mtd = int(args.options)

        plot_hsc_abundance_by_change(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            mtd,
            by_group=args.by_group,
            by_clone=args.by_clone,
            by_mean=args.by_average,
            timepoint=args.timepoint,
            by_sum=args.sum,
            save=args.save,
            save_path=save_path
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
            timepoint=args.timepoint,
            merge_type='inner',
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
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5].assign(gen=lambda x: x.gen.astype(int))
            present_clones_df = present_clones_df[present_clones_df.gen != 8.5].assign(gen=lambda x: x.gen.astype(int))

        plot_perc_survival_bias_heatmap(
            lineage_bias_df,
            present_clones_df,
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
            present_clones_df,
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
                present_clones_df,
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

    if graph_type in ['changed_status_bias_at_time_abundance']:
        save_path = args.output_dir + os.sep + 'change_status_bias_at_time_abundance'

        mtd = 3
        if args.options != 'default':
            mtd = int(args.options)
        
        timepoint = 'last'
        if args.timepoint:
            timepoint = args.timepoint

        plot_change_status_bias_at_time_abundance(
            lineage_bias_df,
            timepoint_col,
            mtd,
            timepoint,
            args.by_group,
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['changed_status_bias_at_time']:
        save_path = args.output_dir + os.sep + 'change_status_bias_at_time'

        mtd = 3
        if args.options != 'default':
            mtd = int(args.options)
        timepoint = 'first'
        if args.timepoint is not None:
            timepoint = args.timepoint

        plot_change_status_bias_at_time(
            lineage_bias_df,
            timepoint_col,
            mtd,
            timepoint,
            args.by_group,
            save=args.save,
            save_path=save_path
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
            present_clones_df,
            timepoint_col,
            mtd,
            args.y_col,
            args.by_clone,
            save=args.save,
            save_path=save_path
        )
        


    if graph_type in ['hsc_abund_bias_last_change']:
        save_path = args.output_dir + os.sep + 'hsc_abund_bias_last_change' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        if timepoint_col == 'gen':
            lineage_bias_df = lineage_bias_df[lineage_bias_df.gen != 8.5]
        mtd=0
        if args.options != 'default':
            mtd = int(args.options)
        plot_hsc_abund_bias_at_last_change(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            by_group=args.by_group,
            mtd=mtd,
            merge_type='inner',
            save=args.save,
            save_path=save_path
        )
    if graph_type in ['hsc_abund_bias_last']:
        save_path = args.output_dir + os.sep + 'hsc_abund_bias_last' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        mtd=3
        if args.options != 'default':
            mtd = int(args.options)
        plot_hsc_abund_bias_at_last(
            lineage_bias_df,
            present_clones_df,
            timepoint_col,
            group=args.group,
            mtd=mtd,
            change_type=args.change_type,
            by_group=args.by_group,
            by_count=args.by_count,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['not_survived_count_box']:
        save_path = args.output_dir + os.sep + 'not_survived_count_box' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')
        if timepoint_col == 'gen':
            present_clones_df = present_clones_df[present_clones_df.gen != 8.5]
        plot_not_survived_count_box(
            present_clones_df,
            timepoint_col,
            by_group=args.by_group,
            save=args.save,
            save_path=save_path
        )

    if graph_type in ['not_survived_abund']:
        save_path = args.output_dir + os.sep + 'not_survived_abundance' \
            + os.sep + str(args.filter_bias_abund).replace('.', '-')

        plot_not_survived_abundance(
            present_clones_df,
            timepoint_col,
            by_average=args.by_average,
            by_mouse=args.by_mouse,
            group=args.group,
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
                present_clones_df,
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
                present_clones_df,
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
    if graph_type in ['abund_change_bias_dist_group_vs']:
        save_path = args.output_dir + os.sep + 'abund_change_bias_dist_group_vs'
        if args.threshold:
            threshold = args.threshold
        else:
            threshold = 0.01

        if args.options != 'default':
            mtd = int(args.options)
        else:
            mtd = 1

        for ct in ['gr', 'b']:
            plot_abund_change_bias_dist_group_vs(
                lineage_bias_df,
                timepoint_col,
                cutoff=threshold,
                mtd=mtd,
                by_mouse=args.by_mouse,
                timepoint=args.timepoint,
                y_col=ct+'_change',
                save=args.save,
                save_path=save_path,
            )
    if graph_type in ['bias_change_mean_dist_vs_group']:
        save_path = args.output_dir + os.sep + 'bias_distribution_mean_abund_vs_g'

        if args.options != 'default':
            mtd = int(args.options)
        else:
            mtd = 1

        plot_bias_dist_mean_abund_group_vs(
            lineage_bias_df,
            timepoint_col,
            change_status=args.change_type,
            mtd=mtd,
            timepoint=args.timepoint,
            y_col=args.y_col,
            by_mouse=args.by_mouse,
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
                present_clones_df,
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
            if timepoint_col == 'month':
                lineage_bias_df = filter_lineage_bias_n_timepoints_threshold(
                    raw_lineage_bias_df,
                    threshold=0.01,
                    n_timepoints=3,
                    timepoint_col=timepoint_col
                )

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

    if graph_type in ['not_first_last_abundance']:
        save_path = args.output_dir + os.sep + 'not_first_last_abundance'
        plot_not_first_last_abundance(
            present_clones_df,
            timepoint_col,
            save=args.save,
            save_path=save_path,
        )
    if graph_type in ['activated_clone_bias']:
        save_path = args.output_dir + os.sep + 'activated_clone_bias'
        plot_activated_clone_bias(
            present_clones_df,
            lineage_bias_df,
            timepoint_col,
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
                present_clones_df,
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
                present_clones_df,
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
                present_clones_df,
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
                present_clones_df,
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

    if graph_type in ['swarm_abund_cut']:
        abundance_cutoff = 0
        thresholds = {
            'gr': 0,
            'b': 0
        }

        if args.by_mouse:
            for cell_type in ['gr', 'b']:
                plot_abundance_clones_per_mouse(
                    present_clones_df,
                    timepoint_col=timepoint_col,
                    cell_type=cell_type,
                    thresholds=thresholds,
                    by_group=args.by_group,
                    save=args.save,
                    save_path=args.output_dir + os.sep + 'swarmplot_abundance',

                )
        else:
            if args.abundance_cutoff:
                abundance_cutoff = args.abundance_cutoff

                _, thresholds = calculate_thresholds_sum_abundance(
                    present_clones_df,
                    abundance_cutoff=abundance_cutoff,
                    timepoint_col=timepoint_col
                )

            if abundance_cutoff == 0:
                print('\n ~~ Cutoff set to 0, due to number of clones plotting will take some time ~~ \n')

            cell_type = 'gr'
            for cell_type in ['gr', 'b']:
                print('Plotting for ' + cell_type.title() + ' Cells')
                swamplot_abundance_cutoff(
                    present_clones_df,
                    abundance_cutoff=abundance_cutoff,
                    thresholds=thresholds,
                    by_group=args.by_group,
                    group=args.group,
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

    if graph_type in ['hsc_abund_at_exh_lb']:
        save_path = args.output_dir + os.sep + 'hsc_abund_at_exh_lb'
        mtd = 1
        if timepoint_col == 'month':
            mtd = 3

        bar_col = 'change_type'
        bar_types = ['Lymphoid', 'Unchanged', 'Myeloid', 'Unknown']
        # Remove a type to not include it from plotting
        bar_types = ['Unchanged', 'Lymphoid', 'Myeloid']

        if args.options != 'default':
            if args.options == 'survived':
                bar_col = str(args.options)
            else:
                raise ValueError('Bar Col must be default or survived')

        if bar_col == 'change_type':
            bias_change_df = calculate_first_last_bias_change(
                lineage_bias_df,
                timepoint_col=timepoint_col,
                by_mouse=False,
            )
            marked_df = mark_changed(
                present_clones_df,
                bias_change_df,
                merge_type='left',
                min_time_difference=mtd,
            )
            print('Change Cutoff:', marked_df.change_cutoff.unique()[0])
        elif bar_col == 'survived':
            bar_types = ['Exhausted', 'Survived', 'Activated', 'Unknown']
            #bar_types = ['Exhausted', 'Survived', 'Activated']
            exhaust_labeled = agg.label_exhausted_clones(
                None,
                agg.remove_gen_8_5(
                    agg.remove_month_17_and_6(
                        present_clones_df,
                        timepoint_col
                    ),
                    timepoint_col,
                    keep_hsc=False
                ),
                timepoint_col,
                present_thresh=0.01,
            )
            marked_df = agg.label_activated_clones(
                exhaust_labeled,
                timepoint_col,
                present_thresh=0.01,
            )
            print('Exhaustion Labeling Results (unique clones):')
            print(marked_df.groupby('survived').code.nunique())
        marked_df[['mouse_id', 'group', 'code', bar_col]].drop_duplicates().to_csv(
            timepoint_col + '_' + bar_col + '.csv',
            index=False,
        )

        gxd_mice = None
        if timepoint_col == 'month':
            gxd_mice = ['M2012', 'M2059', 'M3010', 'M3013', 'M3016', 'M190', 'M2061', 'M3000', 'M3012', 'M3001', 'M3009', 'M3018', 'M3028']
            print(Fore.YELLOW + 'Only using mice ' + str(gxd_mice))

        gxd_mice = None
        plot_hsc_abund_at_exh_lb(
            marked_df,
            present_clones_df,
            timepoint_col=timepoint_col,
            gxd_mice=gxd_mice,
            bar_col=bar_col,
            bar_types=bar_types,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
    if graph_type in ['contrib_change_cell']:
        save_path = args.output_dir + os.sep + 'changed_contribution'
        mtd = 1
        if timepoint_col == 'month':
            mtd = 3

        present_threshold = 0.01
        if args.abundance_cutoff:
            present_threshold = args.abundance_cutoff


        bar_col = 'change_type'
        bar_types = ['Lymphoid', 'Unchanged', 'Myeloid', 'Unknown']
        # Remove a type to not include it from plotting
        bar_types = ['Lymphoid', 'Unchanged', 'Myeloid']

        if args.options != 'default':
            if args.options in ['survived', 'change_type']:
                bar_col = str(args.options)
            else:
                raise ValueError('Bar Col must be change_type or survived')

        if bar_col == 'change_type':
            bias_change_df = calculate_first_last_bias_change(
                lineage_bias_df,
                timepoint_col=timepoint_col,
                by_mouse=False,
            )
            marked_df = mark_changed(
                present_clones_df,
                bias_change_df,
                merge_type='left',
                min_time_difference=mtd,
            )
            marked_df[['mouse_id', 'group', 'code', bar_col]].drop_duplicates().to_csv('aging_OT_change_type.csv', index=False)
            print('Change Cutoff:', marked_df.change_cutoff.unique()[0])
        elif bar_col == 'survived':
            bar_types = ['Exhausted', 'Survived', 'Unknown', 'Activated']
            bar_types = ['Exhausted', 'Survived', 'Activated']
            exhaust_labeled = agg.label_exhausted_clones(
                None,
                agg.remove_gen_8_5(
                    agg.remove_month_17_and_6(
                        present_clones_df,
                        timepoint_col
                    ),
                    timepoint_col,
                    keep_hsc=False
                ),
                timepoint_col,
                present_thresh=present_threshold
            )
            marked_df = agg.label_activated_clones(
                exhaust_labeled,
                timepoint_col,
                present_thresh=present_threshold,
            )
            print('Exhaustion Labeling Results (unique clones):')
            print(marked_df.groupby('survived').code.nunique())
        group = args.group
        timepoint = 'last'
        if args.timepoint:
            timepoint = args.timepoint

        donor_data = parse_wbc_count_file(
            args.donor,
            [args.myeloid_cell, args.lymphoid_cell],
            sep='\t',
            data_type='donor_perc'
        )
        gfp_data = parse_wbc_count_file(
            args.gfp,
            [args.myeloid_cell, args.lymphoid_cell],
            sep='\t',
            data_type='gfp_perc'
        )
        if timepoint_col == 'month':
            gfp_data[timepoint_col] = day_to_month(gfp_data['day'])
            donor_data[timepoint_col] = day_to_month(donor_data['day'])
        elif timepoint_col == 'gen':
            gfp_data[timepoint_col] = day_to_gen(gfp_data['day'])
            donor_data[timepoint_col] = day_to_gen(donor_data['day'])

        gfp_donor = donor_data.merge(
            gfp_data,
            how='inner',
            on=['mouse_id', timepoint_col, 'cell_type'],
            validate='1:1'
        )
        gfp_donor['gfp_x_donor'] = gfp_donor['gfp_perc'] * gfp_donor['donor_perc'] / (100 * 100)
        gfp_donor = gfp_donor[['mouse_id', 'cell_type', timepoint_col, 'gfp_x_donor']].drop_duplicates()
        force_order = False
        if timepoint_col in ['month', 'gen']:
            force_order = True
        
        if args.pie:
            plot_change_contributions_pie(
                marked_df,
                timepoint_col=timepoint_col,
                bar_col=bar_col,
                bar_types=bar_types,
                timepoint=timepoint,
                gfp_donor=gfp_donor,
                gfp_donor_thresh=args.threshold,
                force_order=force_order,
                present_thresh=present_threshold,
                save=args.save,
                save_path=save_path,
                save_format='png',
            )
        else:
            plot_change_contributions_refactor(
                marked_df,
                timepoint_col=timepoint_col,
                bar_col=bar_col,
                bar_types=bar_types,
                timepoint=timepoint,
                gfp_donor=gfp_donor,
                gfp_donor_thresh=args.threshold,
                force_order=force_order,
                present_thresh=present_threshold,
                save=args.save,
                save_path=save_path,
                save_format='png',
            )

    if graph_type in ['sum_abundance']:
        save_path = args.output_dir + os.sep + 'abundance_at_percentile'
        num_points = 200
        
        for cell_type in ['b', 'gr']:
            print(cell_type)
            plot_contributions(
                present_clones_df,
                cell_type=cell_type,
                timepoint_col=timepoint_col,
                save=args.save,
                save_path=save_path,
                save_format='png',

            )



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


    if graph_type in ['bias_change_cutoff']:
        abundance_cutoff = 0
        thresholds = {'gr':0, 'b':0}
        bias_change_df = None
        if args.abundance_cutoff:
            threshold = 'a'+str(args.abundance_cutoff).replace('.', '-')

            abundance_cutoff = args.abundance_cutoff
            _, thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
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
            group=args.group,
            min_time_difference=min_time_difference,
            save=args.save,
            save_path=save_path,
            cache_dir=args.cache_dir,
            cached_change=bias_change_df
        )
        plot_bias_change_cutoff_hist(
            lineage_bias_df=lineage_bias_df,
            thresholds=thresholds,
            timepoint_col=timepoint_col,
            timepoint=args.timepoint,
            abundance_cutoff=abundance_cutoff,
            group=args.group,
            min_time_difference=min_time_difference,
            save=args.save,
            save_path=save_path,
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
            for group in ['all', 'aging_phenotype', 'no_change']:
                plot_bias_change_hist(th_change_df,
                    threshold=threshold,
                    absolute_value=True,
                    group=group,
                    save=args.save,
                    save_path=save_path
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
                present_clones_df,
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



        print('Abundance cutoff set to: ' + str(abundance_cutoff))

        save_path=args.output_dir + os.sep \
            + args.y_col.title() + '_Line_Plot' \
            + os.sep + 'min_abund' \
            + str(args.filter_bias_abund).replace('.', '-') \
            + os.sep

        for cell_type in analyzed_cell_types:
            filt_lineage_bias_df = combine_enriched_clones_at_time(
                input_df=lineage_bias_df,
                enrichment_time=timepoint,
                timepoint_col=timepoint_col,
                thresholds=thresholds,
                analyzed_cell_types=[cell_type],
                lineage_bias=True,
            )
            plot_lineage_average(
                filt_lineage_bias_df,
                present_clones_df,
                title_addon=cell_type.title() + ' > ' + str(round(thresholds[cell_type], 2)) + '% WBC at ' + timepoint_col.title() + ': ' + str(timepoint),
                save=args.save,
                timepoint=timepoint,
                timepoint_col=timepoint_col,
                y_col=args.y_col,
                by_clone=args.by_clone,
                save_path=save_path+cell_type,
                save_format='png',
                abundance=abundance_cutoff,
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
                input_df=present_clones_df,
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

        

    # Lineage Bias Line Plots by threshold
    if graph_type == 'lineage_bias_line':
        threshold = 1
        filt_lineage_bias_df = clones_enriched_at_last_timepoint(input_df=present_clones_df,
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
    
    if plt.get_fignums():
        if not args.save:
            plt.show()
        else:
            print(Style.BRIGHT + Fore.GREEN + '\n*** All Plots Saved ***\n')
    else:
        print(Style.BRIGHT + Fore.RED + '\n !! ERROR: No Figures Drawn !! \n')


if __name__ == "__main__":
    main()
