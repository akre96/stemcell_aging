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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aggregate_functions import filter_threshold, \
    clones_enriched_at_last_timepoint, percentile_sum_engraftment, \
    find_top_percentile_threshold, find_clones_bias_range_at_time, \
    filter_cell_type_threshold, combine_enriched_clones_at_time, \
    mark_changed, sum_abundance_by_change, between_gen_bias_change, \
    calculate_thresholds_sum_abundance, filter_lineage_bias_threshold, \
    across_gen_bias_change, between_gen_bias_change
from plotting_functions import plot_max_engraftment, \
    plot_clone_count_by_thresholds, venn_barcode_in_time, \
    plot_clone_enriched_at_time, plot_counts_at_percentile, \
    plot_lineage_bias_abundance_3d, plot_lineage_bias_line, \
    clustermap_clone_abundance, plot_bias_change_hist, \
    plot_max_engraftment_by_group, plot_bias_change_cutoff, \
    plot_max_engraftment_by_mouse, plot_lineage_bias_violin, \
    plot_lineage_average, plot_contributions, plot_weighted_bias_hist, \
    plot_change_contributions, plot_change_contributions_by_group, \
    plot_counts_at_abundance, plot_average_abundance, \
    swamplot_abundance_cutoff, plot_bias_change_between_gen, \
    plot_bias_change_across_gens, plot_bias_change_time_kdes, \
    plot_abundance_change
     


    
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
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing long format step7 output', default='Ania_M_all_percent-engraftment_100818_long.csv')
    parser.add_argument('-l', '--lineage-bias', dest='lineage_bias', help='Path to csv containing lineage bias data', default='/home/sakre/Code/stemcell_aging/output/lineage_bias/lineage_bias_t0-0_from-counts.csv')
    parser.add_argument('-c', '--bias-change', dest='bias_change', help='Path to csv containing lineage bias change', default='/home/sakre/Code/stemcell_aging/output/lineage_bias/bias_change_t0-0_from-counts.csv')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='/home/sakre/Data/stemcell_aging/Graphs')
    parser.add_argument('-s', '--save', dest='save', help='Set flag if you want to save output graphs', action="store_true")
    parser.add_argument('-g', '--graph', dest='graph_type', help='Type of graph to output', default='default')
    parser.add_argument('-p', '--options', dest='options', help='Graph Options', default='default')
    parser.add_argument('-d', '--by-day', dest='by_day', help='Plotting done on a day by day basis', action="store_true")
    parser.add_argument('-a', '--abundance-cutoff', dest='abundance_cutoff', help='Set threshold based on abundance cutoff', type=float, required=False)
    parser.add_argument('--group', dest='group', help='Set group to inspect', type=str, required=False, default='all')
    parser.add_argument('--time-change', dest='time_change', help='Set time change to across or between for certain graphs', type=str, required=False, default='between')
    parser.add_argument('--timepoint', dest='timepoint', help='Set timepoint to inspect for certain graphs', type=int, required=False)
    parser.add_argument('--line', dest='line', help='Wether to use lineplot for certain graphs', action="store_true")
    parser.add_argument('--by-group', dest='by_group', help='Whether to plot vs group istead of vs cell_type for certain graphs', action="store_true")
    parser.add_argument('--by-clone', dest='by_clone', help='Whether to plot clone color instead of group for certain graphs', action="store_true")
    parser.add_argument('--by-gen', dest='by_gen', help='Plotting done on a generation by generation basis', action="store_true")
    parser.add_argument('--magnitude', dest='magnitude', help='Plot change in magnitude', action="store_true")
    parser.add_argument('--cache', dest='cache', help='Use Cached Data', action="store_true")
    parser.add_argument('--cache-dir', dest='cache_dir', help='Where cache data is stored', default='/home/sakre/Data/cache')

    args = parser.parse_args()
    options = args.options
    input_df = pd.read_csv(args.input)
    lineage_bias_df = pd.read_csv(args.lineage_bias)
    bias_change_df = pd.read_csv(args.bias_change)

    analysed_cell_types = ['gr', 'b']

    presence_threshold = 0.0
    present_clones_df = filter_threshold(input_df, presence_threshold, analysed_cell_types)
    all_clones_df = filter_threshold(input_df, 0.0, analysed_cell_types)
    graph_type = args.graph_type

    color_palettes = json.load(open('color_palettes.json', 'r'))


    if args.save:
        print('\n*** Saving Plots Enabled ***\n')

    if graph_type in ['abundance_change', 'default']:
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
                by_day=args.by_day,
            )

        cached_change = None
        if args.cache:
            cached_change = pd.read_csv(cache_file)
        plot_abundance_change(
            present_clones_df,
            magnitude=args.magnitude,
            cumulative=cumulative,
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
                by_day=args.by_day,
            )

        bias_change_df = between_gen_bias_change(
            lineage_bias_df,
            absolute=args.magnitude
        )
        
        plot_bias_change_time_kdes(
            bias_change_df,
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
                by_day=args.by_day,
            )

        bias_change_df = across_gen_bias_change(
            lineage_bias_df,
            absolute=args.magnitude
        )
        
        plot_bias_change_time_kdes(
            bias_change_df,
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
                by_day=args.by_day,
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
                by_day=args.by_day,
            )

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
        save_path = args.output_dir + os.sep + 'swarmplot_abundance'
        abundance_cutoff = 0
        thresholds = {
            'gr': 0,
            'b': 0
        }
        if args.by_day:
            n_timepoints = 1
        else:
            n_timepoints = 4

        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff

            _, thresholds = calculate_thresholds_sum_abundance(
                input_df,
                abundance_cutoff=abundance_cutoff,
                by_day=args.by_day,
            )
        group = 'all'
        if args.group:
            group = args.group

        cell_type = 'gr'
        swamplot_abundance_cutoff(
            present_clones_df,
            abundance_cutoff=abundance_cutoff,
            thresholds=thresholds,
            group=group,
            n_timepoints=n_timepoints,
            by_day=args.by_day,
            color_col='mouse_id',
            cell_type=cell_type,
            save=args.save,
            save_path=args.output_dir + os.sep + 'swarmplot_abundance',
        )

        cell_type = 'b'
        swamplot_abundance_cutoff(
            present_clones_df,
            abundance_cutoff=abundance_cutoff,
            thresholds=thresholds,
            n_timepoints=n_timepoints,
            group=group,
            by_day=args.by_day,
            color_col='mouse_id',
            cell_type=cell_type,
            save=args.save,
            save_path=save_path,
        )

    if graph_type in ['avg_abund_by_sum']:
        save_path = args.output_dir + os.sep + 'Average_Abundance'
        abundance_cutoff = 50
        if options != 'default':
            abundance_cutoff = float(options)
        analysed_cell_types = ['gr', 'b']
        print('Thresholds calculated based on cumulative abundance')
        _, thresholds = calculate_thresholds_sum_abundance(
            present_clones_df,
            abundance_cutoff=abundance_cutoff
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
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]

        if options not in ['default', 'bar', 'line']:
            abundance_cutoff = float(options)

        line = True
        plot_counts_at_abundance(present_clones_df,
                                 abundance_cutoff=abundance_cutoff,
                                 analyzed_cell_types=analysed_cell_types,
                                 save=args.save,
                                 line=line,
                                 save_path=save_path,
                                 group='all',
                                 )
        plot_counts_at_abundance(present_clones_df,
                                 abundance_cutoff=abundance_cutoff,
                                 analyzed_cell_types=analysed_cell_types,
                                 save=args.save,
                                 save_path=save_path,
                                 line=line,
                                 group='no_change',
                                 )
        plot_counts_at_abundance(present_clones_df,
                                 abundance_cutoff=abundance_cutoff,
                                 analyzed_cell_types=analysed_cell_types,
                                 save=args.save,
                                 line=line,
                                 save_path=save_path,
                                 group='aging_phenotype',
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
        change_marked_df = mark_changed(present_clones_df, bias_change_df)
        line = False
        if options != 'default':
            line = True
        percent_of_total = True
        cell_type = 'gr'
        plot_change_contributions_by_group(change_marked_df,
            cell_type=cell_type,
            percent_of_total=percent_of_total,
            line=line,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
        cell_type = 'b'
        plot_change_contributions_by_group(change_marked_df,
            cell_type=cell_type,
            percent_of_total=percent_of_total,
            line=line,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )

    if graph_type in ['contrib_change_cell']:
        save_path = args.output_dir + os.sep + 'changed_contribution'
        change_marked_df = mark_changed(present_clones_df, bias_change_df)
        group = 'all'
        if options != 'default':
            group = options

        percent_of_total = False

        cell_type = 'gr'
        plot_change_contributions(change_marked_df,
            cell_type=cell_type,
            group=group,
            percent_of_total=percent_of_total,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
        cell_type = 'b'
        plot_change_contributions(change_marked_df,
            cell_type=cell_type,
            group=group,
            percent_of_total=percent_of_total,
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
            group=group,
            save=args.save,
            save_path=save_path,
            save_format='png',
        )
        plot_lineage_bias_abundance_3d(filt_df, group=group)
        group = 'aging_phenotype'
        plot_lineage_bias_violin(
            filt_df,
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
        save_path = args.output_dir + os.sep + 'Lineage_Bias_Line_Plot'
        plot_lineage_bias_violin(
            lineage_bias_df,
            group=group,
            save=args.save,
            save_path=save_path,
            save_format='png',
            by_day=args.by_day
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
        thresholds = ['t'+str(0.00).replace('.', '-')]
        if args.abundance_cutoff:
            thresholds = ['a'+str(args.abundance_cutoff).replace('.', '-')]

        min_time_difference = 0
        if options not in ['default']:
            min_time_difference = int(options)

        bias_data_dir = os.path.dirname(args.bias_change)
        save_path = args.output_dir + os.sep + 'bias_change_cutoff'
        for threshold in thresholds:
            bias_change_file = glob.glob(bias_data_dir + os.sep + 'bias_change_'+ threshold +'_*.csv')
            if len(bias_change_file) != 1:
                print('\nMissing file for threshold: ' + str(threshold))
                print('Results when searching for bias change file:')
                print(bias_change_file)
                continue
            th_change_df = pd.read_csv(bias_change_file[0])
            plot_bias_change_cutoff(
                th_change_df,
                threshold=threshold,
                absolute_value=True,
                group=args.group,
                min_time_difference=min_time_difference,
                save=args.save,
                save_path=save_path
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
        line = ((args.options == 'line') | ( args.options == 'default'))
        plot_counts_at_percentile(present_clones_df,
                                  percentile=percentile,
                                  thresholds=dominant_thresholds,
                                  analyzed_cell_types=analysed_cell_types,
                                  save=args.save,
                                  line=line,
                                  save_path=args.output_dir + os.sep + 'Clone_Count_at_Thresholds_Over_Time',
                                  group='aging_phenotype',
                                 )
        plot_counts_at_percentile(present_clones_df,
                                  percentile=percentile,
                                  thresholds=dominant_thresholds,
                                  analyzed_cell_types=analysed_cell_types,
                                  save=args.save,
                                  line=line,
                                  save_path=args.output_dir + os.sep + 'Clone_Count_at_Thresholds_Over_Time',
                                  group='no_change',
                                 )
        plot_counts_at_percentile(present_clones_df,
                                  percentile=percentile,
                                  thresholds=dominant_thresholds,
                                  analyzed_cell_types=analysed_cell_types,
                                  save=args.save,
                                  line=line,
                                  save_path=args.output_dir + os.sep + 'Clone_Count_at_Thresholds_Over_Time',
                                  group='all',
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
        clone_count_thresholds = [0.01]
        group = 'all'
        if 'default' not in options:
            group = options

        line = args.line
        plot_clone_count_by_thresholds(present_clones_df,
                                       clone_count_thresholds,
                                       analysed_cell_types,
                                       by_day=args.by_day,
                                       line=line,
                                       save=args.save,
                                       save_path=args.output_dir + os.sep + 'Clone_Count_at_Thresholds_Over_Time',
                                       group=group)

    # Lineage Bias Line Plots by percentile
    if graph_type == 'abund_bias_time':
        abundance_cutoff = 0
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff

        timepoint = 4
        if args.timepoint:
            timepoint = args.timepoint

        timepoint_col = 'month'
        if args.by_day:
            timepoint_col = 'day'

        analysed_cell_types = ['gr', 'b']
        print('Thresholds calculated based on cumulative abundance')
        _, thresholds = calculate_thresholds_sum_abundance(
            present_clones_df,
            abundance_cutoff=abundance_cutoff,
            by_day=args.by_day
        )

        print('Abundance cutoff set to: ' + str(abundance_cutoff))
        month = 4

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
        plot_lineage_average(
            filt_lineage_bias_gr_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['gr'], 2)) + '% WBC abundance in GR at ' + timepoint_col.title() + ': ' + str(timepoint),
            save=args.save,
            timepoint=timepoint,
            timepoint_col=timepoint_col,
            save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/gr',
            save_format='png',
            abundance=abundance_cutoff,
        )
        plot_lineage_average(
            filt_lineage_bias_b_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['b'], 2)) + '% WBC abundance in B at ' + timepoint_col.title() + ': ' + str(timepoint),
            save=args.save,
            save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/b',
            timepoint=timepoint,
            timepoint_col=timepoint_col,
            save_format='png',
            abundance=abundance_cutoff,
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
                             title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['gr'], 2)) + '% WBC abundance in GR at Month ' + str(month),
                             save=args.save,
                             month=month,
                             save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/gr',
                             save_format='png',
                             percentile=percentile
                            )
        plot_lineage_average(filt_lineage_bias_b_df,
                             title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['b'], 2)) + '% WBC abundance in b at Month ' + str(month),
                             save=args.save,
                             save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/b',
                             month=month,
                             save_format='png',
                             percentile=percentile
                            )

    if graph_type == 'top_abund_bias':
        abundance_cutoff = 0
        if args.abundance_cutoff:
            abundance_cutoff = args.abundance_cutoff

        if options != 'default':
            abundance_cutoff = float(options)
        analysed_cell_types = ['gr', 'b']
        print('Thresholds calculated based on cumulative abundance')
        _, thresholds = calculate_thresholds_sum_abundance(
            present_clones_df,
            abundance_cutoff=abundance_cutoff,
            by_day=args.by_day,
        )

        filt_lineage_bias_b_df = clones_enriched_at_last_timepoint(
            input_df=input_df,
            lineage_bias_df=lineage_bias_df,
            thresholds=thresholds,
            lineage_bias=True,
            cell_type='gr',
            by_day=args.by_day,
        )
        filt_lineage_bias_gr_df = clones_enriched_at_last_timepoint(
            input_df=input_df,
            lineage_bias_df=lineage_bias_df,
            thresholds=thresholds,
            lineage_bias=True,
            cell_type='b',
            by_day=args.by_day,
        )
        plot_lineage_bias_line(
            filt_lineage_bias_gr_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['gr'], 2)) + '% WBC abundance in GR at last timepoint',
            save=args.save,
            save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/gr',
            save_format='png',
            by_day=args.by_day,
            abundance=abundance_cutoff
        )
        plot_lineage_bias_line(
            filt_lineage_bias_b_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['b'], 2)) + '% WBC abundance in B at last timepoint',
            save=args.save,
            save_path=args.output_dir + os.sep + 'Lineage_Bias_Line_Plot/b',
            save_format='png',
            by_day=args.by_day,
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
            lineage_bias_df=lineage_bias_df,
            thresholds=dominant_thresholds,
            lineage_bias=True,
            cell_type='gr',
        )
        filt_lineage_bias_gr_df = clones_enriched_at_last_timepoint(
            input_df=input_df,
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
        if options == 'default':
            percentile = 0.995
            print('Percentile Set To: ' + str(percentile))
            present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
            dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)
        else:
            print('Thresholds calculated based on cumulative abundance')
            abundance_cutoff = float(options)
            _, dominant_thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                by_day=args.by_day,
                abundance_cutoff=abundance_cutoff
            )

        by_mouse = False
        print(dominant_thresholds)
        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')

        plot_clone_enriched_at_time(all_clones_df,
                                    [4, 14],
                                    dominant_thresholds,
                                    save=args.save,
                                    save_path=args.output_dir + os.sep + 'Dominant_Clone_Abundance_Over_Time',
                                    save_format='png',
                                    by_mouse=by_mouse
                                   )
    
    if not args.save:
        plt.show()
    else:
        print('\n*** All Plots Saved ***\n')


if __name__ == "__main__":
    main()
