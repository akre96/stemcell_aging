""" Create plots from step 7 output data

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
    mark_changed, sum_abundance_by_change, \
    calculate_thresholds_sum_abundance, filter_lineage_bias_threshold
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
    swamplot_abundance_cutoff
     


    
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
    parser.add_argument('-l', '--lineage-bias', dest='lineage_bias', help='Path to csv containing lineage bias data', default='lineage_bias_from_counts.csv')
    parser.add_argument('-c', '--bias-change', dest='bias_change', help='Path to csv containing lineage bias change', default='/home/sakre/Code/stemcell_aging/output/lineage_bias/bias_change_t0-01_from-counts.csv')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='output/Graphs')
    parser.add_argument('-s', '--save', dest='save', help='Set flag if you want to save output graphs', action="store_true")
    parser.add_argument('-g', '--graph', dest='graph_type', help='Type of graph to output', default='default')
    parser.add_argument('-p', '--options', dest='options', help='Graph Options', default='default')

    args = parser.parse_args()
    options = args.options
    input_df = pd.read_csv(args.input)
    lineage_bias_df = pd.read_csv(args.lineage_bias)
    bias_change_df = pd.read_csv(args.bias_change)

    analysed_cell_types = ['gr', 'b']

    presence_threshold = 0.01
    present_clones_df = filter_threshold(input_df, presence_threshold, analysed_cell_types)
    all_clones_df = filter_threshold(input_df, 0.0, analysed_cell_types)
    graph_type = args.graph_type

    color_palettes = json.load(open('color_palettes.json', 'r'))


    if args.save:
        print('\n*** Saving Plots Enabled ***\n')


    if graph_type in ['swarm_abund_cut', 'default']:
        abundance_cutoff = 50
        if options != 'default':
            abundance_cutoff = float(options)

        cell_type = 'gr'
        swamplot_abundance_cutoff(
            present_clones_df,
            abundance_cutoff=abundance_cutoff,
            color_col='mouse_id',
            cell_type=cell_type,
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/swarmplot_abundance',
        )

        cell_type = 'b'
        swamplot_abundance_cutoff(
            present_clones_df,
            abundance_cutoff=abundance_cutoff,
            color_col='mouse_id',
            cell_type=cell_type,
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/swarmplot_abundance',
        )

    if graph_type in ['avg_abund_by_sum']:
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Average_Abundance',
        )
        cell_type = 'b'
        plot_average_abundance(
            filter_df,
            thresholds=thresholds,
            cell_type=cell_type,
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Average_Abundance',
        )
        
    if graph_type in ['sum_abund_counts']:
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
                                 save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                 group='all',
                                 )
        plot_counts_at_abundance(present_clones_df,
                                 abundance_cutoff=abundance_cutoff,
                                 analyzed_cell_types=analysed_cell_types,
                                 save=args.save,
                                 line=line,
                                 save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                 group='no_change',
                                 )
        plot_counts_at_abundance(present_clones_df,
                                 abundance_cutoff=abundance_cutoff,
                                 analyzed_cell_types=analysed_cell_types,
                                 save=args.save,
                                 line=line,
                                 save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Clone_Count_at_Thresholds_Over_Time',
                                 group='aging_phenotype',
                                 )

    if graph_type in ['bias_eng_hist']:
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/hist_bias_weighted_by_abundance',
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/hist_bias_weighted_by_abundance',
            save_format='png',
        )

        
    if graph_type in ['contrib_change_group']:
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/changed_contribution',
            save_format='png',
        )
        cell_type = 'b'
        plot_change_contributions_by_group(change_marked_df,
            cell_type=cell_type,
            percent_of_total=percent_of_total,
            line=line,
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/changed_contribution',
            save_format='png',
        )

    if graph_type in ['contrib_change_cell']:
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/changed_contribution',
            save_format='png',
        )
        cell_type = 'b'
        plot_change_contributions(change_marked_df,
            cell_type=cell_type,
            group=group,
            percent_of_total=percent_of_total,
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/changed_contribution',
            save_format='png',
        )

    if graph_type in ['sum_abundance']:
        num_points = 400
        cell_type = 'gr'
        contributions = percentile_sum_engraftment(present_clones_df, cell_type=cell_type, num_points=num_points)
        plot_contributions(contributions,
            cell_type=cell_type,
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/abundance_at_percentile',
            save_format='png',
        )

        cell_type = 'b'
        contributions = percentile_sum_engraftment(present_clones_df, cell_type=cell_type, num_points=num_points)
        plot_contributions(contributions,
            cell_type=cell_type,
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/abundance_at_percentile',
            save_format='png',
        )

    if graph_type in ['range_bias_month']:
        filt_df = find_clones_bias_range_at_time(
            lineage_bias_df,
            month=4,
            min_bias=.45,
            max_bias=.75,
        )
        group = 'no_change'
        plot_lineage_bias_violin(filt_df,
                               group=group,
                               save=args.save,
                               save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot',
                               save_format='png',
                              )
        plot_lineage_bias_abundance_3d(filt_df, group=group)
        group = 'aging_phenotype'
        plot_lineage_bias_violin(filt_df,
                               group=group,
                               save=args.save,
                               save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot',
                               save_format='png',
                              )
        plot_lineage_bias_abundance_3d(filt_df, group=group)
        

    if graph_type in ['bias_violin']:
        if args.options == 'default':
            group = 'all'
        else:
            group = args.options

        plot_lineage_bias_violin(lineage_bias_df,
                               group=group,
                               save=args.save,
                               save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot',
                               save_format='png',
                              )

    if graph_type in ['max_eng_mouse']:
        percentile = .95
        present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
        dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)
        filtered_df = filter_cell_type_threshold(present_clones_df, thresholds=dominant_thresholds, analyzed_cell_types=['gr', 'b'])
        if args.options == 'default':
            group = 'all'
        else:
            group = args.options
        cell_type = 'b'
        plot_max_engraftment_by_mouse(filtered_df,
                             title='Abundance > '
                             + str(round(dominant_thresholds[cell_type], 2))
                             + ' % WBC, Percentile: '
                             + str(round(100*percentile, 2)),
                             group=group,
                             cell_type=cell_type,
                             percentile=percentile,
                             save=args.save,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Max_Engraftment'
        )
        cell_type = 'gr'
        plot_max_engraftment_by_mouse(filtered_df,
                             title='Abundance > '
                             + str(round(dominant_thresholds[cell_type], 2))
                             + ' % WBC, Percentile: '
                             + str(round(100*percentile, 2)),
                             group=group,
                             cell_type=cell_type,
                             percentile=percentile,
                             save=args.save,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Max_Engraftment'
        )
    if graph_type in ['max_eng_group']:
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Max_Engraftment'
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Max_Engraftment'
        )


    if graph_type in ['bias_change_cutoff']:
        thresholds = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
        bias_data_dir = 'output/lineage_bias'
        for threshold in thresholds:
            bias_change_file = glob.glob(bias_data_dir + os.sep + 'bias_change_t'+str(threshold).replace('.', '-')+'_*.csv')
            if len(bias_change_file) != 1:
                print('\nMissing file for threshold: ' + str(threshold))
                print('Results when searching for bias change file:')
                print(bias_change_file)
                continue
            th_change_df = pd.read_csv(bias_change_file[0])
            plot_bias_change_cutoff(th_change_df,
                threshold=threshold,
                absolute_value=True,
                group='all',
                save=args.save,
                save_path='/home/sakre/Code/stemcell_aging/output/Graphs/bias_change_cutoff'
            )
            


    if graph_type in ['bias_change_dist']:
        thresholds = [0.02, 0.1, 0.2]
        bias_data_dir = 'output/lineage_bias'
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
                save_path='/home/sakre/Code/stemcell_aging/output/Graphs/bias_distribution'
            )
            plot_bias_change_hist(th_change_df,
                threshold=threshold,
                absolute_value=True,
                group='no_change',
                save=args.save,
                save_path='/home/sakre/Code/stemcell_aging/output/Graphs/bias_distribution'
            )
            plot_bias_change_hist(th_change_df,
                threshold=threshold,
                absolute_value=True,
                group='aging_phenotype',
                save=args.save,
                save_path='/home/sakre/Code/stemcell_aging/output/Graphs/bias_distribution'
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
        #venn_barcode_in_time(present_clones_df,
                             #analysed_cell_types,
                             #save=args.save,
                             #save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Venn_Presence_At_Time',
                             #save_format='png',
                             #group='all'
                            #)
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
    if graph_type == 'abund_bias_month':
        abundance_cutoff = 50
        if options != 'default':
            abundance_cutoff = float(options)

        analysed_cell_types = ['gr', 'b']
        print('Thresholds calculated based on cumulative abundance')
        _, thresholds = calculate_thresholds_sum_abundance(
            present_clones_df,
            abundance_cutoff=abundance_cutoff
        )

        print('Abundance cutoff set to: ' + str(abundance_cutoff))
        month = 4

        cell_type = 'gr'
        filt_lineage_bias_gr_df = combine_enriched_clones_at_time(
            input_df=lineage_bias_df,
            enrichment_month=month,
            thresholds=thresholds,
            lineage_bias=True,
            analyzed_cell_types=[cell_type],
        )
        cell_type = 'b'
        filt_lineage_bias_b_df = combine_enriched_clones_at_time(
            input_df=lineage_bias_df,
            enrichment_month=month,
            thresholds=thresholds,
            lineage_bias=True,
            analyzed_cell_types=[cell_type],
        )
        plot_lineage_average(
            filt_lineage_bias_gr_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['gr'], 2)) + '% WBC abundance in GR at Month ' + str(month),
            save=args.save,
            month=month,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/gr',
            save_format='png',
            abundance=abundance_cutoff,
        )
        plot_lineage_average(
            filt_lineage_bias_b_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['b'], 2)) + '% WBC abundance in B at Month ' + str(month),
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/b',
            month=month,
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
                                                                 enrichment_month=month,
                                                                 thresholds=dominant_thresholds,
                                                                 lineage_bias=True,
                                                                 analyzed_cell_types=[cell_type],
        )
        cell_type = 'b'
        filt_lineage_bias_b_df = combine_enriched_clones_at_time(
                                                                 input_df=lineage_bias_df,
                                                                 enrichment_month=month,
                                                                 thresholds=dominant_thresholds,
                                                                 lineage_bias=True,
                                                                 analyzed_cell_types=[cell_type],
        )
        plot_lineage_average(filt_lineage_bias_gr_df,
                             title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['gr'], 2)) + '% WBC abundance in GR at Month ' + str(month),
                             save=args.save,
                             month=month,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/gr',
                             save_format='png',
                             percentile=percentile
                            )
        plot_lineage_average(filt_lineage_bias_b_df,
                             title_addon='Filtered by clones with > ' + str(round(dominant_thresholds['b'], 2)) + '% WBC abundance in b at Month ' + str(month),
                             save=args.save,
                             save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/b',
                             month=month,
                             save_format='png',
                             percentile=percentile
                            )

    if graph_type == 'top_abund_bias':
        abundance_cutoff = 50
        if options != 'default':
            abundance_cutoff = float(options)
        analysed_cell_types = ['gr', 'b']
        print('Thresholds calculated based on cumulative abundance')
        _, thresholds = calculate_thresholds_sum_abundance(
            present_clones_df,
            abundance_cutoff=abundance_cutoff
        )

        filt_lineage_bias_b_df = clones_enriched_at_last_timepoint(
            input_df=input_df,
            lineage_bias_df=lineage_bias_df,
            thresholds=thresholds,
            lineage_bias=True,
            cell_type='gr',
        )
        filt_lineage_bias_gr_df = clones_enriched_at_last_timepoint(
            input_df=input_df,
            lineage_bias_df=lineage_bias_df,
            thresholds=thresholds,
            lineage_bias=True,
            cell_type='b',
        )
        plot_lineage_bias_line(
            filt_lineage_bias_gr_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['gr'], 2)) + '% WBC abundance in GR at last timepoint',
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/gr',
            save_format='png',
            abundance=abundance_cutoff
        )
        plot_lineage_bias_line(
            filt_lineage_bias_b_df,
            title_addon='Filtered by clones with > ' + str(round(thresholds['b'], 2)) + '% WBC abundance in B at last timepoint',
            save=args.save,
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/b',
            save_format='png',
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
            save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Lineage_Bias_Line_Plot/gr',
            save_format='png',
            percentile=percentile
        )
        plot_lineage_bias_line(
            filt_lineage_bias_b_df,
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
        if options == 'default':
            percentile = 0.995
            print('Percentile Set To: ' + str(percentile))
            present_at_month_4 = present_clones_df.loc[present_clones_df.month == 4]
            dominant_thresholds = find_top_percentile_threshold(present_at_month_4, percentile=percentile)
        else:
            print('Thresholds calculated based on cumulative abundance')
            abundance_cutoff = float(options)
            percentiles, dominant_thresholds = calculate_thresholds_sum_abundance(
                present_clones_df,
                abundance_cutoff=abundance_cutoff
            )

        by_mouse = True
        for cell_type, threshold in dominant_thresholds.items():
            print('Threshold for ' + cell_type + ' cells: ' + str(round(threshold, 2)) + '% WBC')

        plot_clone_enriched_at_time(all_clones_df,
                                    [4, 14],
                                    dominant_thresholds,
                                    save=args.save,
                                    save_path='/home/sakre/Code/stemcell_aging/output/Graphs/Dominant_Clone_Abundance_Over_Time',
                                    save_format='png',
                                    by_mouse=by_mouse
                                   )
    
    if not args.save:
        plt.show()
    else:
        print('\n*** All Plots Saved ***\n')


if __name__ == "__main__":
    main()
