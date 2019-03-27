""" Create plots from step 7 output data

Returns:
    None - Shows plots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from aggregate_functions import filter_threshold, \
     clones_enriched_at_last_timepoint, get_bias_change, \
     find_top_percentile_threshold, \
     filter_cell_type_threshold
from plotting_functions import plot_max_engraftment, \
     plot_clone_count_by_thresholds, venn_barcode_in_time, \
     plot_clone_enriched_at_time, plot_counts_at_percentile, \
     plot_lineage_bias_abundance_3d, plot_lineage_bias_line, \
     clustermap_clone_abundance, plot_bias_change_hist



def main():
    """ Create plots set options via command line arguments

    Available graph types:
        default:            Subject to change based on what is being actively developed
        cluster:            Clustered heatmap of present clone engraftment
        venn:               Venn Diagram of clone existance at timepoint
        clone_count_bar:    Bar charts of clone counts by cell type at different thresholds
        clone_count_line:   Lineplots of clone counts by cell type, mice, and average at different thresholds
        lineage_bias_line:  lineplots of lineage bias over time at different abundance from last timepoint
        top_perc_bias:      line plot of lineage bias over time with top percentile of clones by abundance during last time point
                            options -- value 0-1 indicating percentile to filter for (.95 => 95th percentile)
        engraftment_time:   lineplot/swarmplot of abundance of clones with high values at 4, 12, and 14 months
        counts_at_perc:     line or barplot of clone counts where cell-types are filtered at 90th percentile of abundance
        bias_time_abund:    3d plot of lineage bias vs time vs abundance in b and gr cells
        max_engraftment:    point plot of maximum engraftment averaged across mice by phenotype groups and all mice

    """

    parser = argparse.ArgumentParser(description="Plot input data")
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing long format step7 output', default='Ania_M_all_percent-engraftment_100818_long.csv')
    parser.add_argument('-l', '--lineage-bias', dest='lineage_bias', help='Path to csv containing lineage bias data', default='lineage_bias_from_counts.csv')
    parser.add_argument('-c', '--bias-change', dest='bias_change', help='Path to csv containing lineage bias change', default='lineage_bias_change_from_counts.csv')
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='output/Graphs')
    parser.add_argument('-s', '--save', dest='save', help='Set flag if you want to save output graphs', action="store_true")
    parser.add_argument('-g', '--graph', dest='graph_type', help='Type of graph to output', default='default')
    parser.add_argument('-p', '--options', dest='options', help='Graph Options', default='default')

    args = parser.parse_args()
    input_df = pd.read_csv(args.input)
    lineage_bias_df = pd.read_csv(args.lineage_bias)
    bias_change_df = pd.read_csv(args.bias_change)

    analysed_cell_types = ['gr', 'b']

    presence_threshold = 0.01
    present_clones_df = filter_threshold(input_df, presence_threshold, analysed_cell_types)
    all_clones_df = filter_threshold(input_df, 0.0, analysed_cell_types)
    graph_type = args.graph_type



    if args.save:
        print('\n **Saving Plots Enabled** \n')

    if graph_type in ['default', 'bias_change']:
        plot_bias_change_hist(bias_change_df)
        
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
        percentile = 0.95
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
