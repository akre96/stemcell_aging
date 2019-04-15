import os
import argparse
import numpy as np
import pandas as pd
from lineage_bias import parse_wbc_count_file, \
    calculate_baseline_counts, normalize_to_baseline_counts, \
    create_lineage_bias_df

def calculate_unbarcoded_abundance(row) -> np.float:
    return ((row.donor_perc/100) * (1 - row.GFP_perc/100) * row.cell_perc/100)

def calculate_host_clones(row) -> np.float:
    return ((1 - row.donor_perc/100) * row.cell_perc/100)

def main():
    """ Calculate and save lineage bias
    """

    parser = argparse.ArgumentParser(description="Calculate bias and abundance of unbarcoded and host cells")
    parser.add_argument('--gfp', dest='gfp_file', help='Path to txt file containing gfp percent data', required=True)
    parser.add_argument('--wbcs', dest='wbcs_file', help='Path to txt containing FACS cell count data', required=True)
    parser.add_argument('--donor', dest='donor_file', help='Path to file containing donor chimerism data', required=True)
    parser.add_argument('--group', dest='group_file', help='Path to csv file with columns for mouse_id and group', required=True)
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', required=True)
    parser.add_argument('-t', '--timepoint-col', dest='timepoint_col', help='Column to look for time in', required=True)
    parser.add_argument('-b', '--baseline-timepoint', dest='baseline_timepoint', help='Timepoint to set as baseline value for normalization', required=True, type=int)

    args = parser.parse_args()

    group_info_df = pd.read_csv(args.group_file)
    WBC_df = parse_wbc_count_file(args.wbcs_file, analyzed_cell_types=['gr', 'b', 'wbc'])
    GFP_df = parse_wbc_count_file(args.gfp_file).rename(columns={'cell_count':'GFP_perc'})
    donor_df = parse_wbc_count_file(args.donor_file).rename(columns={'cell_count':'donor_perc'})

    cells_df = WBC_df[WBC_df.cell_type != 'wbc']
    wbcs_df = WBC_df[WBC_df.cell_type == 'wbc']

    with_percent_df = pd.DataFrame()
    for name, group in cells_df.groupby(['cell_type']):
        div = 100 * np.divide(group.sort_values(by=['mouse_id', 'day']).cell_count, wbcs_df.sort_values(by=['mouse_id','day']).cell_count)
        div_series = pd.Series(div, name='cell_perc')
        res = pd.concat([group.sort_values(by=['mouse_id', 'day']), div_series], axis=1)
        with_percent_df = with_percent_df.append(res, ignore_index=True)


    full_df = with_percent_df.merge(
        GFP_df,
        on=['mouse_id', 'cell_type', 'day', 'month']
    ).merge(
        donor_df,
        on=['mouse_id', 'cell_type', 'day', 'month']
    )

    unbarcoded_df = full_df.assign(percent_engraftment = lambda x: calculate_unbarcoded_abundance(x))
    unbarcoded_df['code'] = 'unbarcoded'

    host_clones_df = full_df.assign(percent_engraftment = lambda x: calculate_host_clones(x))
    host_clones_df['code'] = 'host_clones'

    rest_of_clones_df = pd.concat([unbarcoded_df, host_clones_df], ignore_index=True)
    rest_of_clones_df = rest_of_clones_df.merge(group_info_df, how='left', on=['mouse_id'])
    rest_of_clones_df.to_csv(args.output_dir + os.sep + 'rest_of_clones_abundance_long.csv')

    with_baseline_counts_df = calculate_baseline_counts(
        rest_of_clones_df[['code', 'mouse_id', 'cell_type', 'day', 'month', 'group', 'percent_engraftment']],
        WBC_df,
        baseline_column=args.timepoint_col,
        baseline_timepoint=args.baseline_timepoint
        )
    norm_data_df = normalize_to_baseline_counts(with_baseline_counts_df)

    print('Calculating lineage bias...')
    lineage_bias_df = create_lineage_bias_df(norm_data_df)


    lineage_bias_file_name = args.output_dir + os.sep + 'rest_of_clones_lineage_bias.csv'
    print('Done.\n')
    print('\nSaving Lineage Bias Data To: ' + lineage_bias_file_name)
    lineage_bias_df.to_csv(lineage_bias_file_name, index=False)


if __name__ == '__main__':
    main()
