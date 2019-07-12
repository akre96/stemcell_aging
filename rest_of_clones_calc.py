import os
import argparse
import numpy as np
import pandas as pd
from lineage_bias import parse_wbc_count_file, \
    calculate_baseline_counts, normalize_to_baseline_counts, \
    create_lineage_bias_df

def calculate_unbarcoded_abundance(row) -> np.float:
    return 100 * ((row.donor_perc/100) * (1 - row.GFP_perc/100) * row.cell_perc/100)

def calculate_total_clones(row) -> np.float:
    return row.cell_perc

def calculate_host_clones(row) -> np.float:
    return 100 * ((1 - row.donor_perc/100) * row.cell_perc/100)

def main():
    """ Calculate and save lineage bias
    """

    parser = argparse.ArgumentParser(description="Calculate bias and abundance of unbarcoded and host cells")
    parser.add_argument('--gfp', dest='gfp_file', help='Path to txt file containing gfp percent data', required=True)
    parser.add_argument('--wbcs', dest='wbcs_file', help='Path to txt containing FACS cell count data', required=True)
    parser.add_argument('--donor', dest='donor_file', help='Path to file containing donor chimerism data', required=True)
    parser.add_argument('--group', dest='group_file', help='Path to csv file with columns for mouse_id and group', required=False)
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', required=True)
    parser.add_argument('-t', '--timepoint-col', dest='timepoint_col', help='Column to look for time in', required=True)
    parser.add_argument('-b', '--baseline-timepoint', dest='baseline_timepoint', help='Timepoint to set as baseline value for normalization', required=False)
    parser.add_argument('--lymph', '--lymphoid-cell-type', dest='lymphoid_cell_type', help='Cell to use for lymphoid representative', default='b', required=False)
    parser.add_argument('--myel', '--myeloid-cell-type', dest='myeloid_cell_type', help='Cell to use for myeloid representative', default='gr', required=False)

    args = parser.parse_args()

    if args.group_file:
        group_info_df = pd.read_csv(args.group_file)
    WBC_df = parse_wbc_count_file(args.wbcs_file, analyzed_cell_types=[args.myeloid_cell_type, args.lymphoid_cell_type, 'wbc'])
    GFP_df = parse_wbc_count_file(args.gfp_file).rename(columns={'cell_count':'GFP_perc'})
    donor_df = parse_wbc_count_file(args.donor_file).rename(columns={'cell_count':'donor_perc'})

    cells_df = WBC_df[WBC_df.cell_type != 'wbc']
    wbcs_df = WBC_df[WBC_df.cell_type == 'wbc']

    with_percent_df = pd.DataFrame()
    for name, group in cells_df.groupby(['cell_type']):
        if len(group) != len(wbcs_df):
            print(group)
            print(wbcs_df)
            raise ValueError('WBC DF not same length as cell group')
        div = 100 * np.divide(group.sort_values(by=['mouse_id', 'day']).cell_count, wbcs_df.sort_values(by=['mouse_id', 'day']).cell_count)
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

    unbarcoded_df = full_df.assign(percent_engraftment=lambda x: calculate_unbarcoded_abundance(x))
    unbarcoded_df['code'] = 'Unbarcoded Clones'

    host_clones_df = full_df.assign(percent_engraftment=lambda x: calculate_host_clones(x))
    host_clones_df['code'] = 'Host Clones'

    total_clones_df = full_df.assign(percent_engraftment=lambda x: calculate_total_clones(x))
    total_clones_df['code'] = 'Total Clones'

    rest_of_clones_df = pd.concat([total_clones_df, unbarcoded_df, host_clones_df], ignore_index=True)
    if args.group_file:
        rest_of_clones_df = rest_of_clones_df.merge(group_info_df, how='left', on=['mouse_id'])
    else:
        rest_of_clones_df['group'] = 'No Groups'
    rest_of_clones_df.to_csv(args.output_dir + os.sep + 'rest_of_clones_abundance_long.csv')

    if args.baseline_timepoint == 'None':
        base_time_point = None
    elif args.baseline_timepoint:
        base_time_point = int(args.baseline_timepoint)
    else:
        base_time_point = input_df['day'].min()

    with_baseline_counts_df = calculate_baseline_counts(
        rest_of_clones_df[['code', 'mouse_id', 'cell_type', 'day', 'month', 'group', 'percent_engraftment']],
        WBC_df,
        baseline_column=args.timepoint_col,
        baseline_timepoint=base_time_point
        )
    norm_data_df = normalize_to_baseline_counts(with_baseline_counts_df)

    print('Calculating lineage bias...')
    lineage_bias_df = create_lineage_bias_df(
        norm_data_df,
        lymphoid_cell_type=args.lymphoid_cell_type,
        myeloid_cell_type=args.myeloid_cell_type,
        )


    lineage_bias_file_name = args.output_dir + os.sep + 'rest_of_clones_lineage_bias.csv'
    print('Done.\n')
    print('\nSaving Lineage Bias Data To: ' + lineage_bias_file_name)
    lineage_bias_df.to_csv(lineage_bias_file_name, index=False)


if __name__ == '__main__':
    main()
