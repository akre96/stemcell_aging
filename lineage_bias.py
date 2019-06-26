"""Calculates lineage bias for clones

Example command using abundance cutoff:
python lineage_bias.py \
    -i Ania_M_all_percent-engraftment_100818_long.csv \
    -c ~/Data/aging_data/OT\ 2.0\ WBCs\ D122\ D274\ D365\ D420.txt \
    -o ./output/lineage_bias \
    -a 80
"""
import argparse
from typing import List
import os
from math import pi, sin, atan
import pandas as pd
from aggregate_functions import filter_threshold, \
    calculate_thresholds_sum_abundance, \
    filter_clones_threshold_anytime
from parse_facs_data import parse_wbc_count_file

def calc_angle(m_value: float, l_value: float) -> float:
    """ Calculates angle towards myeloid bias

    Arguments:
        m_value {float} -- myeloid normalized percent engraftment
        l_value {float} -- lymphoid normalized percent engraftment

    Returns:
        float -- angle in radians towards myeloid bias, range [0,pi/2]
    """

    if l_value == 0.0:
        if m_value == 0.0:
            return pi/4
        return pi/2

    towards_myloid_angle = atan(m_value/l_value)
    return towards_myloid_angle

def calc_bias_from_angle(theta: float) -> float:
    """ Calculates myeloid (+) lymhoid (-) bias from myeloid bias angle
    Output ranges from -1 (lymphoid) to +1 (myloid).

    Arguments:
        theta {float} -- angle of bias towards myeloid range [0, pi/2]

    Returns:
        float -- angle transformed to range [-1, 1]
    """

    myloid_bias = sin(2 * (theta - (pi/4)))
    return myloid_bias

def calc_bias(m_value: float, l_value: float) -> float:
    """ Calculates myeloid (+) lymhoid (-) bias from normalized percent engraftment

    Arguments:
        m_value {float} -- myeloid normalized percent engraftment
        l_value {float} -- lymphoid normalized percent engraftment

    Returns:
        float -- bias from [-1, 1] with myeloid (+) and lymphoid (-)
    """

    return calc_bias_from_angle(calc_angle(m_value, l_value))

def calculate_baseline_max(present_df: pd.DataFrame,
                           baseline_timepoint: int = 4,
                           baseline_column: str = 'month') -> pd.DataFrame:
    """ Calculates baseline maximum for normalizing percent engraftment

    For a cell_type and mouse, its percent engraftment is divided by the
    maximum value for that cell_type/mouse_id at the baseline_timepoint.

    If there is no value for the cell_type-mouse_id, raises a ValueError

    Arguments:
        present_df {pd.DataFrame} -- long step7_output filtered for present clones by
                                     filter_threshold()

    Keyword Arguments:
        baseline_timepoint {int} -- time at which to calculate baseline_max (default: {4})
        baseline_column {str} --  column to look for timepoint (default: {'month'})

    Raises:
        ValueError -- if no baseline_max found for a clone

    Returns:
        pd.DataFrame -- present_df with column for baseline_max
    """

    # Filter by timepoint
    timepoint_df = present_df.loc[present_df[baseline_column] == baseline_timepoint]

    # group by mouse_id cell_type, find min month
    baseline_months_df = pd.DataFrame(present_df.groupby(['mouse_id', 'cell_type'])[baseline_column].min()).reset_index()

    # Check that all min_months for a cell_type/mouse are the desired baseline timepoint
    if not baseline_months_df[baseline_months_df.month != baseline_timepoint].empty:
        print('Error: No baseline_max value establishable')
        print('The following data lacks baseline_max information:')
        print(baseline_months_df[baseline_months_df.month != baseline_timepoint])
        raise ValueError('Error: No baseline_max value establishable')

    # Find max percent engraftment to normalize by
    baseline_max_df = pd.DataFrame(timepoint_df.groupby(['mouse_id', 'cell_type']).percent_engraftment.max()).reset_index()

    # Rename max column to baseline_max --> current columns are 'mouse_id', 'cell_type', 'baseline_max'
    baseline_max_df = baseline_max_df.rename({'percent_engraftment': 'baseline_max'}, axis='columns')

    #Append to input, each row should have 'baseline_max' column
    with_baseline_max_df = present_df.merge(baseline_max_df, how='outer', on=['mouse_id', 'cell_type'])

    return with_baseline_max_df

def normalize_to_baseline_max(with_baseline_max_df: pd.DataFrame) -> pd.DataFrame:
    """ uses baseline_max to add another column for normalized percent engraftment

    Arguments:
        with_baseline_max_df {pd.DataFrame} -- present_df with column for baseline_max

    Returns:
        pd.DataFrame -- dataframe with norm_percent_engraftment column calculated from baseline_max
    """

    norm_data_df = with_baseline_max_df.assign(norm_percent_engraftment=lambda row: row.percent_engraftment/row.baseline_max)
    return norm_data_df

def normalize_input_to_baseline_max(input_df: pd.DataFrame,
                                    analyzed_cell_types: List[str],
                                    baseline_timepoint: int = 4,
                                    baseline_column: str = 'month',
                                    present_threshold: float = 0.01,
                                   ) -> pd.DataFrame:
    """ Wrapper function handling filtering, baseline calculation, and normalization of input

    Arguments:
        input_df {pd.DataFrame} -- unfiltered step7 long format

    Keyword Arguments:
        baseline_timepoint {int} -- timepoint to use for baseline (default: {4})
        baseline_column {str} --  column to find timepoint in (default: {'month'})
        analyzed_cell_types {List[str]} -- cell_types to filter for 
        present_threshold {float} -- threshold to mark clones present (default: {0.01})

    Returns:
        pd.DataFrame -- filtered data frame with normalized percent engraftment
    """


    # Filter for only present clones
    present_df = filter_threshold(input_df,
                                  present_threshold,
                                  analyzed_cell_types)

    # Add baseline information for normalization
    with_baseline_max_df = calculate_baseline_max(present_df,
                                                  baseline_timepoint=baseline_timepoint,
                                                  baseline_column=baseline_column)

    # Add normalized percent engraftment
    norm_data_df = normalize_to_baseline_max(with_baseline_max_df)
    return norm_data_df

def create_lineage_bias_df(
        norm_data_df: pd.DataFrame,
        lymphoid_cell_type: str,
        myeloid_cell_type: str
    ) -> pd.DataFrame:
    """ Calculates Lineage Bias for all present clones

    Arguments:
        norm_data_df {pd.DataFrame} -- long format input with percent engraftments normalized
        lymphoid_cell_type {str} -- Cell type to use as representative for lymphoid
        myeloid_cell_type {str} -- cell type to use as represenetative for myeloid

    Raises:
        ValueError -- Raises error if ant clone-mouse-day has more than expected (2) cell types
        ValueError -- Raises error if multiple clones found after groupby mouse-code-day
        (should never happen)

    Returns:
        pd.DataFrame -- DataFrame of lineage bias for clones over time
    """

    lineage_bias_columns = ['mouse_id',
                            'group',
                            'code',
                            'day',
                            'lineage_bias',
                            'myeloid_percent_engraftment',
                            'lymphoid_percent_engraftment',
                            'has_null']

    lineage_bias_df = pd.DataFrame(columns=lineage_bias_columns)

    # analyze based on each pair of cells for a mouse-code-day
    for _, group in norm_data_df.groupby(['mouse_id', 'code', 'day']):

        # If cell_type not found (filtered out), uses 0 as normalized percent engraftment
        if group[group.cell_type == myeloid_cell_type].empty:
            m_value = 0.0
            myeloid_abundance = 0.0
        else:
            m_value = group[group.cell_type == myeloid_cell_type].norm_percent_engraftment.values[0]
            myeloid_abundance = group[group.cell_type == myeloid_cell_type].percent_engraftment.values[0]
        if group[group.cell_type == lymphoid_cell_type].empty:
            l_value = 0.0
            lymphoid_abundance = 0.0
        else:
            l_value = group[group.cell_type == lymphoid_cell_type].norm_percent_engraftment.values[0]
            lymphoid_abundance = group[group.cell_type == lymphoid_cell_type].percent_engraftment.values[0]

        new_row = pd.DataFrame(columns=lineage_bias_columns)
        new_row['has_null'] = [group.norm_percent_engraftment.isnull().values.any()]
        new_row['lineage_bias'] = [calc_bias(m_value, l_value)]
        new_row['myeloid_percent_abundance'] = myeloid_abundance
        new_row['lymphoid_percent_abundance'] = lymphoid_abundance
        new_row['sum_percent_abundance'] = myeloid_abundance + lymphoid_abundance
        new_row['code'] = group.code.unique()
        new_row['day'] = group.day.unique()
        new_row['mouse_id'] = group.mouse_id.unique()
        if 'group' in group.columns:
            new_row['group'] = group['group'].unique()
        else:
            new_row['group'] = 'None'

        # Ensures row added for unique clone
        if len(new_row.code) > 1:
            raise ValueError('Multiple codes found for group')

        lineage_bias_df = lineage_bias_df.append(new_row, sort=False, ignore_index=True)

    return lineage_bias_df


def calculate_baseline_counts(present_df: pd.DataFrame,
                              cell_counts_df: pd.DataFrame,
                              baseline_timepoint: int = 4,
                              baseline_column: str = 'month'
                             ) -> pd.DataFrame:
    """ Appends column of cell counts to step7 long form data

    Arguments:
        present_df {pd.DataFrame} -- Dataframe of step7 data filtered for presence
        cell_counts_df {pd.DataFrame} -- Dataframe from FACS cell count data

    Keyword Arguments:
        baseline_timepoint {int} -- timepoint to use as reference (default: {4})
        baseline_column {str} --  column to look for timepoint in (default: {'month'})

    Returns:
        pd.DataFrame -- present_df with column of cell_count used in normalization
    """


    if baseline_timepoint is None:
        with_baseline_counts_df = present_df.merge(cell_counts_df[['mouse_id', 'cell_type', 'cell_count', baseline_column]], how='left', on=['mouse_id', 'cell_type', baseline_column])
    else:
        timepoint_df = cell_counts_df.loc[cell_counts_df[baseline_column] == baseline_timepoint]
        with_baseline_counts_df = present_df.merge(timepoint_df[['mouse_id', 'cell_type', 'cell_count']], how='left', on=['mouse_id', 'cell_type'])

    return with_baseline_counts_df

def normalize_to_baseline_counts(with_baseline_counts_df: pd.DataFrame) -> pd.DataFrame:
    """ Use count information to normalize percent engraftment

    Arguments:
        with_baseline_counts_df {pd.DataFrame} -- dataframe output of calculate_baseline_counts

    Returns:
        pd.DataFrame -- dataframe with norm_percent_engraftment column
    """

    norm_data_df = with_baseline_counts_df.assign(norm_percent_engraftment=lambda row: row.percent_engraftment/row.cell_count)
    return norm_data_df

def get_bias_change(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        save_err: bool = False
    ) -> pd.DataFrame:
    """ Calculate bias change between a clones first and last timepoint
    
    Arguments:
        lineage_bias_df {pd.DataFrame} -- Lineage Bias Dataframe
    
    Keyword Arguments:
        timepoint_col {str} -- timepoint to analyze (default: {'day'})
        save_err {bool} -- Whether to save any codes across multiple mice (default: {False})
    
    Returns:
        pd.DataFrame -- bias_change dataframe
    """

    bias_change_cols = ['code', 'mouse_id', 'group', 'bias_change', 'time_change', 'time_unit', 'first_timepoint', 'last_timepoint']
    bias_change_df = pd.DataFrame(columns=bias_change_cols)
    same_code_df = pd.DataFrame()
    for _, group in lineage_bias_df.groupby('code'):
        if len(group) < 2:
            continue
        bias_change_row = pd.DataFrame(columns=bias_change_cols)
        first_timepoint = group.loc[group[timepoint_col].astype(int).idxmin()]
        last_timepoint = group.loc[group[timepoint_col].astype(int).idxmax()]
        if first_timepoint.code != last_timepoint.code:
            ValueError('Not grouped by same code/clone')
        if first_timepoint[timepoint_col] == last_timepoint[timepoint_col]:
            same_code_df = same_code_df.append(group)
            continue

        bias_change_row.bias_change = [last_timepoint.lineage_bias - first_timepoint.lineage_bias]
        bias_change_row.time_change = [last_timepoint[timepoint_col] - first_timepoint[timepoint_col]]
        bias_change_row.first_timepoint = [first_timepoint[timepoint_col]]
        bias_change_row.last_timepoint = [last_timepoint[timepoint_col]]
        bias_change_row.code = first_timepoint.code
        bias_change_row.mouse_id = first_timepoint.mouse_id
        bias_change_row.group = first_timepoint.group
        bias_change_df = bias_change_df.append(bias_change_row, ignore_index=True)

    if save_err:
        same_code_df.to_csv('same_code_mice_lineage_bias.csv')

    return bias_change_df

def main():
    """ Calculate and save lineage bias
    """

    parser = argparse.ArgumentParser(description="Calculate lineage bias and change in lineage bias over time at thresholds")
    parser.add_argument('-i', '--input', dest='input', help='Path to file containing consolidated long format step7 output', required=True)
    parser.add_argument('-c', '--counts', dest='counts_file', help='Path to txt containing FACS cell count data', required=True)
    parser.add_argument('-t', '--threshold', dest='threshold', help='Threshold to filter presence of cells by for percent_engraftment', default=.0, type=float)
    parser.add_argument('-a', '--abundance_cutoff', dest='abundance_cutoff', help='Set thresholds based on desired cumulative abundance based cutoff', type=float, required=False)
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='output/lineage_bias')
    parser.add_argument('-l', '--bias-only', dest='bias_only', help='Set flag if you only want to calculate lineage bias, not its change', action="store_true")
    parser.add_argument('-d', '--by-day', dest='by_day', help='Calculations done using day column and day 127', action="store_true")
    parser.add_argument('--monocyte', dest='monocyte', help='Calculations done using monocytes instead of granulocytes for myeloid', action="store_true")
    parser.add_argument('--baseline-timepoint', dest='baseline_timepoint', help='baseline time point for lineage bias. None if normalize at each time', required=False)

    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    cell_count_data = parse_wbc_count_file(args.counts_file)
    if args.monocyte:
        myeloid_cell_type = 'mo'
    else:
        myeloid_cell_type = 'gr'

    lymphoid_cell_type = 'b'

    time_column = 'day'
    if args.baseline_timepoint == 'None':
        base_time_point = None
    elif args.baseline_timepoint:
        base_time_point = int(args.baseline_timepoint)
    else:
        base_time_point = input_df[time_column].min()

    if args.abundance_cutoff:
        print('Lineage Bias Calculated using baseline by ' + time_column + ' at point: ' + str(base_time_point))
        abundance_cutoff = args.abundance_cutoff
        _, thresholds = calculate_thresholds_sum_abundance(
            input_df,
            abundance_cutoff=abundance_cutoff,
            analyzed_cell_types=[myeloid_cell_type, lymphoid_cell_type],
            timepoint_col=time_column
        )
        print('Calculating Lineage Bias and Change for cumulative abundance cutoff: ' + str(abundance_cutoff) + '\n')

        print('Filtering for present clones (any time point clone > thresholds)...')
        present_df = filter_clones_threshold_anytime(input_df, thresholds, [myeloid_cell_type, lymphoid_cell_type])
        print('Done.\n')
    else:
        print('Lineage Bias Calculated using baseline by ' + time_column + 'at point: ' + str(base_time_point))
        threshold = args.threshold
        print('Calculating Lineage Bias and Change for presence threshold of: ' + str(threshold) + '\n')

        print('Filtering for present clones...')
        present_df = filter_threshold(input_df, threshold, [myeloid_cell_type, lymphoid_cell_type])
        print('Done.\n')

    print('Normalizing data...')
    with_baseline_counts_df = calculate_baseline_counts(
        present_df,
        cell_count_data,
        baseline_column=time_column,
        baseline_timepoint=base_time_point
        )
    norm_data_df = normalize_to_baseline_counts(with_baseline_counts_df)
    print('Done.\n')

    print('Calculating lineage bias...')
    if args.abundance_cutoff:
        fname_suffix = '_a' + str(abundance_cutoff).replace('.', '-') + '_from-counts.csv'
    else:
        fname_suffix = '_t' + str(threshold).replace('.', '-') + '_from-counts.csv'
    lineage_bias_df = create_lineage_bias_df(norm_data_df, lymphoid_cell_type, myeloid_cell_type)
    lineage_bias_file_name = args.output_dir + os.sep + 'lineage_bias' + fname_suffix
    print('Done.\n')
    print('\nSaving Lineage Bias Data To: ' + lineage_bias_file_name)
    lineage_bias_df.to_csv(lineage_bias_file_name, index=False)


    if not args.bias_only:
        print('Calculating change in lineage bias...')
        bias_change_df = get_bias_change(lineage_bias_df, time_column)
        bias_change_file_name = args.output_dir + os.sep + 'bias_change' + fname_suffix
        print('Done.\n')
        print('\nSaving Bias Change Data To: ' + bias_change_file_name)
        bias_change_df.to_csv(bias_change_file_name, index=False)


if __name__ == '__main__':
    main()
