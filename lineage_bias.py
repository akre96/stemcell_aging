"""Calculates lineage bias for clones

Example command using abundance cutoff:
python lineage_bias.py \
    -i Ania_M_all_percent-engraftment_100818_long.csv \
    -c ~/Data/aging_data/OT\ 2.0\ WBCs\ D122\ D274\ D365\ D420.txt \
    -o ./output/lineage_bias \
    --lymph b \
    --myel gr
"""
import argparse
import os
from math import pi, sin, atan
import pandas as pd
import progressbar
from parse_facs_data import parse_wbc_count_file
import warnings

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
                            'myeloid_percent_abundance',
                            'lymphoid_percent_abundance',
                            'has_null']

    lineage_bias_df = pd.DataFrame(columns=lineage_bias_columns)

    # analyze based on each pair of cells for a mouse-code-day
    for _, group in progressbar.progressbar(norm_data_df.groupby(['mouse_id', 'code', 'day'])):

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
                             ) -> pd.DataFrame:
    """ Appends column of cell counts to step7 long form data

    Arguments:
        present_df {pd.DataFrame} -- Dataframe of step7 data filtered for presence
        cell_counts_df {pd.DataFrame} -- Dataframe from FACS cell count data

    Keyword Arguments:
        baseline_timepoint {int} -- timepoint to use as reference (default: {4})

    Returns:
        pd.DataFrame -- present_df with column of cell_count used in normalization
    """

    baseline_column = 'day'
    if baseline_timepoint is None:
        print(' - Baseline time point not stet --> normalizing to mouse FACS data at each time point')
        with_baseline_counts_df = present_df.merge(cell_counts_df[['mouse_id', 'cell_type', 'cell_count', baseline_column]], how='left', on=['mouse_id', 'cell_type', baseline_column])
        print(with_baseline_counts_df)
    elif baseline_timepoint == 'by_mouse':
        min_tp = pd.DataFrame(
            present_df.groupby(['mouse_id'])[baseline_column].min()
        ).reset_index()
        tp_cell_counts = min_tp.merge(
            cell_counts_df,
            how='left',
        )
        print(' - Baseline set by mouse as follows:')
        for m, m_df in tp_cell_counts.groupby('mouse_id'):
            print('\t', m, baseline_column + ':', m_df[baseline_column].unique())

        with_baseline_counts_df = present_df.merge(
            tp_cell_counts[['mouse_id', 'cell_type', 'cell_count']].drop_duplicates(),
            how='left',
            validate='m:1'
        )

    else:
        timepoint_df = cell_counts_df.loc[cell_counts_df[baseline_column] == baseline_timepoint]
        with_baseline_counts_df = present_df.merge(timepoint_df[['mouse_id', 'cell_type', 'cell_count']], how='left', on=['mouse_id', 'cell_type'])

    na_mice = with_baseline_counts_df[with_baseline_counts_df.cell_count.isna()].mouse_id.unique()
    if na_mice:
        warnings.warn('WARNING: Mice with no baseline cell counts found: ' + str(na_mice))
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


def main(args):
    """ Calculate and save lineage bias
    """

    input_df = pd.read_csv(args.input)
    lymphoid_cell_type = args.lymphoid_cell_type
    myeloid_cell_type = args.myeloid_cell_type
    sep = '\t'
    if args.counts_file.split('.')[-1] == 'csv':
        sep = ','
    cell_count_data = parse_wbc_count_file(
        args.counts_file,
        analyzed_cell_types=[myeloid_cell_type, lymphoid_cell_type],
        sep=sep
        )

    time_column = 'day'
    if args.baseline_timepoint == 'None':
        base_time_point = None
    elif args.baseline_timepoint:
        if args.baseline_timepoint == 'by_mouse':
            base_time_point = 'by_mouse'
        else:
            base_time_point = int(args.baseline_timepoint)
    else:
        base_time_point = input_df[time_column].min()


    print('Lineage Bias Will Be Calculated using baseline by ' + time_column + ' at point: ' + str(base_time_point))

    print('Normalizing data...')
    with_baseline_counts_df = calculate_baseline_counts(
        input_df,
        cell_count_data,
        baseline_timepoint=base_time_point
        )
    norm_data_df = normalize_to_baseline_counts(with_baseline_counts_df)
    print('Done.\n')

    print('Calculating lineage bias...')
    lineage_bias_df = create_lineage_bias_df(norm_data_df, lymphoid_cell_type, myeloid_cell_type)
    lineage_bias_file_name = args.output_dir + os.sep + 'lineage_bias.csv'
    print('Done.\n')
    print('\nSaving Lineage Bias Data To: ' + lineage_bias_file_name)
    lineage_bias_df.to_csv(lineage_bias_file_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate lineage bias and change in lineage bias over time at thresholds")
    parser.add_argument('-i', '--input', dest='input', help='Path to file containing consolidated long format step7 output', required=True)
    parser.add_argument('-c', '--counts', dest='counts_file', help='Path to txt containing FACS cell count data', required=True)
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='output/lineage_bias')
    parser.add_argument('--baseline-timepoint', dest='baseline_timepoint', help='baseline time point for lineage bias. None if normalize at each time', required=False)
    parser.add_argument('--lymph', '--lymphoid-cell-type', dest='lymphoid_cell_type', help='Cell to use for lymphoid representative', default='b', required=False)
    parser.add_argument('--myel', '--myeloid-cell-type', dest='myeloid_cell_type', help='Cell to use for myeloid representative', default='gr', required=False)

    args = parser.parse_args()
    main(args)
