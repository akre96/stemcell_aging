"""Calculates lineage bias for clones
"""
from typing import List
from math import pi, sin, atan
import sys
import pandas as pd
from aggregate_functions import filter_threshold

def calc_angle(gr: float, b: float) -> float:
    towards_myloid_angle = atan(gr/b)
    return towards_myloid_angle

def calc_bias_from_angle(theta: float) -> float:
    myloid_bias = sin(2 * (theta - (pi/4)))
    return myloid_bias

def calc_bias(gr: float, b: float) -> float:
    return calc_bias_from_angle(calc_angle(gr, b))

def calculate_baseline(input_df: pd.DataFrame, baseline_timepoint: int, baseline_column: str = 'month') -> pd.DataFrame:
    # Filter by timepoint
    timepoint_df = input_df.loc[input_df[baseline_column] == baseline_timepoint]

    # Find max
    baseline_max_df = pd.DataFrame(timepoint_df.groupby(['mouse_id', 'cell_type']).percent_engraftment.max()).reset_index()

    # Rename max column to baseline_max --> current columns are 'mouse_id', 'cell_type', 'baseline_max'
    baseline_max_df = baseline_max_df.rename({'percent_engraftment': 'baseline_max'}, axis='columns')

    #Append to input, each row should have 'baseline_max' column
    with_baseline_df = input_df.merge(baseline_max_df, how='outer', on=['mouse_id', 'cell_type'])

    return with_baseline_df

def normalize_to_baseline(with_baseline_df: pd.DataFrame):
    norm_data_df = with_baseline_df.assign(norm_percent_engraftment=lambda row: row.percent_engraftment/row.baseline_max)
    return norm_data_df

def normalize_input(input_df: pd.DataFrame,
                    baseline_timepoint: int = 4,
                    baseline_column: str = 'month',
                    analyzed_cell_types: List[str] = ['gr', 'b'],
                    present_threshold: float = 0.01,
                   ) -> pd.DataFrame:

    # Filter for only present clones
    present_df = filter_threshold(input_df,
                                  present_threshold,
                                  analyzed_cell_types)

    # Add baseline information for normalization
    with_baseline_df = calculate_baseline(present_df,
                                          baseline_timepoint=baseline_timepoint,
                                          baseline_column=baseline_column)

    # Add normalized percent engraftment
    norm_data_df = normalize_to_baseline(with_baseline_df)
    return norm_data_df

def create_lineage_bias_df(norm_data_df: pd.DataFrame) -> pd.DataFrame:
    lineage_bias_columns = ['user', 'mouse_id', 'code', 'day', 'month', 'lineage_bias', 'has_null']
    lineage_bias_df = pd.DataFrame(columns=lineage_bias_columns)

    for _, group in norm_data_df.groupby(['mouse_id', 'code', 'day']):
        has_both_types = True
        for cell_type in ['gr', 'b']:
            if group[group.cell_type == cell_type].empty:
                has_both_types = False

        if len(group) < 2:
            has_both_types = False

        if has_both_types:
            new_row = pd.DataFrame(columns=lineage_bias_columns)
            gr = group[group.cell_type == 'gr'].norm_percent_engraftment.values[0]
            b = group[group.cell_type == 'b'].norm_percent_engraftment.values[0]
            new_row['has_null'] = [group.norm_percent_engraftment.isnull().values.any()]
            new_row['lineage_bias'] = [calc_bias(gr, b)]
            new_row['code'] = group.code.unique()
            new_row['user'] = group.user.unique()
            new_row['day'] = group.day.unique()
            new_row['month'] = group.month.unique()
            new_row['mouse_id'] = group.mouse_id.unique()
            new_row['group'] = group['group'].unique()
            if(len(new_row.code) > 1):
                print('ERROR MULTIPLE CODES FOUND FOR GROUP')
                sys.exit(1)
            lineage_bias_df = lineage_bias_df.append(new_row, sort=False, ignore_index=True)
    return lineage_bias_df

def main():
    input_df = pd.read_csv('Ania_M_all_percent-engraftment_100818_long.csv')

    norm_data_df = normalize_input(input_df)
    lineage_bias_df = create_lineage_bias_df(norm_data_df)
    print(lineage_bias_df)
    #lineage_bias_df.to_csv('lineage_bias.csv', index=False)
            

if __name__ == '__main__':
    main()
