""" Commonly used data transform functions for analysis of step7 output data

"""
from typing import List
import os
import pandas as pd

def filter_threshold(input_df: pd.DataFrame,
                     threshold: float,
                     analyzed_cell_types: List[str],
                     threshold_column: str = "percent_engraftment",
                     ) -> pd.DataFrame:
    """Filter dataframe based on numerical thresholds, adds month column

    Arguments:
        input_df {pd.DataFrame} -- Input dataframe
        threshold {float} -- minimum value to allowed in output dataframe
        analyzed_cell_types {List[str]} -- List of cells to filter for
        threshold_column {str} -- column to filter by

    Returns:
        pd.DataFrame -- thresholded dataframe
    """

    analyzed_cell_types_df = input_df.loc[input_df.cell_type.isin(analyzed_cell_types)]
    threshold_filtered_df = analyzed_cell_types_df[analyzed_cell_types_df[threshold_column] > threshold]
    threshold_filtered_df['month'] = pd.to_numeric(round(threshold_filtered_df['day']/30), downcast='integer')
    return threshold_filtered_df


def count_clones(input_df: pd.DataFrame) -> pd.DataFrame:
    """ Count unique clones per cell type

    Arguments:
        input_df {pd.DataFrame} -- long formatted step7 output

    Returns:
        pd.DataFrame -- DataFrame with columns 'mouse_id','day', 'cell_type', 'code' where
        'code' contains count of unique barcodes
    """

    clone_counts = pd.DataFrame(
        input_df.groupby(['mouse_id', 'day', 'cell_type'])['code'].nunique()
        ).reset_index()
    total_clone_counts = pd.DataFrame(input_df.groupby(['mouse_id', 'day'])['code'].nunique()).reset_index()
    total_clone_counts['cell_type'] = 'Total'
    clone_counts = clone_counts.append(total_clone_counts, sort=True)

    return clone_counts


def find_enriched_clones_at_time(input_df: pd.DataFrame,
                                 enrichment_month: int,
                                 enrichment_threshold: float,
                                 cell_type: str,
                                 threshold_column: str = 'percent_engraftment',
                                 ) -> pd.DataFrame:
    """ Finds clones enriched at a specific time point

    Arguments:
        input_df {pd.DataFrame} -- long format data, formatted with filter_threshold()
        enrichment_month {int} -- month of interest
        threshold {int} -- threshold for significant engraftment
        cell_type {str} -- Cell type to select for
        threshold_column {str} -- column on which to apply threshold

    Returns:
        pd.DataFrame -- [description]
    """

    filter_index = (input_df[threshold_column] > enrichment_threshold) & (input_df['month'] == enrichment_month) & (input_df['cell_type'] == cell_type)
    enriched_at_month_df = input_df.loc[filter_index]
    enriched_clones = enriched_at_month_df['code']

    cell_df = input_df.loc[input_df.cell_type == cell_type]
    should_be_empty_index = (cell_df.month == enrichment_month) & (cell_df.percent_engraftment < enrichment_threshold)
    stray_clones = cell_df[should_be_empty_index]['code'].unique()

    enriched_clones_df = cell_df.loc[(~cell_df['code'].isin(stray_clones)) & (cell_df['code'].isin(enriched_clones))]
    return enriched_clones_df

def combine_enriched_clones_at_time(input_df: pd.DataFrame, enrichement_month: int, threshold: float, analyzed_cell_types: List[str]) -> pd.DataFrame:
    all_enriched_df = pd.DataFrame()
    for cell_type in analyzed_cell_types:
        enriched_cell_df = find_enriched_clones_at_time(input_df, enrichement_month, threshold, cell_type)
        all_enriched_df = all_enriched_df.append(enriched_cell_df)
    return all_enriched_df

def long_to_wide_data(input_df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """ Turns input file from long to wide format for analysis
    Rows are mouse ids, values are specified value from data_col.
    input_df can be anythin with the fields mouse_id, cell_type, day, [data_col]

    Arguments:
        input_df {pd.DataFrame} -- Data to reformat
        data_col {str} -- column to serve as value

    Returns:
        pd.DataFrame -- wide formatted dataframe
    """

    output_df = pd.DataFrame()
    for mouse in input_df.mouse_id.unique():
        new_row = pd.DataFrame()
        mouse_df = input_df.loc[input_df.mouse_id == mouse]
        for _, row in mouse_df.iterrows():
            col_name = 'D'+str(row.day)+'C'+str(row.cell_type)
            new_row[col_name] = [row[data_col]]
        new_row['mouse_id'] = mouse
        output_df = output_df.append(new_row, ignore_index=True)
    return output_df

def main():
    """ Outputs clone count csv file at multiple thresholds.
    """

    input_df = pd.read_csv('Ania_M_all_percent-engraftment_100818_long.csv')
    thresholds = [0.0, 0.01, 0.02, 0.2, 0.5]
    analyzed_cell_types = ['gr', 'b']
    outdir = '/home/sakre/Data/clone_counts_long'

    for threshold in thresholds:
        filter_df = filter_threshold(input_df, threshold, analyzed_cell_types)
        clone_counts = count_clones(filter_df)
        wide_counts = long_to_wide_data(clone_counts, 'code')
        columns = wide_counts.columns.tolist()
        columns.insert(0, columns.pop(columns.index('mouse_id')))
        wide_counts = wide_counts[columns]
        fname = outdir + os.sep + 'clone_counts_t' + str(threshold).replace('.', '-') + '.csv'
        wide_counts.to_csv(fname, index=False)

if __name__ == '__main__':
    main()
