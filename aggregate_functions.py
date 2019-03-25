""" Commonly used data transform functions for analysis of step7 output data

"""
from typing import List, Dict
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

def filter_cell_type_threshold(input_df: pd.DataFrame,
                               thresholds: Dict[str, float],
                               analyzed_cell_types: List[str],
                               threshold_column: str = "percent_engraftment",
                              ) -> pd.DataFrame:
    """ Fitlers input by threshold on one cell type

    Arguments:
        input_df {pd.DataFrame} -- Step 7 long format data
        thresholds {Dict[str, float]} -- dictionary of cell_type: threshold
        analyzed_cell_types {List[str]} -- cell types to search for

    Keyword Arguments:
        threshold_column {str} --  column on which to search (default: {"percent_engraftment"})

    Returns:
        pd.DataFrame -- Data Frame filtered by threshold column for each cell type
    """

    filtered_df = pd.DataFrame()
    for cell_type in analyzed_cell_types:
        cell_df = filter_threshold(input_df,
                                   threshold=thresholds[cell_type],
                                   analyzed_cell_types=[cell_type],
                                   threshold_column=threshold_column
                                  )
        filtered_df = filtered_df.append(cell_df)
    return filtered_df

def count_clones(input_df: pd.DataFrame) -> pd.DataFrame:
    """ Count unique clones per cell type

    Arguments:
        input_df {pd.DataFrame} -- long formatted step7 output

    Returns:
        pd.DataFrame -- DataFrame with columns 'mouse_id','day', 'cell_type', 'code' where
        'code' contains count of unique barcodes
    """

    clone_counts = pd.DataFrame(
        input_df.groupby(['mouse_id', 'day', 'month', 'cell_type'])['code'].nunique()
        ).reset_index()
    total_clone_counts = pd.DataFrame(input_df.groupby(['mouse_id', 'day', 'month'])['code'].nunique()).reset_index()
    total_clone_counts['cell_type'] = 'Total'
    clone_counts = clone_counts.append(total_clone_counts, sort=True)

    return clone_counts


def find_enriched_clones_at_time(input_df: pd.DataFrame,
                                 enrichment_month: int,
                                 enrichment_threshold: float,
                                 cell_type: str,
                                 threshold_column: str = 'percent_engraftment',
                                 lineage_bias: bool = False,
                                 ) -> pd.DataFrame:
    """ Finds clones enriched at a specific time point for a cell type

    Arguments:
        input_df {pd.DataFrame} -- long format data, formatted with filter_threshold()
        enrichment_month {int} -- month of interest
        threshold {int} -- threshold for significant engraftment
        cell_type {str} -- Cell type to select for
        threshold_column {str} -- column on which to apply threshold

    Keyword Arguments:
        lineage_bias {bool} -- Checks if running lineage bias data (default: False)

    Returns:
        pd.DataFrame -- DataFrame with only clones enriched at specified timepoint
    """

    if lineage_bias:
        filter_index = (input_df[threshold_column] > enrichment_threshold) & (input_df['month'] == enrichment_month)
    else:
        filter_index = (input_df[threshold_column] > enrichment_threshold) & (input_df['month'] == enrichment_month) & (input_df['cell_type'] == cell_type)

    enriched_at_month_df = input_df.loc[filter_index]
    enriched_clones = enriched_at_month_df['code']

    if lineage_bias:
        cell_df = input_df
    else:
        cell_df = input_df.loc[input_df.cell_type == cell_type]

    should_be_empty_index = (cell_df.month == enrichment_month) & (cell_df[threshold_column] < enrichment_threshold)
    stray_clones = cell_df[should_be_empty_index]['code'].unique()

    enriched_clones_df = cell_df.loc[(~cell_df['code'].isin(stray_clones)) & (cell_df['code'].isin(enriched_clones))]
    return enriched_clones_df

def combine_enriched_clones_at_time(input_df: pd.DataFrame, enrichement_month: int, threshold: float, analyzed_cell_types: List[str]) -> pd.DataFrame:
    """ wrapper of find_enriched_clones_at_time() to combine entries from multiple cell types
    
    Arguments:
        input_df {pd.DataFrame} -- data frame with month value
        enrichement_month {int} -- month to check enrichment at
        threshold {float} -- threshold for enrichment
        analyzed_cell_types {List[str]} -- cell types to analyze
    
    Returns:
        pd.DataFrame -- data frame of enriched clones in specified cell types and timepoint
    """

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

def clones_enriched_at_last_timepoint(input_df: pd.DataFrame, lineage_bias_df: pd.DataFrame, threshold: float = 0, cell_type: str = 'any', lineage_bias: bool = False, percentile: float = 0) -> pd.DataFrame:
    """ Finds clones enriched at last timepoint for clone
    
    Arguments:
        input_df {pd.DataFrame} -- long format step 7 output
        lineage_bias_df {pd.DataFrame} -- lineage bias data output,
        set to empty dataframe if not analyzing lineage bias data
    
    Keyword Arguments:
        threshold {float} -- if analyzing absolute threshold values set (default: {0})
        cell_type {str} --  which cell type to apply threshold agains (default: {'any'})
        lineage_bias {bool} --  set true if analyzing lineage bias data(default: {False})
        percentile {float} --  if not 0, looks for top percentile instead of absolute threshold (default: {0})
    
    Returns:
        pd.DataFrame -- [description]
    """

    groupby_cols = ['mouse_id', 'code']
    if percentile:
        if lineage_bias:
            thresholds = find_top_percentile_threshold(input_df, percentile, ['gr', 'b'])
            filtered_df = filter_cell_type_threshold(input_df, thresholds, ['gr', 'b'])
        else:
            filtered_df = find_top_percentile_threshold(input_df, percentile, ['gr','b'])
            groupby_cols.append('cell_type')
    else:
        if lineage_bias:
            if cell_type == 'any':
                filtered_df = lineage_bias_df.loc[(lineage_bias_df['gr_percent_engraftment'] >= threshold) | (lineage_bias_df['b_percent_engraftment'] >= threshold)]
            else:
                filtered_df = lineage_bias_df.loc[(lineage_bias_df[cell_type + '_percent_engraftment'] >= threshold)]
        else:
            filtered_df = filter_threshold(input_df, threshold, [cell_type])
            groupby_cols.append('cell_type')

    # get max month for clones
    grouped_df = pd.DataFrame(filtered_df.groupby(by=groupby_cols).month.max()).reset_index()
    if lineage_bias:
        filtered_for_enrichment = lineage_bias_df.merge(grouped_df['code'], how='inner', on=['code'])
    else:
        filtered_for_enrichment = input_df.merge(grouped_df['code'], how='inner', on=['code'])

    return filtered_for_enrichment

def filter_mice_with_n_timepoints(input_df: pd.DataFrame, n_timepoints: int = 4) -> pd.DataFrame:
    """ Finds mice with desired number of timepoints.
    Used primarily to only select mice with all four time points

    Arguments:
        input_df {pd.DataFrame} -- Step 7 long format data

    Keyword Arguments:
        n_timepoints {int} -- number of timepoints desired (default: {4})

    Returns:
        pd.DataFrame -- [description]
    """

    output_df = pd.DataFrame()
    for _, group in input_df.groupby(['mouse_id']):
        if group.month.nunique() >= n_timepoints:
            output_df = output_df.append(group)
    return output_df

def find_top_percentile_threshold(input_df: pd.DataFrame, percentile: float, cell_types: List[str] = ['gr', 'b']) -> Dict[str, float]:
    """ Finds threshold to ge top percentile of each cell type

    Arguments:
        input_df {pd.DataFrame} -- step 7 long format output
        percentile {float} -- percentile to apply

    Keyword Arguments:
        cell_types {List[str]} -- cell types to group by (default: {['gr', 'b']})

    Returns:
        Dict[str, float] -- dictionary of format cell_type: threshold
    """

    cell_type_thresholds = {}
    for cell_type in cell_types:
        cell_df = input_df.loc[input_df.cell_type == cell_type]
        cell_type_thresholds[cell_type] = cell_df.quantile(percentile).percent_engraftment
    return cell_type_thresholds


def export_wide_formatted_clone_counts(input_file: pd.DataFrame = 'Ania_M_all_percent-engraftment_100818_long.csv',
                                       thresholds: List[float] = [0.0, 0.01, 0.02, 0.2, 0.5],
                                       outdir: str = '/home/sakre/Data/clone_counts_long',
                                       analyzed_cell_types: List[str] = ['gr', 'b'],
                                      ):
    """ Outputs clone count csv file at multiple thresholds.
    """
    input_df = pd.read_csv(input_file)
    for threshold in thresholds:
        filter_df = filter_threshold(input_df, threshold, analyzed_cell_types)
        clone_counts = count_clones(filter_df)
        wide_counts = long_to_wide_data(clone_counts, 'code')
        columns = wide_counts.columns.tolist()
        columns.insert(0, columns.pop(columns.index('mouse_id')))
        wide_counts = wide_counts[columns]
        fname = outdir + os.sep + 'clone_counts_t' + str(threshold).replace('.', '-') + '.csv'
        wide_counts.to_csv(fname, index=False)

def count_clones_at_percentile(input_df: pd.DataFrame, percentile: float, analyzed_cell_types: List[str] = ['gr','b']) -> pd.DataFrame:
    """ Wrapper function to count clones when applying a percentile based threshold

    Arguments:
        input_df {pd.DataFrame} -- step 7 long form data
        percentile {float} -- percentile to threshold for

    Keyword Arguments:
        analyzed_cell_types {List[str]} -- cell_types to analyze (default: {['gr','b']})

    Returns:
        pd.DataFrame -- clone_counts dataframe
    """

    thresholds = find_top_percentile_threshold(input_df, percentile, cell_types=analyzed_cell_types)
    filtered_df = filter_cell_type_threshold(input_df, thresholds=thresholds, threshold_column='percent_engraftment', analyzed_cell_types=analyzed_cell_types)
    return count_clones(filtered_df)

#INPUT_DF = pd.read_csv('/home/sakre/Code/stemcell_aging/Ania_M_all_percent-engraftment_100818_long.csv')
#FILTER_DF = filter_threshold(INPUT_DF, 0.01, ['gr', 'b'])
