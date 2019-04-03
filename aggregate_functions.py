""" Commonly used data transform functions for analysis of step7 output data

"""
from typing import List, Dict, Tuple
import os
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    threshold_filtered_df = analyzed_cell_types_df.loc[analyzed_cell_types_df[threshold_column] > threshold]
    threshold_filtered_df = threshold_filtered_df.assign(month=lambda row: pd.to_numeric((row.day/30).round(), downcast='integer'))
    return threshold_filtered_df

def filter_cell_type_threshold(input_df: pd.DataFrame,
                               thresholds: Dict[str, float],
                               analyzed_cell_types: List[str],
                               threshold_column: str = "percent_engraftment",
                              ) -> pd.DataFrame:
    """ Fitlers input by threshold on each cell type

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
    if 'any' in thresholds.keys():
        return filter_threshold(input_df,
                                threshold=thresholds['any'],
                                analyzed_cell_types=analyzed_cell_types,
                                threshold_column=threshold_column,
                )
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

def combine_enriched_clones_at_time(
        input_df: pd.DataFrame,
        enrichment_month: int,
        thresholds: Dict[str, float],
        analyzed_cell_types: List[str],
        lineage_bias: bool = False
    ) -> pd.DataFrame:
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
        if lineage_bias:
            enriched_cell_df = find_enriched_clones_at_time(input_df, enrichment_month, thresholds[cell_type], cell_type, lineage_bias=lineage_bias, threshold_column=cell_type+'_percent_engraftment')
        else:
            enriched_cell_df = find_enriched_clones_at_time(input_df, enrichment_month, thresholds[cell_type], cell_type, lineage_bias=lineage_bias)
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

def clones_enriched_at_last_timepoint(
        input_df: pd.DataFrame,
        lineage_bias_df: pd.DataFrame,
        thresholds: Dict[str, float] = {'any' : 0.0},
        cell_type: str = 'any',
        lineage_bias: bool = False
    ) -> pd.DataFrame:
    """ Finds clones enriched at last timepoint for clone
    
    Arguments:
        input_df {pd.DataFrame} -- long format step 7 output
        lineage_bias_df {pd.DataFrame} -- lineage bias data output,
        set to empty dataframe if not analyzing lineage bias data
    
    Keyword Arguments:
        threshold {float} -- if analyzing absolute threshold values set (default: {0})
        cell_type {str} --  which cell type to apply threshold agains (default: {'any'})
        lineage_bias {bool} --  set true if analyzing lineage bias data(default: {False})
    
    Returns:
        pd.DataFrame -- [description]
    """

    groupby_cols = ['mouse_id', 'code']
    if lineage_bias:
        if cell_type == 'any':
            filtered_df = lineage_bias_df.loc[(lineage_bias_df['gr_percent_engraftment'] >= thresholds[cell_type]) | (lineage_bias_df['b_percent_engraftment'] >= thresholds[cell_type])]
        else:
            filtered_df = lineage_bias_df.loc[(lineage_bias_df[cell_type + '_percent_engraftment'] >= thresholds[cell_type])]
    else:
        filtered_df = filter_threshold(input_df, thresholds[cell_type], [cell_type])
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

def count_clones_at_percentile(input_df: pd.DataFrame, percentile: float, analyzed_cell_types: List[str] = ['gr','b'], thresholds: Dict[str,float] = None) -> pd.DataFrame:
    """ Wrapper function to count clones when applying a percentile based threshold

    Arguments:
        input_df {pd.DataFrame} -- step 7 long form data
        percentile {float} -- percentile to threshold for

    Keyword Arguments:
        analyzed_cell_types {List[str]} -- cell_types to analyze (default: {['gr','b']})

    Returns:
        pd.DataFrame -- clone_counts dataframe
    """

    if not thresholds:
        thresholds = find_top_percentile_threshold(input_df, percentile, cell_types=analyzed_cell_types)
    filtered_df = filter_cell_type_threshold(input_df, thresholds=thresholds, threshold_column='percent_engraftment', analyzed_cell_types=analyzed_cell_types)
    return count_clones(filtered_df)

def get_max_by_mouse_timepoint(input_df: pd.DataFrame, timepoint_column: str = 'month') -> pd.DataFrame:
    """ Find the maximum % engraftment by mouse/cell_type/month
    
    Arguments:
        input_df {pd.DataFrame} -- Dataframe pre-filtered for desired group to look for
    
    Keyword Arguments:
        timepoint_column {str} -- time column to group by (default: {'month'})
    
    Returns:
        pd.DataFrame -- data frame of max percent_engraftment
    """

    max_group = input_df.groupby(by=['cell_type', timepoint_column, 'mouse_id', 'group']).percent_engraftment.max()
    max_df = pd.DataFrame(max_group).reset_index()
    return max_df


def get_data_from_mice_missing_at_time(input_df: pd.DataFrame, exclusion_timepoint: int, timepoint_column: str = 'month') -> pd.DataFrame:
    """ Function used to exclude mice with data at a certain timepoint
    
    Arguments:
        input_df {pd.DataFrame} -- input dataframe (lineage_bias or perc_engraftment)
        exclusion_timepoint {int} -- timepoint during which to filter out mice
    
    Keyword Arguments:
        timepoint_column {str} -- column to search for timepoint (default: {'month'})
    
    Returns:
        pd.DataFrame -- dataframe without mice meant to be excluded
    """

    exclusion_mice = input_df.loc[input_df[timepoint_column] == exclusion_timepoint].mouse_id.unique()
    excluded_df = input_df.loc[~input_df.mouse_id.isin(exclusion_mice)]
    return excluded_df

def t_test_on_venn_data():
    """ Conducts independent t_test on data used to generate venn diagrams
    Data for each timepoint taken from print statements in the venn_barcode_in_time
    function in plotting_functions.py
    """

    no_change_b_4 = [91, 17, 155, 19, 26, 37, 88, 42, 3]
    aging_phenotype_b_4 = [105, 94, 107, 160, 11, 98, 10, 105, 28]
    no_change_gr_14 = [28, 15, 0, 1, 0]
    aging_phenotype_gr_14 = [13, 9, 31]
    no_change_b_14 = [60, 9, 13, 15, 7]
    aging_phenotype_b_14 = [5, 0, 33]

    vals = stats.ttest_ind(no_change_b_4, aging_phenotype_b_4)
    print('\nAging vs No Change 4 month only B clones ttest p-value: ')
    print(vals.pvalue)

    vals = stats.ttest_ind(no_change_gr_14, aging_phenotype_gr_14)
    print('\nAging vs No Change 14 month only Gr clones ttest p-value: ')
    print(vals.pvalue)

    vals = stats.ttest_ind(no_change_b_14, aging_phenotype_b_14)
    print('\nAging vs No Change 14 month only B clones ttest p-value: ')
    print(vals.pvalue)

def find_clones_bias_range_at_time(
        lineage_bias_df: pd.DataFrame,
        month: int,
        min_bias: float,
        max_bias: float,
    ) -> pd.DataFrame:
    """ Find clones with bias within a range at a time point
    
    Arguments:
        lineage_bias_df {pd.DataFrame} -- lineage bias data
        month {int} -- time point to select for
        min_bias {float} -- minimum bias in range
        max_bias {float} -- maximum bias in range
    
    Returns:
        pd.DataFrame -- input, filtered for clones with bias in range at desired timepoint
    """

    filt_index = (lineage_bias_df.month == month) & (lineage_bias_df.lineage_bias > min_bias) & (lineage_bias_df.lineage_bias < max_bias)
    filt_df = lineage_bias_df.loc[filt_index]
    filt_df = filt_df[['code', 'mouse_id']]
    return lineage_bias_df.merge(filt_df, on=['code', 'mouse_id'])

def percentile_sum_engraftment(input_df: pd.DataFrame, cell_type: str, num_points: int = 400) -> pd.DataFrame:
    """ Create dataframe sum of abundance due to clones below percentile ranked by clonal abundance
    
    Arguments:
        input_df {pd.DataFrame} -- abundance data frame
        cell_type {str} -- cell type to analyze
    
    Keyword Arguments:
        num_points {int} -- data points to create. Higher increases comp time and granularity (default: {400})
    
    Returns:
        pd.DataFrame -- Percentile vs Cumulative Abundance Dataframe
    """

    percentile_range = np.linspace(0, 100, num_points)
    cell_type_df = input_df.loc[input_df.cell_type == cell_type]
    contribution_df_cols = ['percentile', 'percent_sum_abundance', 'total_abundance', 'month', 'month_str', 'cell_type', 'quantile']
    contribution_df = pd.DataFrame(columns=contribution_df_cols)
    for percentile in percentile_range:
        for month in cell_type_df.month.unique():
            month_df = cell_type_df.loc[cell_type_df.month == month]
            contribution_row = pd.DataFrame(columns=contribution_df_cols)
            contribution_row.percentile = [percentile]
            contribution_row.cell_type = [cell_type]
            contribution_row.quantile = [month_df.quantile(percentile/100)]
            contribution_row.month_str = ['Month: ' + str(month)]
            contribution_row.month = [month]
            contribution_row.total_abundance = [month_df.percent_engraftment.sum()]
            sum_top = month_df.loc[month_df.percent_engraftment <= month_df.quantile(percentile/100).percent_engraftment].percent_engraftment.sum()
            contribution_row['percent_sum_abundance'] = 100*(sum_top)/month_df.percent_engraftment.sum()
            contribution_df = contribution_df.append(contribution_row, ignore_index=True)
    return contribution_df.reset_index()

def calculate_bias_change_cutoff(bias_change_df: pd.DataFrame) -> float:
    """ Calculates change amount that qualifies as "change"

    Arguments:
        bias_change_df {pd.DataFrame} -- change in lineage bias dataframe

    Returns:
        float -- cutoff value for change

    ValueError:
        if more than 1 cutoff found, throws error
    """
    fig = plt.figure()
    kde = sns.kdeplot(bias_change_df.bias_change.abs(), shade=True)
    x, y = kde.get_lines()[0].get_data()
    plt.close(fig=fig)
    # Calculate first derivative
    dy_dx = np.diff(y)/np.diff(x)
    dx_vals = x[1:]
    cutoff_candidates: List = []
    for i, _ in enumerate(dy_dx):
        if i != 0:
            # numerically check if 2nd derivative > 0 and first derivative ~0
            if dy_dx[i - 1] <= 0 and dy_dx[i] >= 0:
                cutoff_candidates.append(dx_vals[i])
    if len(cutoff_candidates) > 1:
        print(cutoff_candidates)
        raise ValueError('Too many candidates found')

    return cutoff_candidates[0]

def mark_changed(input_df: pd.DataFrame, bias_change_df: pd.DataFrame) -> pd.DataFrame:
    """ Adds column to input df based on if clone has 'changed' or not
    
    Arguments:
        input_df {pd.DataFrame} -- long form step7 output or lineage bias dataframe
        bias_change_df {pd.DataFrame} -- Dataframe of bias change
    
    Returns:
        pd.DataFrame -- input_df with changed column
    """
    cutoff = calculate_bias_change_cutoff(bias_change_df)
    with_bias_df = input_df.merge(bias_change_df[['code', 'group', 'mouse_id', 'bias_change']], how='left', on=['code', 'group', 'mouse_id'])
    with_bias_df['change_cutoff'] = cutoff
    print('Lineage Bias Change Cutoff: ' + str(round(cutoff,2)))

    with_change_df = with_bias_df.assign(
        changed=lambda row: row.bias_change.abs() >= cutoff,
    )

    return with_change_df

def sum_abundance_by_change(with_change_contribution_df: pd.DataFrame, percent_of_total: bool = False) -> pd.DataFrame:
    """ Cumulative abundance at percentiles
    
    Arguments:
        with_change_df {pd.DataFrame} -- contribution data frame where clones are marked as having changed or not changed (True/false)
    
    Keyword Arguments:
        percent_of_total {bool} -- To calculate based on %engraftment or as a percent of total engraftment for a cell_type (default: {False})
    
    Returns:
        pd.DataFrame -- sum of abundance in chagen vs not changed at each time point, per cell_type, mouse_id
    """

    total_sum = pd.DataFrame(with_change_contribution_df.groupby(['cell_type', 'group', 'mouse_id', 'month']).percent_engraftment.sum()).reset_index()
    by_change_sum = pd.DataFrame(with_change_contribution_df.groupby(['cell_type', 'group', 'mouse_id', 'month', 'changed']).percent_engraftment.sum()).reset_index()
    total_sum['changed'] = 'Total'
    total_sum['total_abundance'] = total_sum['percent_engraftment']
    if percent_of_total:
        total_merged_df = by_change_sum.merge(
            total_sum[['mouse_id', 'cell_type', 'group', 'month', 'total_abundance']],
            how='left',
            on=['mouse_id', 'cell_type', 'group', 'month']
        )
        all_change_data_sum = total_merged_df.assign(
            percent_engraftment=lambda x: 100*x.percent_engraftment/x.total_abundance
        )
    else:
        all_change_data_sum = total_sum.append(by_change_sum, ignore_index=True)

    all_change_data_sum = all_change_data_sum.assign(
        change_status=all_change_data_sum.changed.map({'Total': 'Total', False: 'Unchanged', True: 'Changed'})
    )
    return all_change_data_sum

def find_intersect(data, y, x_col: str = 'percentile', y_col: str = 'percent_sum_abundance'):
    """ Find where percentile matches sum abundance value on cumulative abundance vs percentile dataframe
    
    Arguments:
        data {[type]} -- Cumulative abundance data frame
        y {[type]} -- value to find percentile for
    
    Keyword Arguments:
        x_col {str} -- column to search for x values in (default: {'percentile'})
        y_col {str} -- column to search for y values in (default: {'percent_sum_abundance'})
    
    Returns:
        Tuple(float, float) -- x, y values for intersections
    """

    if y in data[y_col].values:
        print('Found intersection in data')
        y_val = y
        x_val = data[data[y_col] == y][x_col]
    else:
        idx = np.argwhere(np.diff(np.sign(data[y_col] - y))).flatten()
        if len(idx) > 1:
            print('More than one intersect found: ')
            print(idx)
        x_val = data.iloc[idx[0]][x_col]
        y_val = data.iloc[idx[0]][y_col]

    print(x_val, y_val)
    return (x_val, y_val)

def calculate_thresholds_sum_abundance(
    input_df: pd.DataFrame,
    abundance_cutoff: float = 50.0,
    analyzed_cell_types: List[str] = ['gr', 'b']
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Calculates abundance thresholds by cell type based on cumulative abundance at month 4
    
    Arguments:
        input_df {pd.DataFrame} --  clone abundance dataframe
    
    Keyword Arguments:
        abundance_cutoff {float} -- % of cells lower portion should contribute (default: {50.0})
        analyzed_cell_types {List[str]} -- cell types to analyze against (default: {['gr', 'b']})
    
    Returns:
        Tuple[Dict[str, float], Dict[str, float]] -- Dictionary of percentile and thresholds in format {cell_type: value}
    """


    thresholds: Dict[str, float] = {}
    percentiles: Dict[str, float] = {}

    for cell_type in analyzed_cell_types:
        month_4_cell_df = input_df.loc[(input_df.month == 4) & (input_df.cell_type == cell_type)]
        contributions = percentile_sum_engraftment(month_4_cell_df, cell_type)
        percentile, _ = find_intersect(
            data=contributions,
            y=abundance_cutoff,
            x_col='percentile',
            y_col='percent_sum_abundance'
        )
        percentile_threshold = find_top_percentile_threshold(
            month_4_cell_df,
            percentile/100,
            cell_types=[cell_type]
        )
        thresholds[cell_type] = percentile_threshold[cell_type]
        percentiles[cell_type] = percentile
    
    print('\nPercentiles: ')
    print(percentiles)
    print('\nThresholds: ')
    print(thresholds)
    return (percentiles, thresholds)
