""" Commonly used data transform functions for analysis of step7 output data

"""
from typing import List, Dict, Tuple, Any
from operator import itemgetter
from math import pi, sin
import os
import numpy as np
import scipy.stats as stats
import pandas as pd
from colorama import init, Fore, Style
from skbio.diversity import alpha_diversity
from intersection.intersection import intersection
from parse_facs_data import parse_wbc_count_file
import progressbar

init(autoreset=True)

UNIQUE_CODE_COLS = ['code', 'mouse_id']
  
MAP_LINEAGE_BIAS_CATEGORY = {
    'LC': 'Lymphoid Committed',
    'LB': 'Lymphoid Biased',
    'BL': 'Balanced - Lymphoid Leaning',
    'B': 'Balanced',
    'BM': 'Balanced - Myeloid Leaning',
    'MB': 'Myeloid Biased',
    'MC': 'Myeloid Committed',
}
MAP_BIAS_CELL_TYPE = {
    'gr': 'myeloid',
    'mo': 'myeloid',
    'b': 'lymphoid',
}
MAP_LINEAGE_BIAS_CATEGORY_SHORT = {
    'LC': 'Ly Committed',
    'LB': 'Ly Biased',
    'B': 'Balanced',
    'MB': 'My Biased',
    'MC': 'My Committed',
}
def sort_xy_lists(x: List, y: List):
    return [list(x) for x in zip(*sorted(zip(x, y), key=itemgetter(0)))]
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
        print(
            '\t Filtering for clones with', 
            cell_type,
            '>',
            thresholds[cell_type]
        )
        cell_df = filter_threshold(input_df,
                                   threshold=thresholds[cell_type],
                                   analyzed_cell_types=[cell_type],
                                   threshold_column=threshold_column
                                  )
        filtered_df = filtered_df.append(cell_df)
    return filtered_df

def find_first_clones_in_mouse(
        input_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    first_clones = pd.DataFrame(
        input_df.groupby(['mouse_id'])[timepoint_col].min()
        ).reset_index()
    first_clones_with_data = first_clones.merge(
        input_df,
        how='inner',
        on=['mouse_id', timepoint_col],
        validate='1:m'
    )
    return first_clones_with_data
def find_last_clones_in_mouse(
        input_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    last_clones = pd.DataFrame(
        input_df.groupby(['mouse_id'])[timepoint_col].max()
        ).reset_index()
    last_clones_with_data = last_clones.merge(
        input_df,
        how='inner',
        on=['mouse_id', timepoint_col],
        validate='1:m'
    )
    return last_clones_with_data

def find_last_clones(
        input_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    if 'code' in input_df.columns:
        last_clones = pd.DataFrame(
            input_df.groupby(['code', 'mouse_id'])[timepoint_col].max()
            ).reset_index()
        last_clones_with_data = last_clones.merge(
            input_df,
            how='inner',
            on=['code', 'mouse_id', timepoint_col],
        )
    else:
        last_clones = pd.DataFrame(
            input_df.groupby(['mouse_id'])[timepoint_col].max()
            ).reset_index()
        last_clones_with_data = last_clones.merge(
            input_df,
            how='inner',
            on=['mouse_id', timepoint_col],
            validate='1:m'
        )
    return last_clones_with_data

def find_n_to_last_clones(
        input_df: pd.DataFrame,
        timepoint_col: str,
        n: int,
    ):
    print('... Filtering for', n, 'to last clones')
    time_cols = input_df[['mouse_id', timepoint_col]].drop_duplicates()
    grouped_df = time_cols.sort_values(by=['mouse_id', timepoint_col], ascending=False).groupby(['mouse_id'])
    max_time_df = pd.DataFrame(grouped_df.nth(n=n-1)[timepoint_col]).reset_index()
    last_clones_with_data = input_df.merge(
        max_time_df,
        how='inner',
        validate='m:1'
    )
    count_group = last_clones_with_data.groupby(['mouse_id', timepoint_col])

    return last_clones_with_data

def filter_clones_threshold_anytime(
        input_df: pd.DataFrame,
        thresholds: Dict[str, float],
        analyzed_cell_types: List[str],
        filter_exempt_cell_types: List[str] = ['hsc'],
        filt_0_out_exempt: bool = False,
    ) -> pd.DataFrame:
    """ Filter for clones above a threshold at any time point
    
    Arguments:
        input_df {pd.DataFrame} -- abundance data frame
        thresholds {Dict[str,float]} -- {cell_type: threshold}
        analyzed_cell_types {List[str]} -- cell types to consider
    
    Returns:
        pd.DataFrame -- filtered for clones which pass the threshold at any timepoint
    """


    exempt_df = input_df[input_df.cell_type.isin(filter_exempt_cell_types)]
    if filt_0_out_exempt:
        print('\t Filtering for clones > 0 abundance in ', filter_exempt_cell_types)
        exempt_df = exempt_df[exempt_df.percent_engraftment > 0]
    not_exempt_df = input_df[~input_df.cell_type.isin(filter_exempt_cell_types)]
    hard_filtered_df = filter_cell_type_threshold(
        not_exempt_df,
        thresholds,
        analyzed_cell_types,
    )
    deduped = hard_filtered_df.drop_duplicates(subset=['code', 'mouse_id'])
    clones_above_thresh_at_any_time = not_exempt_df.merge(
        deduped[['code', 'mouse_id']],
        how='inner',
        on=['code', 'mouse_id'],
        validate='m:1'
    ).append(exempt_df)
    change_in_length = len(input_df) - len(clones_above_thresh_at_any_time)
    print('Filters:', thresholds)
    print('Change in length of abundanc post filtering:', change_in_length)
    return clones_above_thresh_at_any_time[clones_above_thresh_at_any_time.cell_type.isin(analyzed_cell_types)]


def filter_lineage_bias_threshold(
        lineage_bias_df: pd.DataFrame,
        threshold: float,
        cell_type: str,
    ) -> pd.DataFrame:
    filter_col = MAP_BIAS_CELL_TYPE[cell_type] + '_percent_abundance'
    return lineage_bias_df.loc[lineage_bias_df[filter_col] >= threshold]
    
def filter_lineage_bias_thresholds(
        lineage_bias_df: pd.DataFrame,
        thresholds: Dict[str, float],
    ) -> pd.DataFrame:
    filt_df = pd.DataFrame()
    for cell_type in thresholds.keys():
        filter_col = MAP_BIAS_CELL_TYPE[cell_type] + '_percent_abundance'
        filt_df = filt_df.append(lineage_bias_df.loc[lineage_bias_df[filter_col] > thresholds[cell_type]])
    return filt_df.drop_duplicates()

def filter_lineage_bias_anytime(
        lineage_bias_df: pd.DataFrame,
        thresholds: Dict[str, float],
    ) -> pd.DataFrame:

    print('Filtering Lineage Bias for clones passing threshold at anytime:', thresholds)
    filt_df = pd.DataFrame()
    for cell_type, threshold in thresholds.items():
        filt_df = filt_df.append(
            lineage_bias_df[(
                lineage_bias_df[MAP_BIAS_CELL_TYPE[cell_type] + '_percent_abundance'] >= threshold
            )]
            )
    filt_codes = filt_df[['code', 'mouse_id']].drop_duplicates(subset=['code','mouse_id'])
    anytime_thresh_df = lineage_bias_df.merge(
        filt_codes,
        how='inner',
        validate='m:1'
    )
    print('Length post filtering abundance anytime:', len(anytime_thresh_df))
    return anytime_thresh_df

def get_clones_at_timepoint(
        input_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: Any,
        by_mouse: bool,
        n: Any = None,
    ) -> pd.DataFrame:
    if timepoint_col == 'month' and timepoint == 'first':
        timepoint = 9
        print(Fore.RED + 'First month changed to month 9')
    if n is not None:
        if timepoint == 'last':
            if by_mouse:
                filt_df = find_n_to_last_clones(
                    input_df,
                    timepoint_col,
                    n
                )
            else:
                raise ValueError('N to ' + str(timepoint) + ' Not Implemented for by clone yet')
        else:
            raise ValueError('N to ' + str(timepoint) + ' Not Implemented')

    elif timepoint == 'last':
        if by_mouse:
            filt_df = find_last_clones_in_mouse(
                input_df,
                timepoint_col
            )
        else:
            filt_df = find_last_clones(
                input_df,
                timepoint_col
            )
    elif timepoint == 'first':
        if by_mouse:
            filt_df = find_first_clones_in_mouse(
                input_df,
                timepoint_col
            )
        else:
            filt_df = pd.DataFrame(
                input_df.sort_values(timepoint_col).groupby(['mouse_id', 'code']).first()
            ).reset_index()
    else:
        filt_df = input_df[input_df[timepoint_col] == int(timepoint)]
    return filt_df

def filter_biased_clones_at_timepoint(
        lineage_bias_df: pd.DataFrame,
        bias_cutoff: float,
        timepoint: Any,
        timepoint_col: str,
        within_cutoff: bool,
    ) -> pd.DataFrame:
    """ Filter for clones with lineage bias at specified extreme, at timepoint
    
    Arguments:
        lineage_bias_df {pd.DataFrame} -- Lineage Bias Data Frame
        bias_cutoff {float} -- Value of Lineage bias to be more extreme than
        timepoint {int} -- timepoint to inspect
        timepoint_col {str} -- column to look for timepoint in
        within_cutoff {bool} -- Check for clones NOT extremely biased instead

    Raises:
        ValueError -- If bias_cutoff set to 0
    
    Returns:
        pd.DataFrame -- filtered lineage_bias_df with clones extreme at timepoint
    """
    
    if within_cutoff:
        filt_df = lineage_bias_df[lineage_bias_df.lineage_bias.abs() < np.abs(bias_cutoff)]
    else:
        # If cutoff is positive, check extreme as >, otherwise as <
        if bias_cutoff > 0:
            filt_df = lineage_bias_df[lineage_bias_df.lineage_bias > bias_cutoff]
        elif bias_cutoff < 0:
            filt_df = lineage_bias_df[lineage_bias_df.lineage_bias < bias_cutoff]
        else:
            raise ValueError('bias_cutoff cannot be 0')

    if timepoint == 'last':
        temp_df = pd.DataFrame()
        for _, m_df in filt_df.groupby('mouse_id'):
            tp = m_df[timepoint_col].max()
            temp_df = temp_df.append(
                m_df[m_df[timepoint_col] == tp]
            )
        filt_df = temp_df
    elif timepoint == 'first':
        tp = filt_df[timepoint_col].min()
        filt_df = filt_df[filt_df[timepoint_col] == tp]
    else:
        filt_df = filt_df[filt_df[timepoint_col] == int(timepoint)]
    passing_clones = filt_df[['mouse_id', 'code']]
    biased_at_timepoint_df = lineage_bias_df.merge(
        passing_clones,
        how='inner',
        on=['mouse_id', 'code'],
        validate='m:1',
    )
    return biased_at_timepoint_df

def count_clones(input_df: pd.DataFrame, timepoint_col: str) -> pd.DataFrame:
    """ Count unique clones per cell type

    Arguments:
        input_df {pd.DataFrame} -- long formatted step7 output
        timepoint_col {str} -- column to look for time values in

    Returns:
        pd.DataFrame -- DataFrame with columns 'mouse_id','day', 'cell_type', 'code' where
        'code' contains count of unique barcodes
    """

    clone_counts = pd.DataFrame(
        input_df.groupby(['mouse_id', 'group', timepoint_col, 'cell_type'])['code'].nunique()
        ).reset_index()
    total_clone_counts = pd.DataFrame(input_df.groupby(['mouse_id', 'group', timepoint_col])['code'].nunique()).reset_index()
    total_clone_counts['cell_type'] = 'Total'
    clone_counts = clone_counts.append(total_clone_counts, sort=True)

    return clone_counts


def find_enriched_clones_at_time(input_df: pd.DataFrame,
                                 enrichment_time: Any,
                                 enrichment_threshold: float,
                                 cell_type: str,
                                 timepoint_col: str,
                                 threshold_column: str = 'percent_engraftment',
                                 lineage_bias: bool = False,
                                 by_mouse: bool = True,
                                 ) -> pd.DataFrame:
    """ Finds clones enriched at a specific time point for a cell type

    Arguments:
        input_df {pd.DataFrame} -- long format data, formatted with filter_threshold()
        enrichment_time {int} -- timepoint of interest
        threshold {int} -- threshold for significant engraftment
        cell_type {str} -- Cell type to select for
        threshold_column {str} -- column on which to apply threshold

    Keyword Arguments:
        lineage_bias {bool} -- Checks if running lineage bias data (default: False)
        by_mouse {bool} -- If timepoint first or last, finds it by mouse not by clone

    Returns:
        pd.DataFrame -- DataFrame with only clones enriched at specified timepoint
    """

    time_df = get_clones_at_timepoint(
        input_df,
        timepoint_col,
        enrichment_time,
        by_mouse,
    )
    enriched_at_time_df = time_df[time_df[threshold_column] > enrichment_threshold].drop_duplicates(['code', 'mouse_id'])

    if lineage_bias:
        cell_df = input_df
    else:
        cell_df = input_df[input_df.cell_type == cell_type]
        enriched_at_time_df = enriched_at_time_df[enriched_at_time_df.cell_type == cell_type]

    return cell_df.merge(enriched_at_time_df[['code', 'mouse_id']], how='inner', on=['code', 'mouse_id'], validate="m:1")

def combine_enriched_clones_at_time(
        input_df: pd.DataFrame,
        enrichment_time: Any,
        timepoint_col: str,
        thresholds: Dict[str, float],
        analyzed_cell_types: List[str],
        lineage_bias: bool = False,
        by_mouse: bool = True,
    ) -> pd.DataFrame:
    """ wrapper of find_enriched_clones_at_time() to combine entries from multiple cell types

    Arguments:
        input_df {pd.DataFrame} -- data frame with month value
        enrichement_time {int} -- month to check enrichment at
        threshold {float} -- threshold for enrichment
        analyzed_cell_types {List[str]} -- cell types to analyze
    
    Returns:
        pd.DataFrame -- data frame of enriched clones in specified cell types and timepoint
    """

    all_enriched_df = pd.DataFrame()
    for cell_type in analyzed_cell_types:
        if lineage_bias:
            if cell_type == 'gr':
                lin_type = 'myeloid'
            if cell_type == 'b':
                lin_type = 'lymphoid'
            enriched_cell_df = find_enriched_clones_at_time(input_df, enrichment_time, thresholds[cell_type], cell_type, by_mouse=by_mouse, timepoint_col=timepoint_col, lineage_bias=lineage_bias, threshold_column=lin_type+'_percent_abundance')
        else:
            enriched_cell_df = find_enriched_clones_at_time(
                input_df,
                enrichment_time,
                thresholds[cell_type],
                cell_type,
                by_mouse=by_mouse,
                timepoint_col=timepoint_col,
                lineage_bias=lineage_bias
            )
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
        timepoint_col: str,
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
    grouped_df = pd.DataFrame(filtered_df.groupby(by=groupby_cols)[timepoint_col].max()).reset_index()
    if lineage_bias:
        filtered_for_enrichment = lineage_bias_df.merge(grouped_df['code'], how='inner', on=['code'])
    else:
        filtered_for_enrichment = input_df.merge(grouped_df['code'], how='inner', on=['code'])

    return filtered_for_enrichment

def filter_mice_with_n_timepoints(input_df: pd.DataFrame, n_timepoints: int = 4, time_col: str = 'month') -> pd.DataFrame:
    """ Finds mice with desired number of timepoints.
    Used primarily to only select mice with all four time points

    Arguments:
        input_df {pd.DataFrame} -- Step 7 long format data

    Keyword Arguments:
        n_timepoints {int} -- number of timepoints desired (default: {4})
        time_col {string} -- column to look for timepoints in

    Returns:
        pd.DataFrame -- [description]
    """

    output_df = pd.DataFrame()
    pass_mice = []
    # Allow mice if serial transplant and 8 time points
    if (time_col == 'gen') and (n_timepoints == 8):
        pass_mice = [ 'M1', 'M10', 'M5', 'M12', 'M3', 'M16', 'M7', 'M6']
    for mouse_id, group in input_df.groupby(['mouse_id']):
        if group[time_col].nunique() >= n_timepoints:
            output_df = output_df.append(group)
        elif mouse_id in pass_mice:
                print(n_timepoints, time_col + 's', 'not found, but adding mouse data:', mouse_id)
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
                                       timepoint_col = 'month',
                                       thresholds: List[float] = [0.0, 0.01, 0.02, 0.2, 0.5],
                                       outdir: str = '/home/sakre/Data/clone_counts_long',
                                       analyzed_cell_types: List[str] = ['gr', 'b'],
                                      ):
    """ Outputs clone count csv file at multiple thresholds.
    """
    input_df = pd.read_csv(input_file)
    for threshold in thresholds:
        filter_df = filter_threshold(input_df, threshold, analyzed_cell_types)
        clone_counts = count_clones(filter_df, timepoint_col)
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

def percentile_sum_engraftment(input_df: pd.DataFrame, timepoint_col: str, cell_type: str, num_points: int = 400) -> pd.DataFrame:
    """ Create dataframe sum of abundance due to clones below percentile ranked by clonal abundance
    
    Arguments:
        input_df {pd.DataFrame} -- abundance data frame
        cell_type {str} -- cell type to analyze
    
    Keyword Arguments:
        num_points {int} -- data points to create. Higher increases comp time and granularity (default: {400})
    
    Returns:
        pd.DataFrame -- Percentile vs Cumulative Abundance Dataframe
    """

    time_col = timepoint_col
    percentile_range = np.linspace(0, 100, num_points)
    cell_type_df = input_df.loc[input_df.cell_type == cell_type]
    contribution_df_cols = ['percentile', 'percent_sum_abundance', 'total_abundance', time_col, 'cell_type']
    contribution_df = pd.DataFrame(columns=contribution_df_cols)
    for percentile in percentile_range:
        for time_point in cell_type_df[time_col].unique():
            time_df = cell_type_df.loc[cell_type_df[time_col] == time_point]
            contribution_row = pd.DataFrame(columns=contribution_df_cols)
            contribution_row.percentile = [percentile]
            contribution_row.cell_type = [cell_type]
            contribution_row.quantile = [time_df.quantile(percentile/100)]
            contribution_row[time_col] = [time_point]
            contribution_row.total_abundance = [time_df.percent_engraftment.sum()]
            sum_top = time_df.loc[
                time_df.percent_engraftment <= time_df.quantile(percentile/100).percent_engraftment
            ].percent_engraftment.sum()
            contribution_row['percent_sum_abundance'] = 100*(sum_top)/time_df.percent_engraftment.sum()
            contribution_df = contribution_df.append(contribution_row, ignore_index=True)
    return contribution_df.reset_index()

def calculate_bias_change_cutoff(
        bias_change_df: pd.DataFrame,
        min_time_difference: int,
        timepoint=None,
        **kde_kwargs
    ) -> Tuple:
    """ Calculates change amount that qualifies as "change"

    Arguments:
        bias_change_df {pd.DataFrame} -- change in lineage bias dataframe
        min_time_difference {int} -- minimum number of days to count use

    Returns:
        float -- cutoff value for change

    ValueError:
        if more than 1 cutoff found, throws error
    """
    
    bias_change_df = bias_change_df[bias_change_df.time_change >= min_time_difference] 
    if timepoint is not None:
        bias_change_df = filter_bias_change_timepoint(
            bias_change_df,
            timepoint
        )
        print(bias_change_df)
    
    # C0 KDE of all clones
    kde = stats.gaussian_kde(
        bias_change_df.bias_change.abs(),
        bw_method='scott'
    )
    x = np.linspace(-.5, 2.5, 100)
    y = kde.pdf(x)
    y_peak = y.argmax() + 1

    # C1 KDE of "unchanged" clones
    y1 = np.zeros(x.shape)
    y_vals = np.append(y[:y_peak], y[:y_peak][::-1])
    y1[:y_vals.shape[0]] = y_vals

    # C2 KDE of "changed" clones
    y2 = y - y1

    x_c, y_c = intersection(x, y1, x, y2)

    if len(x_c) > 1:
        print(Fore.YELLOW + Style.BRIGHT + 'Warning: Too many change cutoff candidates found')
        print(Fore.YELLOW + ','.join([str(k) for k in x_c]))
    print('Bias Change Cutoff:', x_c[0])
    return x, y, y1, y2, x_c, y_c, kde

def mark_changed(
        input_df: pd.DataFrame,
        bias_change_df: pd.DataFrame,
        min_time_difference: int,
        merge_type: str = 'inner',
        timepoint: Any = None,
    ) -> pd.DataFrame:
    """ Adds column to input df based on if clone has 'changed' or not
    
    Arguments:
        input_df {pd.DataFrame} -- long form step7 output or lineage bias dataframe
        bias_change_df {pd.DataFrame} -- Dataframe of bias change
    
    Returns:
        pd.DataFrame -- input_df with changed column
    """
    _, _, _, _, cutoffs, _, _ = calculate_bias_change_cutoff(
        bias_change_df,
        min_time_difference,
        timepoint=timepoint
    )
    cutoff = cutoffs[0]
    if not 'bias_change' in input_df.columns:
        with_bias_df = input_df.merge(
            bias_change_df[['code', 'group', 'mouse_id', 'bias_change']],
            how=merge_type,
            on=['code', 'group', 'mouse_id'],
            validate='m:m',
        )
    else:
        with_bias_df = input_df
    with_bias_df['change_cutoff'] = cutoff

    with_change_df = with_bias_df.assign(
        changed=lambda row: row.bias_change.abs() >= cutoff,
    )
    with_change_df = with_change_df.assign(
        change_status=with_change_df.changed.map({
            False: 'Unchanged',
            True: 'Changed',
        })
    )
    with_change_df['change_type'] = np.sign(with_change_df.bias_change)
    with_change_df = with_change_df.assign(
        change_type=with_change_df.change_type.map({
            -1: 'Lymphoid',
            1: 'Myeloid'
        })
    )

    with_change_df.loc[~with_change_df.changed, 'change_type'] = 'Unchanged'
    with_change_df.loc[with_change_df.bias_change.isna(), 'change_type'] = 'Unknown'
    with_change_df.loc[with_change_df.bias_change.isna(), 'change_status'] = 'Unknown'

    return with_change_df

def sum_abundance_by_change(
        with_change_contribution_df: pd.DataFrame,
        percent_of_total: bool = False,
        timepoint_col: str = 'month',
        change_col: str = 'change_type'
    ) -> pd.DataFrame:
    """ Cumulative abundance at percentiles
    
    Arguments:
        with_change_df {pd.DataFrame} -- contribution data frame where clones are marked as having changed or not changed (True/false)
    
    Keyword Arguments:
        percent_of_total {bool} -- To calculate based on %engraftment or as a percent of total engraftment for a cell_type (default: {False})
    
    Returns:
        pd.DataFrame -- sum of abundance in chagen vs not changed at each time point, per cell_type, mouse_id
    """

    total_sum = pd.DataFrame(
        with_change_contribution_df.groupby(
            ['cell_type', 'group', 'mouse_id', timepoint_col]
        ).percent_engraftment.sum()).reset_index()
    by_change_sum = pd.DataFrame(
        with_change_contribution_df.groupby(
            ['cell_type', 'group', 'mouse_id', timepoint_col, change_col]
            ).percent_engraftment.sum()
        ).reset_index()
    total_sum['changed'] = 'Total'
    total_sum['total_abundance'] = total_sum['percent_engraftment']
    if percent_of_total:
        total_merged_df = by_change_sum.merge(
            total_sum[['mouse_id', 'cell_type', 'group', timepoint_col, 'total_abundance']],
            how='left',
            on=['mouse_id', 'cell_type', 'group', timepoint_col]
        )
        all_change_data_sum = total_merged_df.assign(
            percent_engraftment=lambda x: 100*x.percent_engraftment/x.total_abundance
        )
    else:
        all_change_data_sum = total_sum.append(by_change_sum, ignore_index=True)


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
        x_val = data.iloc[idx[-1]][x_col]
        y_val = data.iloc[idx[-1]][y_col]

    return (x_val, y_val)

def calculate_thresholds_sum_abundance(
        input_df: pd.DataFrame,
        timepoint_col: str,
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

    print(
        '\n -- Calculating Threshold for Based on Cumulative Abundance ' \
        + str(abundance_cutoff) + ' -- \n'
    )
    for cell_type in analyzed_cell_types:
        month_4_cell_df = get_clones_at_timepoint(
            input_df,
            timepoint_col,
            'first',
            by_mouse=True
        )
        month_4_cell_df = month_4_cell_df[month_4_cell_df.cell_type == cell_type]
        # USE LAST IF HSC DATA
        if cell_type == 'hsc':
            month_4_cell_df = find_last_clones_in_mouse(
                input_df,
                timepoint_col
            )
        month_4_cell_df[timepoint_col] = 'first'

        contributions = percentile_sum_engraftment(month_4_cell_df, timepoint_col, cell_type=cell_type)
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
        print(
            cell_type.title() \
            + ' Percentile: ' + str(percentile) \
            + ' Threshold: ' + str(thresholds[cell_type])
        )
    
    return (percentiles, thresholds)

def between_gen_bias_change(
        lineage_bias_df: pd.DataFrame,
        absolute: bool = False
    ) -> pd.DataFrame:

    bias_gen_change_cols = ['mouse_id', 'code', 'group', 'gen_change', 'bias_change']
    bias_gen_change_df = pd.DataFrame(columns=bias_gen_change_cols)
    for _, group in lineage_bias_df.groupby(['code', 'mouse_id']):
        if group.day.nunique() != len(group):
            print('\n *** mismatch days to group length *** \n')
            print(group)
        for _, row in group.iterrows():
            next_gen_row = group[group.day == row.day + 2]
            if next_gen_row.empty:
                continue
            if not row.mouse_id:
                print(row)
            bias_gen_row = pd.DataFrame(columns=bias_gen_change_cols)
            gen = (row.day - 127)/2 + 1
            bias_gen_row.code = [row.code]
            bias_gen_row.mouse_id = [row.mouse_id]
            bias_gen_row.group = [row.group]
            bias_gen_row.gen_change = [str(int(gen)) + ' to ' + str(int(gen + 1))]
            bias_change = next_gen_row.lineage_bias - row.lineage_bias
            if absolute:
                bias_change = bias_change.abs()
            bias_gen_row.bias_change = bias_change.tolist()
            bias_gen_change_df = bias_gen_change_df.append(bias_gen_row, ignore_index=True)

    return bias_gen_change_df

def across_gen_bias_change(
        lineage_bias_df: pd.DataFrame,
        absolute: bool = False
    ) -> pd.DataFrame:

    bias_gen_change_cols = ['mouse_id', 'code', 'group', 'gen_change', 'bias_change', 'first_gen', 'end_gen']
    bias_gen_change_df = pd.DataFrame(columns=bias_gen_change_cols)
    for _, group in lineage_bias_df.groupby(['code', 'mouse_id']):
        sorted_group = group.sort_values(by=['day'])
        first_gen_row = sorted_group.iloc[0]
        first_gen = (first_gen_row.day - 127)/2 + 1

        if int(first_gen) != 1:
            continue

        if sorted_group.day.nunique() != len(group):
            print('\n *** mismatch days to group length *** \n')
            print(group)
            raise ValueError('Non-unique value found for clone at day')

        for _, row in sorted_group.iloc[1:].iterrows():
            bias_gen_row = pd.DataFrame(columns=bias_gen_change_cols)
            end_gen = (row.day - 127)/2 + 1
            bias_gen_row.code = [row.code]
            bias_gen_row.mouse_id = [row.mouse_id]
            bias_gen_row.group = [row.group]
            bias_gen_row.gen_change = [str(int(first_gen)) + ' to ' + str(int(end_gen))]
            bias_gen_row.first_gen = [int(first_gen)]
            bias_gen_row.end_gen = [int(end_gen)]
            bias_change = row.lineage_bias - first_gen_row.lineage_bias
            if absolute:
                bias_change = np.abs(bias_change)
            bias_gen_row.bias_change = bias_change.tolist()
            bias_gen_change_df = bias_gen_change_df.append(bias_gen_row, ignore_index=True)

    return bias_gen_change_df

def day_to_gen(day: int):
    gen = (day-127)/2 + 1
    return gen

def day_to_month(day: pd.Series):
    month = round((day)/30).astype(int)
    month[month == 14] = 15
    month[month == 18] = 17
    return month

def calculate_abundance_change(
        abundance_df: pd.DataFrame,
        timepoint_col: str = 'month',
        cumulative: bool = False,
        first_timepoint: int = 1,
    ) -> pd.DataFrame:
    """ Calculate change in abundance across time
    
    Arguments:
        abundance_df {pd.DataFrame} -- abundance dataframe
    
    Keyword Arguments:
        timepoint_col {str} -- time change to consider (default: {'month'})
        cumulative {bool} -- True if looking across time (1-2, 1-3, 1-4, etc..),
        false if between (1-2, 2-3, 3-4, etc..) (default: {False})
        first_timepoint {int} -- which timepoint to set as first for cumulative
    
    Raises:
        ValueError -- [description]
    
    Returns:
        pd.DataFrame -- [description]
    """


    abundance_change_cols = ['mouse_id', 'code', 'cell_type', 'group', 'abundance_change', 'time_change', 'time_unit', 't1', 't2', 'label_change']
    abundance_change_df = pd.DataFrame(columns=abundance_change_cols)
    unique_timepoints = np.sort(abundance_df[timepoint_col].unique()).tolist()
    for _, group in abundance_df.groupby(['code', 'mouse_id', 'cell_type']):
        if len(group) < 2:
            continue
        sorted_group = group.sort_values(by=[timepoint_col])
        t1_row = sorted_group.iloc[0]
        if cumulative and t1_row[timepoint_col] != first_timepoint:
            continue

        if sorted_group[timepoint_col].nunique() != len(group):
            print('\n *** mismatch ' + timepoint_col + 's to group length *** \n')
            print(group)
            raise ValueError('Non-unique value found for clone at ' + timepoint_col)

        for i in range(len(sorted_group) - 1):
            t2_row = sorted_group.iloc[i+1]
            if not cumulative:
                t1_row = sorted_group.iloc[i]
                if unique_timepoints[unique_timepoints.index(t1_row[timepoint_col]) + 1] != t2_row[timepoint_col]:
                    continue

            abundance_change_row = pd.DataFrame(columns=abundance_change_cols)
            abundance_change_row.code = [t2_row.code]
            abundance_change_row.mouse_id = [t2_row.mouse_id]
            abundance_change_row.cell_type = [t2_row.cell_type]
            abundance_change_row.group = [t2_row.group]
            abundance_change_row.time_change = [t2_row[timepoint_col] - t1_row[timepoint_col]]
            abundance_change_row.label_change = [(str(int(np.ceil(t1_row[timepoint_col]))) + ' to ' + str(int(np.ceil(t2_row[timepoint_col])))).title()]
            abundance_change_row.t1 = [t1_row[timepoint_col]]
            abundance_change_row.t2 = [t2_row[timepoint_col]]
            abundance_change = t2_row.percent_engraftment - t1_row.percent_engraftment

            abundance_change_row.abundance_change = abundance_change.tolist()
            abundance_change_df = abundance_change_df.append(abundance_change_row, ignore_index=True)

    abundance_change_df.time_unit = timepoint_col
    return abundance_change_df

def calculate_bias_change(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str = 'month',
        cumulative: bool = False,
        first_timepoint: int = 1,
        use_month_17: bool = False,
    ) -> pd.DataFrame:
    """ Calculate change in bias across time
    
    Arguments:
        lineage_bias_df {pd.DataFrame} -- lineage_bias dataframe
    
    Keyword Arguments:
        timepoint_col {str} -- time change to consider (default: {'month'})
        cumulative {bool} -- True if looking across time (1-2, 1-3, 1-4, etc..),
        false if between (1-2, 2-3, 3-4, etc..) (default: {False})
        first_timepoint {int} -- which timepoint to set as first for cumulative
    
    Raises:
        ValueError -- If set grouped by code,mouse id  has length greater than number of timepoints
                      raise error
    
    Returns:
        pd.DataFrame -- Bias_change_df
    """

    if  (not use_month_17):
        if 17 in lineage_bias_df[timepoint_col].unique():
            print(Fore.YELLOW + 'EXCLUDING MONTH 17 FROM BIAS CHANGE')
            lineage_bias_df = lineage_bias_df[lineage_bias_df.month != 17]

    bias_change_cols = ['mouse_id', 'code', 'group', 'bias_change', 'time_change', 'time_unit', 't1', 't2', 'label_change']
    bias_change_df = pd.DataFrame(columns=bias_change_cols)
    unique_timepoints = np.sort(lineage_bias_df[timepoint_col].unique()).tolist()
    for _, group in lineage_bias_df.groupby(['code', 'mouse_id']):
        if len(group) < 2:
            continue
        sorted_group = group.sort_values(by=[timepoint_col])
        t1_row = sorted_group.iloc[0]
        if cumulative and t1_row[timepoint_col] != first_timepoint:
            continue

        if sorted_group[timepoint_col].nunique() != len(group):
            print('\n *** mismatch ' + timepoint_col + 's to group length *** \n')
            print(group)
            raise ValueError('Non-unique value found for clone at ' + timepoint_col)

        for i in range(len(sorted_group) - 1):
            t2_row = sorted_group.iloc[i+1]
            if not cumulative:
                t1_row = sorted_group.iloc[i]
                if unique_timepoints[unique_timepoints.index(t1_row[timepoint_col]) + 1] != t2_row[timepoint_col]:
                    continue

            bias_change_row = pd.DataFrame(columns=bias_change_cols)
            bias_change_row.code = [t2_row.code]
            bias_change_row.mouse_id = [t2_row.mouse_id]
            bias_change_row.group = [t2_row.group]
            bias_change_row.time_change = [t2_row[timepoint_col] - t1_row[timepoint_col]]
            bias_change_row.label_change = [(str(int(np.ceil(t1_row[timepoint_col]))) + ' to ' + str(int(np.ceil(t2_row[timepoint_col])))).title()]
            bias_change_row.t1 = [t1_row[timepoint_col]]
            bias_change_row.t2 = [t2_row[timepoint_col]]
            bias_change = t2_row.lineage_bias - t1_row.lineage_bias

            bias_change_row.bias_change = bias_change.tolist()
            bias_change_df = bias_change_df.append(bias_change_row, ignore_index=True)

    bias_change_df.time_unit = timepoint_col
    return bias_change_df

def filter_stable_initially(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        t1: int,
        bias_change_cutoff: float = 0.5,
    ) -> pd.DataFrame:
    stable_df = pd.DataFrame(columns=lineage_bias_df.columns)
    for _, code_df in lineage_bias_df.groupby(['mouse_id', 'code']):
        if len(code_df) == 1:
            continue
        code_df = code_df.sort_values(by=timepoint_col)
        t1_df = code_df.iloc[0]
        if t1_df[timepoint_col] != t1:
            continue

        t2_df = code_df.iloc[1]
        if np.abs(t2_df.lineage_bias - t1_df.lineage_bias) <= bias_change_cutoff:
            stable_df = stable_df.append(code_df)

    return stable_df
    
def bias_clones_to_abundance(
        lineage_bias_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        y_col: str,
    ) -> pd.DataFrame:
    """ Find Abundance for clones filtered based on lineage bias
    Formats output in expected way for lineage_bias_df's, i.e
    the abundance column is gr_percent_engraftment instead of just
    percent_engraftment.

    Arguments:
        lineage_bias_df {pd.DataFrame} -- Lineage Bias DF Filtered for desired clones
        clonal_abundance_df {pd.DataFrame} -- Clonal Abundance DF
        y_col {str} -- Column being analyzed

    Returns:
        pd.DataFrame -- DF in form of Lineage bias, containing only abundance informatio
    """
    filtered_clones = lineage_bias_df.code.unique()
    cell_type = None
    if y_col == 'myeloid_percent_abundance':
        cell_type = 'gr'
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.cell_type == cell_type]
    elif y_col == 'lymphoid_percent_abundance':
        cell_type = 'b'
        clonal_abundance_df = clonal_abundance_df[clonal_abundance_df.cell_type == cell_type]
    if cell_type:
        bias_clones_abundance_df = clonal_abundance_df[clonal_abundance_df.code.isin(filtered_clones)]
        bias_clones_abundance_df = bias_clones_abundance_df.rename(columns={'percent_engraftment': y_col})
    else:
        raise ValueError('No Cell Type Detected To Find Abundance In')
    return bias_clones_abundance_df

def calculate_first_last_bias_change(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        by_mouse: bool,
        exclude_month_17: bool = True,
    ):
    print(Fore.RED + ' FORCING BIAS CHANGE BY MOUSE')
    by_mouse = True

    group_cols = ['mouse_id', 'code', 'group']
    if exclude_month_17 and timepoint_col == 'month':
        if 17 in lineage_bias_df.month.unique():
            print(Fore.YELLOW + 'EXCLUDING MONTH 17 FROM BIAS CHANGE')
            lineage_bias_df = lineage_bias_df[lineage_bias_df.month != 17]
    if timepoint_col == 'gen':
        print(Fore.YELLOW + 'ONLY KEEPING GEN 1 and 8')
        lineage_bias_df = lineage_bias_df[
            lineage_bias_df[timepoint_col].isin([1,8])
        ]

    # TESTING MONTH 9 instead of FIRST
   # lineage_bias_at_first_df = get_clones_at_timepoint(
   #     lineage_bias_df,
   #     timepoint_col,
   #     'first',
   #     by_mouse=by_mouse,
   # )
    lineage_bias_at_first_df = get_clones_at_timepoint(
        lineage_bias_df,
        timepoint_col,
        'first',
        by_mouse=by_mouse,
    )
    lineage_bias_at_last_df = get_clones_at_timepoint(
        lineage_bias_df,
        timepoint_col,
        'last',
        by_mouse=by_mouse,
    )
    both_time_bias_df = lineage_bias_at_first_df.merge(
        lineage_bias_at_last_df,
        on=group_cols,
        suffixes=['_first', '_last'],
        how='inner',
        validate='1:1'
    )
    bias_change_df = both_time_bias_df.assign(
        bias_change=lambda x: x.lineage_bias_last - x.lineage_bias_first,
        time_change=lambda x: x[timepoint_col+'_last'] - x[timepoint_col+'_first'],
    )
    bias_change_df = bias_change_df[bias_change_df['time_change'] != 0]
    print(bias_change_df.groupby('mouse_id').code.nunique())
    return bias_change_df
def calculate_first_last_bias_change_with_avg_data(
    lineage_bias_df: pd.DataFrame,
    y_col: str,
    timepoint_col: str,
    ) -> pd.DataFrame:

    df_cols = ['mouse_id', 'code', 'group', 'average_'+y_col, 'bias_change', 'gr_change', 'b_change']
    add_change_status = False
    
    if 'change_status' in lineage_bias_df.columns:
        add_change_status = True
        df_cols.append('change_status')

    bias_dist_df = pd.DataFrame(columns=df_cols)
    for name, group in lineage_bias_df.groupby(['code', 'mouse_id', 'group']):
        if len(group) < 2:
            continue
        bias_change_row = pd.DataFrame(columns=df_cols)
        sorted_group = group.sort_values(by=timepoint_col)
        t1 = sorted_group.iloc[0]
        t2 = sorted_group.iloc[-1]
        if y_col == 'sum_abundance':
            avg_val = (sorted_group['gr_percent_engraftment'] + sorted_group['b_percent_engraftment']).mean()
        else:
            avg_val = sorted_group[y_col].mean()
        bias_change = t2.lineage_bias - t1.lineage_bias
        bias_change_row['code'] = [name[0]]
        bias_change_row['mouse_id'] = [name[1]]
        bias_change_row['group'] = [name[2]]
        bias_change_row['average_'+y_col] = [avg_val]
        bias_change_row['gr_change'] = [t2['gr_percent_engraftment'] - t1['gr_percent_engraftment']]
        bias_change_row['b_change'] = [t2['b_percent_engraftment'] - t1['b_percent_engraftment']]
        bias_change_row['bias_change'] = [bias_change]
        if add_change_status:
            bias_change_row['change_status'] = t1['change_status']

        bias_dist_df = bias_dist_df.append(bias_change_row, ignore_index=True)
    return bias_dist_df

def add_average_abundance_to_lineage_bias(
    lineage_bias_df: pd.DataFrame,
    ) -> pd.DataFrame:

    g_df = pd.DataFrame(
        lineage_bias_df.groupby(['mouse_id', 'code'])\
            .gr_percent_engraftment.mean()
        ).reset_index()\
            .rename(
                columns={'gr_percent_engraftment': 'average_gr_percent_engraftment'}
            )

    b_df = pd.DataFrame(
        lineage_bias_df.groupby(['mouse_id', 'code'])\
            .b_percent_engraftment.mean()
        ).reset_index()\
            .rename(
                columns={'b_percent_engraftment': 'average_b_percent_engraftment'}
            )
    
    g_b_df = g_df.merge(
        b_df,
        on=['mouse_id', 'code'],
        validate='1:1',
        how='inner'
    )
    both_df = g_b_df.assign(
        average_sum_abundance=lambda x: x.average_b_percent_engraftment + x.average_gr_percent_engraftment
    )
    new_lineage_bias_df = lineage_bias_df.merge(
        both_df,
        on=['mouse_id', 'code'],
        how='inner',
        validate='m:1'
    )
    return new_lineage_bias_df

def add_first_last_to_lineage_bias(
    lineage_bias_df: pd.DataFrame,
    timepoint_col: str,
    ) -> pd.DataFrame:

    filt_df = pd.DataFrame()

    for _, group_df in lineage_bias_df.groupby(['mouse_id', 'code']):
        if len(group_df) < 2:
            continue
        sort_df = group_df.sort_values(by=timepoint_col).reset_index()
        sort_df['time_description'] = 'Middle'
        sort_df.loc[0, 'time_description'] = 'First'
        sort_df.loc[len(sort_df) -1, 'time_description'] = 'Last'
        filt_df = filt_df.append(sort_df, ignore_index=True)
    return filt_df


def add_time_difference(
    lineage_bias_or_clonal_abundance_df: pd.DataFrame,
    timepoint_col: str,
) -> pd.DataFrame:

    # Find Last Time Point
    last_timepoint_df = pd.DataFrame(
        lineage_bias_or_clonal_abundance_df.groupby(UNIQUE_CODE_COLS)[timepoint_col].max()
    ).reset_index().rename(columns={
        timepoint_col: 't2'
    })
    last_timepoint_df['t1'] = lineage_bias_or_clonal_abundance_df[timepoint_col].min()

    # Find Total Change in Time Clone Survives
    time_diff_df = last_timepoint_df.assign(
        total_time_change=lambda x: x.t2 - x.t1
    )

    # Assign Time Change to all timepoints relative to first time point
    with_t_diff_df = lineage_bias_or_clonal_abundance_df.merge(
        time_diff_df,
        on=UNIQUE_CODE_COLS,
        how='outer',
        validate='m:1'
    ).assign(
        time_change=lambda x: x[timepoint_col] - x.t1
    )

    return with_t_diff_df

def define_bias_category(lineage_bias: float) -> str:
    """ Defines categorical classification for lineage bias dataframes
        Intended for use in assigning an additional row to LB dataframes
    Arguments:
        row {pd.Series} -- row of a lineage_bias dataframe

    Returns:
        str -- categorical classification of lineage bias
    """
    balanced_angle_min = pi/8
    balanced_value_min = sin(2 * (balanced_angle_min - (pi/4)))
    balanced_angle_max = 3*pi/8
    balanced_value_max = sin(2 * (balanced_angle_max - (pi/4)))
    comit_val = 1
    
    #if lineage_bias >= comit_val:
        #return 'MC'
    #if lineage_bias <= -comit_val:
        #return 'LC'
    if lineage_bias >= balanced_value_max:
        return 'MB'
    if lineage_bias <= balanced_value_min:
        return 'LB'
    return 'B'

def add_bias_category(
        lineage_bias_df: pd.DataFrame
    ) -> pd.DataFrame:
    """ Appends long (7 bins) and short (5 bins) descriptions lineage bias
    
    Arguments:
        lineage_bias_df {pd.DataFrame}
    
    Returns:
        lineage_bias_df
    """

    lineage_bias_df['bias_category'] = lineage_bias_df.lineage_bias.apply(
        define_bias_category,
    )
    lineage_bias_df['bias_category_long'] = lineage_bias_df.bias_category.map(
       MAP_LINEAGE_BIAS_CATEGORY
    )
    lineage_bias_df['bias_category_short'] = lineage_bias_df.bias_category.map(
       MAP_LINEAGE_BIAS_CATEGORY_SHORT
    )
    return lineage_bias_df

def add_lineage_bias_labels_for_survival(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    with_time_diff_df = add_time_difference(
        lineage_bias_df,
        timepoint_col
    )
    with_bias_cat = add_bias_category(
        with_time_diff_df,
    )
    return with_bias_cat

def not_survived_bias_by_time_change(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        ignore_bias_cat: bool = False,
    ) -> pd.DataFrame:

    with_labels_df = add_lineage_bias_labels_for_survival(
        lineage_bias_df,
        timepoint_col
    )
    time_changes = with_labels_df['time_change'].unique()
    not_survived_count_df = pd.DataFrame()
    groupby_cols = ['mouse_id', 'group', 'bias_category']
    if ignore_bias_cat:
        groupby_cols = ['mouse_id', 'group']
    for t_diff in time_changes:
        not_survived_df = with_labels_df[with_labels_df.total_time_change == t_diff]
        not_survived_counts = pd.DataFrame(
            not_survived_df.groupby(groupby_cols).code.nunique()
        ).reset_index().rename(columns={'code': 'count'})
        not_survived_counts['time_survived'] = t_diff + 1
        not_survived_count_df = not_survived_count_df.append(not_survived_counts)
    return not_survived_count_df


def add_avg_abundance_until_timepoint_clonal_abundance_df(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    output_df = pd.DataFrame()
    print('...Adding average abundance until each time point per clone/cell_type')
    for _, g_df in progressbar.progressbar(clonal_abundance_df.groupby(UNIQUE_CODE_COLS + ['cell_type'])):
        if len(g_df) > clonal_abundance_df[timepoint_col].nunique():
            print(g_df)
            raise ValueError('More time points than should exist for a unique clone/cell-type')
        sort_df = g_df.sort_values(by=timepoint_col).reset_index()
        for i in range(len(sort_df)):
            sort_df.loc[i, 'avg_abundance'] = sort_df.loc[:i, ['percent_engraftment']].sum(axis=1).mean()
        output_df = output_df.append(sort_df)
    return output_df

def add_avg_abundance_until_timepoint(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    output_df = pd.DataFrame()
    for _, g_df in progressbar.progressbar(lineage_bias_df.groupby(UNIQUE_CODE_COLS)):
        sort_df = g_df.sort_values(by=timepoint_col).reset_index()
        for i in range(len(sort_df)):
            sort_df.loc[i, 'accum_abundance'] = sort_df.loc[:i, ['myeloid_percent_abundance', 'lymphoid_percent_abundance']].sum(axis=1).mean()
            sort_df.loc[i, 'myeloid_avg_abundance'] = sort_df.loc[:i, ['myeloid_percent_abundance']].sum(axis=1).mean()
            sort_df.loc[i, 'lymphoid_avg_abundance'] = sort_df.loc[:i, ['lymphoid_percent_abundance']].sum(axis=1).mean()
        output_df = output_df.append(sort_df)
    return output_df


def filter_bias_change_timepoint(
        bias_change_df: pd.DataFrame,
        timepoint: Any,
    ) -> pd.DataFrame:
    """ Only include bias changes for clones with data at required timepoint
    
    Arguments:
        bias_change_df {pd.DataFrame} 
        timepoint {Any}
    
    Returns:
        pd.DataFrame
    """

    if timepoint == 'last':
        filt_df = pd.DataFrame()
        for mouse_id, m_df in bias_change_df.groupby('mouse_id'):
            last_tp = m_df['last_timepoint'].max()
            filt_df = filt_df.append(m_df[
                m_df.last_timepoint.isin([last_tp])
            ])
    elif timepoint == 'first':
        filt_df = pd.DataFrame()
        for mouse_id, m_df in bias_change_df.groupby('mouse_id'):
            first_tp = m_df['first_timepoint'].min()
            filt_df = filt_df.append(m_df[
                m_df.first_timepoint.isin([first_tp])
            ])
    else:
        filt_df = bias_change_df[
            (bias_change_df.first_timepoint.isin([float(timepoint)])) | \
            (bias_change_df.last_timepoint.isin([float(timepoint)]))
        ]
    return filt_df

def calculate_survival_perc(
        clonal_survival_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    """ Calculates percentage of clones
         survived and exhausted at each timepoint
    
    Arguments:
        clonal_survival_df {pd.DataFrame} -- output from create_lineage_bias_survival_df()
    
    Returns:
        pd.DataFrame
    """
    survival_counts = pd.DataFrame(
        clonal_survival_df.groupby(
            ['survived', 'time_change', 'bias_category', timepoint_col, 'mouse_id', 'group']
        ).code.nunique()
    ).reset_index()
    survived = survival_counts[survival_counts['survived'] == 'Survived'].rename(
        columns={'code': 'survived_count'}
    )
    exhausted = survival_counts[survival_counts['survived'] == 'Exhausted'].rename(
        columns={'code': 'exhausted_count'}
    )
    survival_perc = survived.merge(
        exhausted,
        on=['mouse_id', 'bias_category', 'time_change', timepoint_col, 'group'],
        how='outer',
        validate='1:1'
    ).fillna(0).assign(
        exhausted_perc=lambda x: 100 * x.exhausted_count / (x.exhausted_count + x.survived_count),
        survived_perc=lambda x: 100 * x.survived_count / (x.exhausted_count + x.survived_count)
    )
    first_time = clonal_survival_df[timepoint_col].min()
    survival_perc = survival_perc.assign(
        last_time=lambda x: x.time_change + first_time
    )
    return survival_perc

def filter_lymphoid_exhausted_at_time(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        timepoint: int,
    ) -> pd.DataFrame:
    survival_df = create_lineage_bias_survival_df(
        lineage_bias_df,
        timepoint_col
    )
    min_time =  lineage_bias_df[timepoint_col].min()
    filt_df = survival_df[
        (survival_df.survived == 'Exhausted') &\
        (survival_df.bias_category.isin(['LC', 'LB'])) &\
        (survival_df['time_change'] == int(timepoint) - min_time)
    ]
    return filt_df

def filter_first_last_by_mouse(
        input_df: pd.DataFrame,
        timepoint_col: str,
        include_middle: bool = False,
    ) -> pd.DataFrame:
    out_df = pd.DataFrame()
    for _ , m_df in input_df.groupby('mouse_id'):
        first_t = m_df[timepoint_col].min()
        if timepoint_col == 'month':
            first_t = 9
        last_t = m_df[timepoint_col].max()
        out_df = out_df.append(
            get_clones_at_timepoint(
                m_df,
                timepoint_col,
                'first',
                by_mouse=True
            ).assign(
                mouse_time_desc='First'
            )
        )
        out_df = out_df.append(
            get_clones_at_timepoint(
                m_df,
                timepoint_col,
                'last',
                by_mouse=True
            ).assign(
                mouse_time_desc='Last'
            )
        )
        if include_middle:
            out_df = out_df.append(
                m_df[~m_df[timepoint_col].isin([first_t, last_t])].assign(
                    mouse_time_desc='Middle'
                )
            )
    return out_df

def exhausted_clones_without_MPPs(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        present_thresh: float,
    ):
    if 'percent_engraftment' not in clonal_abundance_df.columns:
        print(clonal_abundance_df.columns)
        raise ValueError('Cannot find "percent_engraftment" column. Input must be clonal abundance Dataframe')

    if timepoint_col != 'month':
        raise ValueError('Current Method Only Available for Time Course Data')

    with_time_labels_df = filter_first_last_by_mouse(
        clonal_abundance_df,
        timepoint_col,
        include_middle=True,
    )
    present_clones_df = with_time_labels_df[with_time_labels_df.percent_engraftment >= present_thresh] 
    exhaustion_df = pd.DataFrame()
    for _, g_df in present_clones_df.groupby(['code', 'mouse_id', 'group']):
        if g_df[timepoint_col].nunique() < 2:
            continue

        # Only append clones with both month 4 and 9
        if not g_df[(g_df[timepoint_col] == 9)].empty:
            if not g_df[(g_df[timepoint_col] == 4)].empty:
                # If clone has the last timepoint for a mouse,
                if not g_df[g_df.mouse_time_desc.isin(['Last'])].empty:
                    temp_df = g_df.assign(exhausted=False, survived='Survived')
                else:
                    temp_df = g_df.assign(exhausted=True, survived='Exhausted')
            else:
                continue
        else:
            continue
        exhaustion_df = exhaustion_df.append(temp_df)
    return exhaustion_df

def label_activated_clones(
    clonal_abundance_df: pd.DataFrame,
    timepoint_col: str,
    present_thresh: float,
):
    present_clones = clonal_abundance_df[
        clonal_abundance_df.percent_engraftment >= present_thresh
    ]
    last_clones = get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        'last',
        by_mouse=True,
    )
    last_clones = last_clones[last_clones.percent_engraftment >= 0.01]
    piv = present_clones.pivot_table(
        index=['code', 'mouse_id'],
        columns=timepoint_col,
        values='percent_engraftment',
        aggfunc=np.max,
    )
    # Activated clones are present at M12 or 15, NOT 4/9
    if timepoint_col == 'month':
        act_bool = (piv[[4, 9]].isna().sum(axis=1) == 2)
        print(piv[[4, 9]].isna().sum(axis=1).describe())
    if timepoint_col == 'gen':
        act_bool = (piv[[1, 2]].isna().sum(axis=1) == 2)

    activating = piv[act_bool].reset_index()[['code', 'mouse_id']]
    activating = last_clones.merge(
        activating,
        how='inner',
        validate='m:1'
    )[['code', 'mouse_id']].drop_duplicates()
    activating['survived'] = 'Activated'
    labeled = clonal_abundance_df
    for m, m_df in activating.groupby('mouse_id'):
        labeled.loc[
            (labeled.mouse_id == m) & labeled.code.isin(m_df.code.unique()),
            'survived'
        ] = 'Activated'

    labeled.loc[labeled.survived.isna(), 'survived'] = 'Unknown'
    return labeled
        



def abundance_to_long_by_cell_type(
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
    ) -> pd.DataFrame:
    cell_types = clonal_abundance_df['cell_type'].unique()
    cell_type = cell_types[0]

    temp_df = pd.DataFrame()
    c_df = clonal_abundance_df[clonal_abundance_df.cell_type == cell_type]
    c_df = c_df.rename(columns={'percent_engraftment': cell_type + '_percent_engraftment'})
    temp_df = temp_df.append(c_df, ignore_index=True)

    for cell_type in cell_types[1:]:
        c_df = clonal_abundance_df[clonal_abundance_df.cell_type == cell_type]
        c_df = c_df.rename(columns={'percent_engraftment': cell_type + '_percent_engraftment'})
        temp_df = temp_df.merge(
            c_df[['mouse_id', 'code', timepoint_col, cell_type + '_percent_engraftment']],
            on=['mouse_id', 'code', timepoint_col],
            validate='1:1',
            how='left'
        )

    return temp_df



def mark_outliers(input_df: pd.DataFrame, outlier_df: pd.DataFrame):
    outlier_mask = outlier_df['unadj_p'] < 0.05
    temp_df = input_df[outlier_mask].assign(outlier=True)
    temp_df = temp_df.append(input_df[~outlier_mask].assign(outlier=False), ignore_index=True)
    return temp_df

def get_hsc_abundance_perc_per_mouse(
        clonal_abundance_df: pd.DataFrame,
    ):
    if clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc'].empty:
        raise ValueError('Input DF Has No HSC Data')
    
    hsc_data = clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']\
        .rename(columns={'percent_engraftment': 'hsc_percent_engraftment'})
    total_hsc_per_mouse = pd.DataFrame(
        hsc_data.groupby(['mouse_id']).hsc_percent_engraftment.sum()
    ).reset_index().rename(
        columns={'hsc_percent_engraftment': 'hsc_total_abundance'}
    )
    hsc_data = hsc_data.merge(
        total_hsc_per_mouse,
        validate='m:1'
    )
    hsc_data = hsc_data.assign(
        perc_tracked_hsc=lambda x: 100 * x.hsc_percent_engraftment/x.hsc_total_abundance
    )
    return hsc_data

def filter_gxd_first_last(
    abundance_with_gxd_df: pd.DataFrame,
    timepoint_col: str,
    threshold: float,
) -> List[str]:
    mice_left = abundance_with_gxd_df.mouse_id.unique()
    abundance_with_gxd_df['gfp_x_donor'] = abundance_with_gxd_df['gfp_x_donor'].round(decimals=2) 
    for t in ['first', 'last']:
        t_df = get_clones_at_timepoint(
            abundance_with_gxd_df,
            timepoint_col,
            t,
            by_mouse=True,
        )
        t_df = t_df[t_df.cell_type.isin(['gr', 'b'])]
        t_filt_gxd = t_df[t_df.gfp_x_donor >= threshold]
        filt_agg = pd.DataFrame(
            t_filt_gxd.groupby('mouse_id').cell_type.nunique()
        ).reset_index()
        filt_agg = filt_agg[filt_agg.cell_type == 2]
        mice_left = [m for m in mice_left if m in filt_agg.mouse_id.unique()]
    return mice_left

def get_clones_exist_first_and_last_per_mouse(
        input_df: pd.DataFrame(),
        timepoint_col: str,
    ):
    first_df = get_clones_at_timepoint(
        input_df,
        timepoint_col,
        'first',
        by_mouse=True,
    )
    last_df = get_clones_at_timepoint(
        input_df,
        timepoint_col,
        'last',
        by_mouse=True,
    )
    both_df = first_df.merge(
        last_df[['mouse_id', 'code']].drop_duplicates(),
        how='inner',
        validate='m:1'
    )
    output_df = input_df.merge(
        both_df[['mouse_id', 'code']].drop_duplicates(),
        how='inner',
        validate='m:1'
    )
    return output_df

def filter_lineage_bias_n_timepoints_threshold(
        lineage_bias_df: pd.DataFrame,
        threshold: float,
        n_timepoints: int,
        timepoint_col:str,
    ) -> pd.DataFrame:
    filt_df = pd.DataFrame()
    for _, group in lineage_bias_df.groupby(['mouse_id', 'code']):
        if len(group) < n_timepoints:
            continue
        pass_thresh = group[
            (group.gr_percent_engraftment > threshold) |\
            (group.b_percent_engraftment > threshold)
        ]
        if pass_thresh[timepoint_col].nunique() < n_timepoints:
            continue
        else:
            filt_df = filt_df.append(group)
    return filt_df

def calc_min_hsc_per_mouse(
        cell_count_file: str,
        gfp_file: str,
        donor_file: str,
        tenx_hsc_counts_file: str,
    ):

    cell_count_df = parse_wbc_count_file(cell_count_file, ['hsc', 'wbc'])

    wbc_count_df = cell_count_df[cell_count_df.cell_type == 'wbc'].rename(columns={'cell_count': 'wbc_count'})
    tenx_hsc_df = pd.read_csv(tenx_hsc_counts_file)
    hsc_count_df = cell_count_df[cell_count_df.cell_type == 'hsc']
    gfp_df = parse_wbc_count_file(gfp_file, ['hsc']).rename(columns={'cell_count':'gfp_perc'})
    donor_df = parse_wbc_count_file(donor_file, ['hsc']).rename(columns={'cell_count':'donor_perc'})

    facs_data = wbc_count_df[['mouse_id']].drop_duplicates()
    if hsc_count_df.empty:
        print(Fore.YELLOW + 'Warning: No Cell Count Data for HSCs -- setting to 2000 HSC cells')
        facs_data['cell_count'] = 2000
    else:
        facs_data = facs_data.merge(
            hsc_count_df,
        )
    for mouse_id, m_df in tenx_hsc_df.groupby('mouse_id'):
        if not facs_data[facs_data.mouse_id == mouse_id].empty:
            print('\t Using 10x HSC count to filter: ', mouse_id)
            facs_data.loc[facs_data.mouse_id == mouse_id, 'cell_count'] = m_df['hsc'].values[0]


    if donor_df.empty:
        print( Fore.YELLOW + 'Warning: No Donor Chimerism Data for HSCs -- setting to 100 percent Donor Chimerism')
        facs_data['donor_perc'] = 100
    else:
        facs_data = facs_data.merge(
            donor_df.fillna(value=100),
            validate='1:1',
        )
    # (DONOR)/ (HSC_COUNT)
    facs_data['min_eng_hsc'] = facs_data['donor_perc'] / (facs_data['cell_count'])
    return facs_data[['mouse_id', 'min_eng_hsc']].drop_duplicates()

def merge_hsc_min_abund(
        input_df: pd.DataFrame,
        min_hsc_per_mouse: pd.DataFrame,
    ) -> pd.DataFrame:
    """ Add minimum Abundance in 1 HSC data to input DF as new column
    Column added is called 'min_eng_hsc' 

    Arguments:
        input_df {pd.DataFrame}
        min_hsc_per_mouse {pd.DataFrame}
    
    Returns:
        pd.DataFrame
    """
    return input_df.merge(
        min_hsc_per_mouse,
        how='inner',
        validate='m:1'
    )

def add_almost_zero(
        input_df: pd.DataFrame,
        col: str,
        min_div_factor: float,
    ):
    """ Add an arbitrarily small amount to zero values for log scale plots
    
    Arguments:
        input_df {pd.DataFrame}
        col {str} -- column to add value to
        min_div_factor {float} -- amount of minimum non-zero to divide by
    
    Returns:
        pd.DataFrame -- input_df but with sepecified column uniformly increased
    """
    min_not_zero = input_df[input_df[col] != 0][col].min()
    almost_zero = min_not_zero/min_div_factor
    print(Fore.YELLOW + 'Adding ' + str(almost_zero) + ' to: ' + col)
    input_df[col] = input_df[col] + almost_zero
    return input_df

def filter_for_survival_in_serial_transplant(
        survival_df: pd.DataFrame,
    ):
    survival_df = survival_df[survival_df['gen'] != 8.5]

    # Survived Clones only those in generation 8
    survived = survival_df[survival_df['survived'] == 'Survived']
    survived = survived[survived['gen'] == 8]

    # Exhausted clones are those exhausted at any point
    exhaust = survival_df[survival_df['survived'] == 'Exhausted']

    return survived.append(exhaust)


def filter_lineage_bias_cell_type_ratio_per_mouse(
        lineage_bias_df: pd.DataFrame,
        timepoint_col: str,
        wbc_df: pd.DataFrame,
        filter_threshold: float,
        myeloid_cell:str,
        lymphoid_cell:str,
    ) -> pd.DataFrame:
    """ Filter Lineage bias based on the ratio of Gr
    
    Arguments:
        lineage_bias_df {pd.DataFrame} -- [description]
        timepoint_col {str} -- column time is calculated with
        WBC_file_path {str} -- cell count from facs data
        filter_threshold {float} -- B Abundance to filter
    
    Returns:
        pd.DataFrame -- filtered lineage bias dataframe
    """
    print('\n Filtering Lineage Bias Ratio Based on Gr/B Ratio Per Mouse at First Time Point')
    # TODO: Instead of merging gr and b, stack the dataframe
    filt_df = lineage_bias_df

    # Only use data from first time point to calculate ratio
    first_day = wbc_df['day'].min()
    first_day_per_mouse = pd.DataFrame(
        wbc_df.groupby(['mouse_id']).day.min()
        ).reset_index()
    wbc_df = wbc_df.merge(
        first_day_per_mouse,
        how='inner',
        validate='m:1'
    )

    pivotted = wbc_df.pivot_table(
        index='mouse_id',
        columns='cell_type',
        values='cell_count'
    )
    pivotted['m-l_ratio'] = pivotted[myeloid_cell]/pivotted[lymphoid_cell]
    pivotted['m_filter'] = pivotted['m-l_ratio'] * filter_threshold
    pivotted['l_filter'] = filter_threshold
    print('Length Before Filtering: ', len(lineage_bias_df))

    filters = pivotted[['m_filter', 'l_filter']].reset_index()
    with_filters_df = lineage_bias_df.merge(
        filters,
        how='inner',
        validate='m:1',
    )
    filt_df = with_filters_df[
        (with_filters_df['myeloid_percent_abundance'] >= with_filters_df['m_filter']) |\
        (with_filters_df['lymphoid_percent_abundance'] >= with_filters_df['l_filter'])
    ]
    print('Length After Filtering: ', len(filt_df), '\n')

    return filt_df

def fill_mouse_id_zeroes(
        clonal_abundance_df: pd.DataFrame,
        info_cols: List[str],
        fill_col: str,
        fill_cat_col: str,
        fill_cats: List[str],
        fill_val: Any = 0,
        mice_to_fill: List = None,
    ) -> pd.DataFrame:

    if 'mouse_id' in info_cols:
        raise ValueError('Mouse ID Cannot be in info cols')

    filled_df = pd.DataFrame()
    if mice_to_fill is not None:
        for mouse_id in mice_to_fill:
            m_df = clonal_abundance_df[clonal_abundance_df.mouse_id == mouse_id]
            for cat in fill_cats:
                if m_df[m_df[fill_cat_col] == cat].empty:
                    fill_vals = pd.DataFrame.from_dict(
                        {
                            'mouse_id': [mouse_id],
                            fill_col: [fill_val],
                            fill_cat_col: [cat],
                        }
                    )
                    filled_df = filled_df.append(fill_vals)
    else:
        for mouse_id, m_df in clonal_abundance_df.groupby(['mouse_id']):
            for cat in fill_cats:
                if m_df[m_df[fill_cat_col] == cat].empty:
                    fill_vals = pd.DataFrame.from_dict(
                        {
                            'mouse_id': [mouse_id],
                            fill_col: [fill_val],
                            fill_cat_col: [cat],
                        }
                    )
                    filled_df = filled_df.append(fill_vals)
    if filled_df.empty:
        return clonal_abundance_df
    with_info = filled_df.merge(
        clonal_abundance_df[info_cols + ['mouse_id']].drop_duplicates(),
        on=['mouse_id'],
        how='left',
    )
    if with_info[info_cols].isna().any(axis=None):
        print(Fore.YELLOW + 'Warning: NA Values in info cols')
        for col in info_cols:
            print(with_info.groupby('mouse_id')[col].unique())
    return clonal_abundance_df.append(with_info).drop_duplicates()

def filter_low_abund_hsc(
        min_hsc_per_mouse: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
    ) -> pd.DataFrame:
    with_min_hsc = clonal_abundance_df.merge(
        min_hsc_per_mouse[['mouse_id', 'min_eng_hsc']].drop_duplicates(),
        how='left',
        validate='m:1',
    )
    no_low_hsc = with_min_hsc[
        ~(
            (with_min_hsc.cell_type == 'hsc') &\
            (with_min_hsc.percent_engraftment < with_min_hsc.min_eng_hsc )
        )
    ]
    print('Length before filtering HSCs:', len(clonal_abundance_df[clonal_abundance_df.cell_type == 'hsc']))
    print('Length after filtering HSCs:', len(no_low_hsc[no_low_hsc.cell_type == 'hsc']))
    return no_low_hsc

def remove_month_17(
    input_df: pd.DataFrame,
    timepoint_col,
) -> pd.DataFrame:
    if timepoint_col != 'month':
        return input_df
    if 17 in input_df[timepoint_col].unique():
        print (Fore.YELLOW + '\t REMOVING MONTH 17 DATA')
        return input_df[input_df.month != 17]
    return input_df

def remove_month_17_and_6(
    input_df: pd.DataFrame,
    timepoint_col,
) -> pd.DataFrame:
    if timepoint_col != 'month':
        return input_df
    months_in_data = []
    if 17 in input_df[timepoint_col].unique():
        months_in_data.append(17)
    if 6 in input_df[timepoint_col].unique():
        months_in_data.append(6)
    if input_df[timepoint_col].max() == 6:
        return input_df
    if months_in_data:
        print (Fore.YELLOW + '\t REMOVING MONTHS: ' + ', '.join([str(x) for x in months_in_data]))
        filt_df = input_df[~input_df.month.isin([17, 6])]
        return filt_df
    else:
        return input_df

def label_exhausted_clones(
        add_labels_df: pd.DataFrame,
        clonal_abundance_df: pd.DataFrame,
        timepoint_col: str,
        present_thresh: float
    ) -> pd.DataFrame:
    """ Add exhaustion/survival labels to clones
    
    Arguments:
        add_labels_df {pd.DataFrame} -- Data frame to label,
            if None labels clonal abundance dataframe
        clonal_abundance_df {pd.DataFrame}
        timepoint_col {str}
    
    Returns:
        pd.DataFrame -- Labeled 'add_labels_df'
    """
    if add_labels_df is None:
        add_labels_df = clonal_abundance_df

    no_hsc_df = clonal_abundance_df[clonal_abundance_df.cell_type != 'hsc']
    if timepoint_col == 'month':
        unlabeled_exhaust_df = exhausted_clones_without_MPPs(
            no_hsc_df,
            timepoint_col,
            present_thresh
        )
    
        abundance_with_time_difference = add_time_difference(
            unlabeled_exhaust_df,
            timepoint_col
        )

        exhaust_df = abundance_with_time_difference.assign(
            isLast=lambda x: x.total_time_change == x.time_change
        )
    elif timepoint_col == 'gen':
        exhaust_df = exhausted_clones_serial_transplant(
            no_hsc_df,
            timepoint_col,
            present_thresh=present_thresh,
            match_aging=True,
        )
    else:
        exhaust_df = pd.DataFrame()
        no_hsc_df = no_hsc_df[
            no_hsc_df.percent_engraftment >= present_thresh
        ]
        abundance_with_time_difference = add_time_difference(
            no_hsc_df,
            timepoint_col
        )

        last_labeled_df = abundance_with_time_difference.assign(
            isLast=lambda x: x.total_time_change == x.time_change
        )
        not_survived = last_labeled_df[last_labeled_df.isLast].assign(
            survived='Exhausted'
        )
        survived = last_labeled_df[~last_labeled_df.isLast].assign(
            survived='Survived'
        )
        exhaust_df = survived.append(not_survived)

    exhaust_df = exhaust_df[
        ['mouse_id', 'group', 'code', 'survived']
        ].drop_duplicates()

    input_with_labels = add_labels_df.merge(
        exhaust_df,
        how='left',
    )
    print('\tLength before adding exhaustion labels:', len(add_labels_df))
    print('\tLength after adding exhaustion labels:', len(input_with_labels))
    return input_with_labels

def exhausted_clones_serial_transplant(
    no_hsc_df: pd.DataFrame,
    timepoint_col: str,
    present_thresh: float,
    match_aging: bool = False,
    ):
    if timepoint_col != 'gen':
        raise ValueError('Time point column must be gen, recieved ' + str(timepoint_col) )

    if match_aging:
        print('MATCHING AGING TYPE EXHAUSTION FOR S.T. DATA')
        last_clones = get_clones_at_timepoint(no_hsc_df, timepoint_col, 'last', by_mouse=True)
        last_clones = last_clones[last_clones.percent_engraftment >= present_thresh]

        last_clones = last_clones[last_clones.gen >= 7]

        grb_df = no_hsc_df[no_hsc_df.cell_type.isin(['gr','b'])]
        grb_df = grb_df[grb_df.percent_engraftment >= present_thresh]

        piv = grb_df[['mouse_id', 'code', 'gen', 'percent_engraftment']].pivot_table(
            values='percent_engraftment',
            index=['code', 'mouse_id'],
            columns='gen',
            aggfunc=np.max,
        )
        exh_bool = (
            (piv[[1, 2]].isna().sum(axis=1) == 0) &
            (piv[[7,8]].isna().sum(axis=1) == 0)
        )
        surv_bool = (
            (piv[[1, 2]].isna().sum(axis=1) == 0)
        )
        exhausted = piv[exh_bool].reset_index()[['code', 'mouse_id']].drop_duplicates()
        exhausted['survived'] = 'Exhausted'
        exhausted = exhausted.merge(no_hsc_df, how='inner', validate='1:m')

        survived = piv[surv_bool].reset_index()[['code', 'mouse_id']].drop_duplicates()
        survived['survived'] = 'Survived'
        survived = last_clones.merge(survived, how='inner', validate='m:1')
        return pd.concat([exhausted, survived])


    present_no_hsc_df = no_hsc_df[no_hsc_df.percent_engraftment >= present_thresh]
    last_two_time_points = present_no_hsc_df[present_no_hsc_df['gen'].isin([7, 8])]

    # Count number times clone in last 2 gens
    last_gens_count = pd.DataFrame(
        last_two_time_points.groupby(
            ['mouse_id', 'code']
        )[timepoint_col].nunique()
    ).reset_index()

    # Clones not in last two time points are exhausted
    not_last_two_time_points = present_no_hsc_df[~present_no_hsc_df.code.isin(last_two_time_points.code.unique())]
    exhausted_clones = not_last_two_time_points[['mouse_id', 'code']]\
        .drop_duplicates()\
        .assign(survived='Exhausted')

    # Clones in 7 and 8 are survived
    survived_clones = last_gens_count[last_gens_count[timepoint_col] == 2]\
        [['mouse_id', 'code']].drop_duplicates()\
        .assign(survived='Survived')
    
    abundance_with_time_difference = add_time_difference(
        present_no_hsc_df,
        timepoint_col
    )

    last_labeled_df = abundance_with_time_difference.assign(
        isLast=lambda x: x.total_time_change == x.time_change
    )

    surv_df = last_labeled_df.merge(
        survived_clones,
        how='inner',
        validate='m:1'
    )
    ex_df = last_labeled_df.merge(
        exhausted_clones,
        how='inner',
        validate='m:1'
    )
    return pd.concat([surv_df, ex_df])


def remove_gen_8_5(
    input_df: pd.DataFrame,
    timepoint_col: str,
    keep_hsc: bool,
) -> pd.DataFrame:
    if timepoint_col != 'gen':
        return input_df
    gens_in_data = []
    if 8.5 in input_df[timepoint_col].unique():
        gens_in_data.append(8.5)
    if gens_in_data:
        print (Fore.YELLOW + '\t REMOVING GENS: ' + ', '.join([str(x) for x in gens_in_data]))
        filt_df = input_df.copy()
        if keep_hsc:
            print(Fore.YELLOW + '\t Keepig HSC data, setting generation of HSC data to 8')
            filt_df.loc[filt_df.cell_type == 'hsc', timepoint_col] = 8
            filt_df = filt_df[
                (~filt_df.gen.isin(gens_in_data))
                ]
        else:
            filt_df = filt_df[~filt_df.gen.isin(gens_in_data)]
        filt_df['gen'] = filt_df['gen'].astype(int)
        return filt_df
    else:
        return input_df

def add_short_group(input_df: pd.DataFrame):
    SHORT_GROUP_MAP = {
        'aging_phenotype': 'E',
        'no_change': 'D'
    }
    input_df.loc[:,'group_short'] = input_df.group.map(SHORT_GROUP_MAP)
    return input_df

def label_lymphoid_comitted(
    lineage_bias_df: pd.DataFrame,
    max_myeloid_abundance: float,
    ) -> pd.DataFrame:
    labeled_bias_df = add_bias_category(lineage_bias_df)
    labeled_bias_df.loc[
        (
            (labeled_bias_df.bias_category == 'LB') &\
            (labeled_bias_df.myeloid_percent_abundance <= max_myeloid_abundance)
        ),
        'bias_category'
    ] = 'LC'
    return labeled_bias_df


def shannon_diversity_wrapper(data):
    return alpha_diversity('shannon', data.percent_engraftment.tolist())


def calculate_shannon_diversity(
    clonal_abundance_df: pd.DataFrame,
    group_cols: list,
    ) -> pd.DataFrame:
    diversity_df = pd.DataFrame(
        clonal_abundance_df.groupby(group_cols)\
            .apply(shannon_diversity_wrapper)
    ).reset_index().rename(columns={0: 'Shannon Diversity'})
    return diversity_df


def get_n_most_abundant_at_time(
    clonal_abundance_df: pd.DataFrame,
    n: int,
    timepoint_col: str,
    timepoint: Any,
    by_mouse: bool,
    ) -> pd.DataFrame:
    """ returns all data from top n clones per cell type at a timepoint
    
    Arguments:
        clonal_abundance_df {pd.DataFrame}
        n {int} -- number of top clones
        timepoint_col {str} -- column to find time
        timepoint {Any} -- time at which to be top
        by_mouse {bool} -- look at time points per mouse
    
    Returns:
        pd.DataFrame
    """
    time_clones = get_clones_at_timepoint(
        clonal_abundance_df,
        timepoint_col,
        timepoint,
        by_mouse
    ).sort_values(by='percent_engraftment', ascending=False)
    top_n = pd.DataFrame(
        time_clones.groupby(['mouse_id', 'cell_type'])\
            .head(n)
    ).reset_index()
    return clonal_abundance_df.merge(
        top_n[['mouse_id', 'code', 'cell_type']],
        how='inner',
        validate='m:1',
    )



