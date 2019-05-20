import pandas as pd
from pandas.util.testing import assert_series_equal
import numpy.testing as npt
from ..aggregate_functions import filter_threshold, count_clones, find_enriched_clones_at_time, combine_enriched_clones_at_time

TEST_DATA = pd.read_csv('test/test_data/test_all_long.csv')
REAL_DATA = pd.read_csv('test/test_data/Ania_M_all_percent-engraftment_100818_long.csv')
TIMEPOINT_COL='month'
def test_filter_threshold():
    threshold = .021
    analyzed_cell_types = ['gr']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    assert len(filt_df) == 9
    assert filt_df[filt_df.cell_type == 'b'].empty

    analyzed_cell_types_with_b = ['gr', 'b']
    filt_df_with_b = filter_threshold(TEST_DATA, threshold, analyzed_cell_types_with_b)
    assert len(filt_df_with_b) == 18
    assert len(filt_df_with_b[filt_df_with_b.cell_type == 'b']) == 9

def test_count_clones():
    threshold = .01
    analyzed_cell_types = ['gr']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    clone_count = count_clones(filt_df, timepoint_col=TIMEPOINT_COL)

    # if 1 cell type, total == cell type counts 
    assert_series_equal(clone_count[clone_count.cell_type == 'gr']['code'], clone_count[clone_count.cell_type == 'Total']['code'])

    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    clone_count = count_clones(filt_df, timepoint_col=TIMEPOINT_COL)

    # Correct value (2) assigned
    npt.assert_array_equal(clone_count[clone_count.cell_type == 'gr']['code'].values, [3, 2, 2, 2])
    npt.assert_array_equal(clone_count[clone_count.cell_type == 'b']['code'].values, [2, 2, 2, 2])

    should_have_no_elements = clone_count[(clone_count['code'] > 4) | ((clone_count['code'] < 1) & (clone_count['code'] != 0))]
    assert should_have_no_elements.empty

def test_enriched_clones():
    threshold = .0
    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    enrichment_month = 4
    enrichment_threshold = .04

    # Finds the right barcode
    enriched_clones_df = find_enriched_clones_at_time(filt_df, enrichment_month, enrichment_threshold, 'b', timepoint_col='month')
    print(enriched_clones_df)
    unique_clones = enriched_clones_df.code.unique()
    assert enriched_clones_df.code.nunique() == 1
    assert unique_clones[0] == 'AGG'

def test_enriched_clones_real_data():
    threshold = .01
    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(REAL_DATA, threshold, analyzed_cell_types)
    enrichment_month = 4
    enrichment_threshold = .2

    # Finds the right barcode
    enriched_clones_df = find_enriched_clones_at_time(filt_df, enrichment_month, enrichment_threshold, 'b', timepoint_col=TIMEPOINT_COL)
    should_be_empty_index = (enriched_clones_df.cell_type == 'gr') | \
        (
            (enriched_clones_df.month == enrichment_month) & \
            (enriched_clones_df.percent_engraftment < enrichment_threshold)
        )

    assert enriched_clones_df[should_be_empty_index].empty
    codes_index = (enriched_clones_df.month == enrichment_month) & (enriched_clones_df.percent_engraftment >= enrichment_threshold)
    for code in enriched_clones_df.code.unique():
        if code not in enriched_clones_df[codes_index].code.unique():
            print(enriched_clones_df.loc[enriched_clones_df.code == code])
    npt.assert_array_equal(enriched_clones_df[codes_index].code.unique().sort(), enriched_clones_df.code.unique().sort())

    enriched_clones_df = find_enriched_clones_at_time(filt_df, enrichment_month, enrichment_threshold, 'gr', timepoint_col=TIMEPOINT_COL)
    should_be_empty_index = (enriched_clones_df.cell_type == 'b') | \
        (
            (enriched_clones_df.month == enrichment_month) & \
            (enriched_clones_df.percent_engraftment < enrichment_threshold)
        )

    assert enriched_clones_df[should_be_empty_index].empty
    codes_index = (enriched_clones_df.month == enrichment_month) & (enriched_clones_df.percent_engraftment >= enrichment_threshold)
    for code in enriched_clones_df.code.unique():
        if code not in enriched_clones_df[codes_index].code.unique():
            print(enriched_clones_df.loc[enriched_clones_df.code == code, ['mouse_id','month', 'code']])
    npt.assert_array_equal(enriched_clones_df[codes_index].code.unique(), enriched_clones_df.code.unique())

def test_combine_enriched_clones_real_data():
    threshold = .01
    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(REAL_DATA, threshold, analyzed_cell_types)
    enrichment_month = 14
    enrichment_threshold = .2
    thresholds = {'b':enrichment_threshold}

    # Finds the right barcode
    enriched_df = combine_enriched_clones_at_time(
           filt_df,
           enrichment_month,
           TIMEPOINT_COL,
           thresholds,
           'b',
    )
    should_be_empty_index = (enriched_df.month == enrichment_month) & (enriched_df.percent_engraftment < enrichment_threshold)
    print(enriched_df[should_be_empty_index][['percent_engraftment', 'month', 'cell_type', 'mouse_id']])
    assert enriched_df[should_be_empty_index].empty
    assert enriched_df[(enriched_df.month == enrichment_month) & (enriched_df.percent_engraftment < enrichment_threshold)].empty