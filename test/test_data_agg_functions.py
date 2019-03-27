import pandas as pd
from pandas.util.testing import assert_series_equal
import numpy.testing as npt
from ..aggregate_functions import filter_threshold, count_clones, find_enriched_clones_at_time, combine_enriched_clones_at_time

TEST_DATA = pd.read_csv('test/test_data/test_all_long.csv')
REAL_DATA = pd.read_csv('test/test_data/Ania_M_all_percent-engraftment_100818_long.csv')

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
    clone_count = count_clones(filt_df)
    # if 1 cell type, total == cell type counts 
    assert_series_equal(clone_count[clone_count.cell_type == 'gr']['code'], clone_count[clone_count.cell_type == 'Total']['code'])

    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    clone_count = count_clones(filt_df)

    # Correct value (2) assigned
    npt.assert_array_equal(clone_count[clone_count.cell_type == 'gr']['code'].values, clone_count[clone_count.cell_type == 'b']['code'].values)

    should_have_no_elements = clone_count[(clone_count['code'] > 4) | ((clone_count['code'] < 1) & (clone_count['code'] != 0))]
    assert should_have_no_elements.empty

def test_enriched_clones():
    threshold = .0
    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    enrichment_month = 4
    enrichment_threshold = .04

    # Finds the right barcode
    enriched_clones_df = find_enriched_clones_at_time(filt_df, enrichment_month, enrichment_threshold, 'b')
    print(enriched_clones_df)
    unique_clones = list(set(enriched_clones_df.code.values))
    assert enriched_clones_df[enriched_clones_df.code == 'gr'].empty
    assert len(enriched_clones_df) == 1
    assert len(unique_clones) == 1
    assert unique_clones[0] == 'AGGA'

def test_enriched_clones_real_data():
    threshold = .01
    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(REAL_DATA, threshold, analyzed_cell_types)
    enrichment_month = 4
    enrichment_threshold = .2

    # Finds the right barcode
    enriched_clones_df = find_enriched_clones_at_time(filt_df, enrichment_month, enrichment_threshold, 'b')
    should_be_empty_index = (enriched_clones_df.cell_type == 'gr') | ((enriched_clones_df.month == enrichment_month) & (enriched_clones_df.percent_engraftment < threshold))
    assert enriched_clones_df[should_be_empty_index].empty
    codes_index = (enriched_clones_df.month == enrichment_month) & (enriched_clones_df.percent_engraftment >= enrichment_threshold)
    assert set(enriched_clones_df[codes_index].code.values) == set(enriched_clones_df.code.values)

    enriched_clones_df = find_enriched_clones_at_time(filt_df, enrichment_month, enrichment_threshold, 'gr')
    should_be_empty_index = (enriched_clones_df.cell_type == 'b') | ((enriched_clones_df.month == enrichment_month) & (enriched_clones_df.percent_engraftment < enrichment_threshold))
    assert enriched_clones_df[should_be_empty_index].empty
    codes_index = (enriched_clones_df.month == enrichment_month) & (enriched_clones_df.percent_engraftment >= enrichment_threshold)
    assert set(enriched_clones_df[codes_index].code.values) == set(enriched_clones_df.code.values)

def test_combine_enriched_clones_real_data():
    threshold = .01
    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(REAL_DATA, threshold, analyzed_cell_types)
    enrichment_month = 4
    enrichment_threshold = .2
    thresholds = {'b':enrichment_threshold}

    # Finds the right barcode
    enriched_df = combine_enriched_clones_at_time(filt_df, enrichment_month, thresholds, 'b')
    should_be_empty_index = (enriched_df.month == enrichment_month) & (enriched_df.percent_engraftment < enrichment_threshold)
    assert enriched_df[should_be_empty_index].empty
    assert enriched_df[(enriched_df.month == enrichment_month) & (enriched_df.percent_engraftment < enrichment_threshold)].empty