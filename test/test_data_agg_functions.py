import pandas as pd
from pandas.util.testing import assert_series_equal
import numpy.testing as npt
from ..plot_data import filter_threshold, count_clones, find_enriched_clones_at_time

TEST_DATA = pd.read_csv('test/test_data/test_all_long.csv')

def test_filter_threshold():
    threshold = .021
    analyzed_cell_types = ['gr']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    assert len(filt_df) == 8
    assert filt_df[filt_df.cell_type == 'b'].empty

    analyzed_cell_types_with_b = ['gr', 'b']
    filt_df_with_b = filter_threshold(TEST_DATA, threshold, analyzed_cell_types_with_b)
    assert len(filt_df_with_b) == 17
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

    should_have_no_elements = clone_count[(clone_count['code'] > 2) | ((clone_count['code'] < 1) & (clone_count['code'] != 0))]
    assert should_have_no_elements.empty

def test_enriched_clones():
    threshold = .0
    analyzed_cell_types = ['gr', 'b']
    filt_df = filter_threshold(TEST_DATA, threshold, analyzed_cell_types)
    enrichment_month = 4
    enrichment_threshold = .04

    # Finds the right barcode
    enriched_clones = find_enriched_clones_at_time(filt_df, enrichment_month, enrichment_threshold)
    assert len(enriched_clones) == 1
    assert enriched_clones[0] == 'AGG'

