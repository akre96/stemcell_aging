""" Test lineage bias calculation scripts
"""

import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from ..lineage_bias import create_lineage_bias_df, calculate_baseline, calc_bias, normalize_to_baseline, normalize_input

TEST_LONG_DATA = pd.read_csv('test/test_data/test_long_for_lineage_bias.csv')
EXPECTED_LINEAGE_BIAS = pd.read_csv('test/test_data/expected_lineage_bias.csv')

def test_no_norm_change_one_mouse():
    test_data = TEST_LONG_DATA.loc[:3]
    norm_data_df = normalize_input(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr','b'])

    # all values should be 1 normalized percent engraftment
    assert_array_equal(norm_data_df.norm_percent_engraftment, test_data.norm_percent_engraftment_expected)
    assert norm_data_df.loc[~norm_data_df.baseline_max.isin([.11, .1])].empty

def test_no_norm_change_two_mice():
    test_data = TEST_LONG_DATA.loc[:7]
    norm_data_df = normalize_input(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr','b'])

    # all values should be 1 normalized percent engraftment
    assert norm_data_df.loc[norm_data_df.norm_percent_engraftment != 1.0].empty

def test_norm_engraftment_change():
    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M3']
    norm_data_df = normalize_input(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
    print(test_data[['cell_type', 'norm_percent_engraftment_expected']])
    print(norm_data_df[['cell_type', 'norm_percent_engraftment']])

    assert_array_equal(test_data.sort_values(by=['cell_type', 'norm_percent_engraftment_expected']).norm_percent_engraftment_expected, norm_data_df.sort_values(by=['cell_type', 'norm_percent_engraftment']).norm_percent_engraftment)

def test_baseline_missing_value():
    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M4']
    baseline_df = calculate_baseline(test_data, baseline_timepoint=4, baseline_column='month')
    print(baseline_df)

    # Currently keeps nan's if no value
    assert_array_equal(baseline_df.baseline_max, baseline_df.baseline_max_expected)

def test_norm_missing_value():
    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M4']
    norm_data_df = normalize_input(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
    print(norm_data_df)

    # Norm percent engraftment set to nan
    assert_array_equal(norm_data_df.norm_percent_engraftment, norm_data_df.norm_percent_engraftment_expected)

def test_lineage_bias_0():
    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M1']
    test_expected = EXPECTED_LINEAGE_BIAS.loc[EXPECTED_LINEAGE_BIAS.mouse_id == 'M1']

    norm_data_df = normalize_input(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
    lineage_bias_df = create_lineage_bias_df(norm_data_df)
    assert_array_equal(lineage_bias_df.lineage_bias, test_expected.lineage_bias_expected)
