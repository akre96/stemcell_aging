""" Test lineage bias calculation scripts

"""

from math import pi, sin
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..lineage_bias import create_lineage_bias_df, calculate_baseline_max, \
    calc_bias, normalize_to_baseline_max, normalize_input_to_baseline_max, \
    calc_angle, calc_bias_from_angle

TEST_LONG_DATA = pd.read_csv('test/test_data/test_long_for_lineage_bias.csv')
EXPECTED_LINEAGE_BIAS = pd.read_csv('test/test_data/expected_lineage_bias.csv')

def test_no_norm_change_one_mouse():
    test_data = TEST_LONG_DATA.loc[:3]
    norm_data_df = normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr','b'])

    # all values should be 1 normalized percent engraftment
    assert_array_equal(norm_data_df.norm_percent_engraftment, test_data.norm_percent_engraftment_expected)
    assert norm_data_df.loc[~norm_data_df.baseline_max.isin([.11, .1])].empty

def test_no_norm_change_two_mice():
    test_data = TEST_LONG_DATA.loc[:7]
    norm_data_df = normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr','b'])

    # all values should be 1 normalized percent engraftment
    assert norm_data_df.loc[norm_data_df.norm_percent_engraftment != 1.0].empty

def test_norm_engraftment_change():
    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M3']
    norm_data_df = normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
    print(test_data[['cell_type', 'norm_percent_engraftment_expected']])
    print(norm_data_df[['cell_type', 'norm_percent_engraftment']])

    assert_array_equal(test_data.sort_values(by=['cell_type', 'norm_percent_engraftment_expected']).norm_percent_engraftment_expected, norm_data_df.sort_values(by=['cell_type', 'norm_percent_engraftment']).norm_percent_engraftment)

def test_baseline_missing_value():
    """ Test that if missing value at baseline, throws ValueError
    """

    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M4']
    try:
        calculate_baseline_max(test_data, baseline_timepoint=4, baseline_column='month')
        assert False
    except ValueError:
        assert True

def test_norm_missing_baseline_value():
    """ Test normalized value, if value missing

    Currently we expect the normalized baseline engraftment to be NaN
    if the value is missing at 4 month, but since no values like that
    exist in current dataset, throws value error.
    """

    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M4']
    try:
        normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
        assert False
    except ValueError:
        assert True


def test_baseline_set_max_mouse():
    """ Test that baseline values are set using the max from all clones for a mouse&day&cell_type

    Test data has two clones for mouse M5 in 2 timepoints
    """
    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M5']
    norm_data_df = normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])

    assert_array_equal(norm_data_df.baseline_max, norm_data_df.baseline_max_expected)
    assert_array_equal(norm_data_df.norm_percent_engraftment, norm_data_df.norm_percent_engraftment_expected)

def test_lineage_bias_0():
    """ Test lineage bias for balanced full data

    Expect the lineage bias value to be 0
    """

    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M1']
    test_expected = EXPECTED_LINEAGE_BIAS.loc[EXPECTED_LINEAGE_BIAS.mouse_id == 'M1']

    norm_data_df = normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
    lineage_bias_df = create_lineage_bias_df(norm_data_df)
    assert_array_equal(lineage_bias_df.lineage_bias, test_expected.lineage_bias_expected)

def test_lineage_bias_change():
    """ Test Lineage bias that should change over time

    Test done on data with complete values all around
    """

    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M3']
    test_expected = EXPECTED_LINEAGE_BIAS.loc[EXPECTED_LINEAGE_BIAS.mouse_id == 'M3']

    norm_data_df = normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
    lineage_bias_df = create_lineage_bias_df(norm_data_df)
    assert_array_almost_equal(lineage_bias_df.lineage_bias, test_expected.lineage_bias_expected, decimal=6)

def test_baseline_gen_nobaseline():
    """ Test baseline calculation if there is no baseline for an entry

    This doesn't happen, expect a valueError

    If there is no 4month baseline value to normalize to,
    normalize to max at current timepoint for mouse, propogate
    that timepoint for any later values.
    """

    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M4']
    try:
        normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
        assert False
    except ValueError:
        assert True


def test_lineage_bias_no_gr():
    """ Test behaviour if no gr value at non-baseline forming month

    """
    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M6']
    test_expected = EXPECTED_LINEAGE_BIAS.loc[EXPECTED_LINEAGE_BIAS.mouse_id == 'M6']
    
    norm_data_df = normalize_input_to_baseline_max(test_data, baseline_timepoint=4, baseline_column='month', analyzed_cell_types=['gr', 'b'])
    lineage_bias_df = create_lineage_bias_df(norm_data_df)
    assert_array_almost_equal(lineage_bias_df.lineage_bias, test_expected.lineage_bias_expected, decimal=6)

def test_bias_angle():
    """ test calculated lineage bias angle is as expected from known scenarios
    """
    assert calc_angle(1, 1) == pi/4
    assert calc_angle(0, 0) == pi/4
    assert calc_angle(1, 0) == pi/2
    assert calc_angle(0, 1) == 0

def test_angle_to_bias():
    """ Test certain angles lead to expected lineage biases
    """
    balanced_angle_min = pi/8
    balanced_value_min = sin(2 * (balanced_angle_min - (pi/4)))
    balanced_angle_max = 3*pi/8
    balanced_value_max = sin(2 * (balanced_angle_max - (pi/4)))
    balanced_angle = pi/4
    
    assert calc_bias_from_angle(balanced_angle) == 0
    assert calc_bias_from_angle(balanced_angle_min) == balanced_value_min
    assert calc_bias_from_angle(balanced_angle_max) == balanced_value_max

