""" Test lineage bias calculation scripts

"""

from math import pi, sin
from argparse import Namespace
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
import sys,os
sys.path.append(os.getcwd())
import lineage_bias as lb
import aggregate_functions as agg
from parse_facs_data import parse_wbc_count_file

TEST_LONG_DATA = pd.read_csv('test/test_data/test_long_for_lineage_bias.csv')
EXPECTED_LINEAGE_BIAS = pd.read_csv('test/test_data/expected_lineage_bias.csv')
TEST_WBC = parse_wbc_count_file(
    'test/test_data/test_WBC.txt',
)

WITH_BASELINE = lb.calculate_baseline_counts(
    TEST_LONG_DATA,
    TEST_WBC,
    baseline_timepoint='by_mouse'
)

def test_calc_baseline_counts():
    WITH_BASELINE = lb.calculate_baseline_counts(
        TEST_LONG_DATA,
        TEST_WBC,
        baseline_timepoint='by_mouse'
    )
    NO_M2 = WITH_BASELINE[WITH_BASELINE.mouse_id != 'M2']
    assert NO_M2[NO_M2['cell_count'].isna()].empty
    assert TEST_LONG_DATA.shape[0] == WITH_BASELINE.shape[0]

def test_no_norm_change_one_mouse():
    test_data = WITH_BASELINE.loc[:3]
    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    print(norm_data_df)
    # all values should be 1 normalized percent engraftment
    assert_array_almost_equal(
        norm_data_df.norm_percent_engraftment,
        test_data.norm_percent_engraftment_expected,
        decimal=10
    )

def test_no_norm_change_two_mice():
    test_data = WITH_BASELINE.loc[:7]
    norm_data_df = lb.normalize_to_baseline_counts(test_data)

    assert_array_almost_equal(
        norm_data_df.norm_percent_engraftment,
        test_data.norm_percent_engraftment_expected,
        decimal=10
    )

def test_norm_engraftment_change():
    test_data = WITH_BASELINE.loc[WITH_BASELINE.mouse_id == 'M3']
    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    print(test_data[['cell_type', 'norm_percent_engraftment_expected']])
    print(norm_data_df[['cell_type', 'norm_percent_engraftment']])

    assert_array_equal(test_data.sort_values(by=['cell_type', 'norm_percent_engraftment_expected']).norm_percent_engraftment_expected, norm_data_df.sort_values(by=['cell_type', 'norm_percent_engraftment']).norm_percent_engraftment)

def test_baseline_missing_value():
    """ Test that if missing value at baseline, passes NaN
    """

    test_data = TEST_LONG_DATA.loc[TEST_LONG_DATA.mouse_id == 'M2']
    bc = lb.calculate_baseline_counts(test_data, TEST_WBC, baseline_timepoint=4)
    assert not bc.empty
    assert bc[~bc.cell_count.isna()].empty

def test_norm_missing_baseline_value():
    """ Test normalized value, if value missing is NaN

    """

    test_data = WITH_BASELINE.loc[WITH_BASELINE.mouse_id == 'M2']
    nbc = lb.normalize_to_baseline_counts(test_data)
    assert not nbc.empty
    assert nbc[~nbc.norm_percent_engraftment.isna()].empty


def test_lineage_bias_myeloid():
    """ Test lineage bias for balanced full data

    Expect the lineage bias value to be 0
    """

    test_data = WITH_BASELINE.loc[WITH_BASELINE.mouse_id == 'M1']

    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    lineage_bias_df = lb.create_lineage_bias_df(norm_data_df, lymphoid_cell_type='b', myeloid_cell_type='gr')
    with_exp = lineage_bias_df.merge(EXPECTED_LINEAGE_BIAS)
    print(with_exp)
    print(lineage_bias_df.columns)
    print(with_exp.columns)
    # Ensure that adds only 1 column
    assert lineage_bias_df.shape[0] == with_exp.shape[0]
    assert (lineage_bias_df.shape[1] + 1) == with_exp.shape[1]
    assert_array_almost_equal(
        lineage_bias_df.lineage_bias,
        with_exp.lineage_bias_expected,
    )

def test_lineage_bias_0():
    """ Test Lineage bias that should change over time

    Test done on data with complete values all around
    """

    test_data = WITH_BASELINE.loc[WITH_BASELINE.mouse_id == 'M3']

    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    lineage_bias_df = lb.create_lineage_bias_df(norm_data_df, lymphoid_cell_type='b', myeloid_cell_type='gr')
    with_exp = lineage_bias_df.merge(EXPECTED_LINEAGE_BIAS)
    print(lineage_bias_df[['myeloid_percent_abundance', 'lymphoid_percent_abundance']])

    assert lineage_bias_df.shape[0] == with_exp.shape[0]
    assert (lineage_bias_df.shape[1] + 1) == with_exp.shape[1]
    assert_array_almost_equal(
        lineage_bias_df.lineage_bias,
        with_exp.lineage_bias_expected,
    )

def test_lineage_bias_lymphoid():
    """ Test behaviour if no gr value at non-baseline forming month

    """
    test_data = WITH_BASELINE.loc[WITH_BASELINE.mouse_id == 'M6']
    
    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    print(norm_data_df[['day', 'norm_percent_engraftment', 'cell_type']])
    lineage_bias_df = lb.create_lineage_bias_df(norm_data_df, lymphoid_cell_type='b', myeloid_cell_type='gr')
    with_exp = lineage_bias_df.merge(EXPECTED_LINEAGE_BIAS)
    print(lineage_bias_df[['myeloid_percent_abundance', 'lymphoid_percent_abundance', 'lineage_bias']])

    assert lineage_bias_df.shape[0] == with_exp.shape[0]
    assert (lineage_bias_df.shape[1] + 1) == with_exp.shape[1]
    assert_array_almost_equal(
        lineage_bias_df.lineage_bias,
        with_exp.lineage_bias_expected,
    )

def test_bias_angle():
    """ test calculated lineage bias angle is as expected from known scenarios
    """
    assert lb.calc_angle(1, 1) == pi/4
    assert lb.calc_angle(0, 0) == pi/4
    assert lb.calc_angle(1, 0) == pi/2
    assert lb.calc_angle(0, 1) == 0

def test_angle_to_bias():
    """ Test certain angles lead to expected lineage biases
    """
    balanced_angle_min = pi/8
    balanced_value_min = sin(2 * (balanced_angle_min - (pi/4)))
    balanced_angle_max = 3*pi/8
    balanced_value_max = sin(2 * (balanced_angle_max - (pi/4)))
    balanced_angle = pi/4
    
    assert lb.calc_bias_from_angle(balanced_angle) == 0
    assert lb.calc_bias_from_angle(balanced_angle_min) == balanced_value_min
    assert lb.calc_bias_from_angle(balanced_angle_max) == balanced_value_max
    assert balanced_value_max == balanced_value_min * -1

def test_filter_by_ratio_0thresh():
    test_data = WITH_BASELINE[WITH_BASELINE.mouse_id != 'M2']
    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    lineage_bias_df = lb.create_lineage_bias_df(norm_data_df, lymphoid_cell_type='b', myeloid_cell_type='gr')
    filt_df = agg.filter_lineage_bias_cell_type_ratio_per_mouse(
        lineage_bias_df,
        'day',
        TEST_WBC,
        filter_threshold=0,    
        myeloid_cell='gr',
        lymphoid_cell='b'
    )
    assert_frame_equal(
        lineage_bias_df,
        filt_df.drop(columns=['m_filter', 'l_filter'])
    )

def test_filter_by_ratio_exact_thresh():
    """ 1 row should pass, since B needs to be >= 1
    """
    test_data = WITH_BASELINE[WITH_BASELINE.mouse_id == 'M6']
    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    lineage_bias_df = lb.create_lineage_bias_df(norm_data_df, lymphoid_cell_type='b', myeloid_cell_type='gr')
    filt_df = agg.filter_lineage_bias_cell_type_ratio_per_mouse(
        lineage_bias_df,
        'day',
        TEST_WBC,
        filter_threshold=1,
        myeloid_cell='gr',
        lymphoid_cell='b'
    )
    print(filt_df)
    # Just first entry should pass filter
    assert_series_equal(
        lineage_bias_df.iloc[0],
        filt_df.drop(columns=['m_filter', 'l_filter']).iloc[0]
    )
    assert lineage_bias_df.shape[0] == 2
    assert filt_df.shape[0] == 1

def test_filter_by_ratio_too_high_thresh():
    test_data = WITH_BASELINE[WITH_BASELINE.mouse_id == 'M6']
    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    lineage_bias_df = lb.create_lineage_bias_df(norm_data_df, lymphoid_cell_type='b', myeloid_cell_type='gr')
    filt_df = agg.filter_lineage_bias_cell_type_ratio_per_mouse(
        lineage_bias_df,
        'day',
        TEST_WBC,
        filter_threshold=1.0001,
        myeloid_cell='gr',
        lymphoid_cell='b'
    )
    assert filt_df.empty

def test_lineage_bias_pipeline():
    args = Namespace(
        input='test/test_data/test_long_for_lineage_bias.csv',
        lymphoid_cell_type='b',
        myeloid_cell_type='gr',
        counts_file='test/test_data/test_WBC.txt',
        baseline_timepoint='by_mouse',
        output_dir='test/test_data/validate'
    )
    lb.main(args)
    pipeline_out = pd.read_csv('test/test_data/validate/lineage_bias.csv')


    test_data = WITH_BASELINE
    norm_data_df = lb.normalize_to_baseline_counts(test_data)
    lineage_bias_df = lb.create_lineage_bias_df(norm_data_df, lymphoid_cell_type='b', myeloid_cell_type='gr')

    assert_frame_equal(pipeline_out, lineage_bias_df, check_dtype=False)
