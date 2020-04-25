""" Test functions related to retrieving specific clones
"""
import pandas as pd
from pandas.testing import assert_frame_equal
import sys ,os
sys.path.append(os.getcwd())
import aggregate_functions as agg

most_abund_clone_first_1 = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 8,
    'month': [4, 9, 12, 15] * 2,
    'group': ['E-MOLD'] * 8,
    'percent_engraftment': ([100] + [0.011] * 3) * 2,
    'cell_type': ['gr'] * 4 + ['b'] * 4,
    'code': ['abund_clone_first_1'] * 8
})
most_abund_clone_first_2 = pd.DataFrame.from_dict({
    'mouse_id': ['test_2'] * 8,
    'month': [4, 9, 12, 15] * 2,
    'group': ['E-MOLD'] * 8,
    'percent_engraftment': ([100] + [0.011] * 3) * 2,
    'cell_type': ['gr'] * 4 + ['b'] * 4,
    'code': ['abund_clone_first_2'] * 8
})
most_abund_clone_last_1 = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 8,
    'month': [4, 9, 12, 15] * 2,
    'group': ['E-MOLD'] * 8,
    'percent_engraftment': ([0.011] * 3 + [100]) * 2,
    'cell_type': ['gr'] * 4 + ['b'] * 4,
    'code': ['abund_clone_last_1'] * 8
})
most_abund_clone_last_2 = pd.DataFrame.from_dict({
    'mouse_id': ['test_2'] * 8,
    'month': [4, 9, 12, 15] * 2,
    'group': ['E-MOLD'] * 8,
    'percent_engraftment': ([0.011] * 3 + [100]) * 2,
    'cell_type': ['gr'] * 4 + ['b'] * 4,
    'code': ['abund_clone_last_2'] * 8
})

def test_get_n_most_abundant_at_first():
    test_data = pd.concat([
        most_abund_clone_first_1,
        most_abund_clone_last_1,
    ])
    result = agg.get_n_most_abundant_at_time(
        test_data,
        n=1,
        timepoint_col='month',
        timepoint='first',
        by_mouse=True
    )
    assert result.code.nunique() == 1
    assert_frame_equal(
        result,
        most_abund_clone_first_1
    )

    result = agg.get_n_most_abundant_at_time(
        test_data,
        n=2,
        timepoint_col='month',
        timepoint='first',
        by_mouse=True
    )
    assert result.code.nunique() == 2
    assert_frame_equal(
        result,
        test_data.reset_index(drop=True)
    )


def test_get_n_most_abundant_at_last():
    test_data = pd.concat([
        most_abund_clone_first_1,
        most_abund_clone_last_1,
    ])
    result = agg.get_n_most_abundant_at_time(
        test_data,
        n=1,
        timepoint_col='month',
        timepoint='last',
        by_mouse=True
    )
    print(result)
    assert result.code.nunique() == 1
    assert_frame_equal(
        result,
        most_abund_clone_last_1
    )

    result = agg.get_n_most_abundant_at_time(
        test_data,
        n=2,
        timepoint_col='month',
        timepoint='last',
        by_mouse=True
    )
    assert result.code.nunique() == 2
    assert_frame_equal(
        result,
        test_data.reset_index(drop=True)
    )

def test_get_n_most_abundant_by_mouse():
    test_data = pd.concat([
        most_abund_clone_first_1,
        most_abund_clone_first_2,
        most_abund_clone_last_1,
        most_abund_clone_last_2,
    ])
    result = agg.get_n_most_abundant_at_time(
        test_data,
        n=1,
        timepoint_col='month',
        timepoint='last',
        by_mouse=True
    )
    print(result)
    assert result.code.nunique() == 2
    assert result.mouse_id.nunique() == 2
    assert_frame_equal(
        result,
        pd.concat([
            most_abund_clone_last_1,
            most_abund_clone_last_2
        ]).reset_index(drop=True)
    )

    result = agg.get_n_most_abundant_at_time(
        test_data,
        n=2,
        timepoint_col='month',
        timepoint='last',
        by_mouse=True
    )
    assert result.code.nunique() == 4
    assert_frame_equal(
        result,
        test_data.reset_index(drop=True)
    )
