""" Test if 0s are being correctly filled in data
"""
import sys
import os
import pandas as pd
from pandas.testing import assert_frame_equal
sys.path.append(os.getcwd())
import aggregate_functions as agg

miss_m4_at_t2 = pd.DataFrame.from_dict({
    'mouse_id': ['1', '2', '3'] * 2 + ['4'],
    'group': ['group'] * 7,
    'val': [100] * 7,
    'time': [1]*3 + [2] * 4,
})
filled_data = pd.DataFrame.from_dict({
    'mouse_id': ['1', '2', '3', '4'] * 2,
    'group': ['group'] * 8,
    'val': [100] * 3 + [0] + [100] * 4,
    'time': [1]*4 + [2] * 4,
})
m5_data = pd.DataFrame.from_dict({
    'mouse_id': ['5'] * 2,
    'group': ['group'] * 2,
    'val': [0] * 2,
    'time': [1, 2],
})

def test_miss_mouse_1tp():
    result = agg.fill_mouse_id_zeroes(
        miss_m4_at_t2,
        info_cols=['group'],
        fill_col='val',
        fill_cat_col='time',
        fill_cats=[1, 2],
        fill_val=0
    )
    assert_frame_equal(
        result.sort_values(by=['mouse_id', 'time'])
        .reset_index(drop=True),
        filled_data.sort_values(by=['mouse_id', 'time'])
        .reset_index(drop=True),
    )


def test_unchange_filled():
    result = agg.fill_mouse_id_zeroes(
        filled_data,
        info_cols=['group'],
        fill_col='val',
        fill_cat_col='time',
        fill_cats=[1, 2],
        fill_val=0
    )
    assert_frame_equal(
        result.sort_values(by=['mouse_id', 'time']).reset_index(drop=True),
        filled_data.sort_values(by=['mouse_id', 'time']).reset_index(drop=True),
    )

def test_fill_mouse_list():
    result = agg.fill_mouse_id_zeroes(
        filled_data,
        info_cols=['group'],
        fill_col='val',
        fill_cat_col='time',
        fill_cats=[1, 2],
        fill_val=0,
        mice_to_fill=['1', '2', '3', '4', '5']
    )
    with_5_validate = pd.concat([filled_data, m5_data])
    assert with_5_validate.mouse_id.nunique() == 5
    assert_frame_equal(
        result.sort_values(by=['mouse_id', 'time'])
        .drop(columns='group').reset_index(drop=True),
        with_5_validate.sort_values(by=['mouse_id', 'time'])
        .drop(columns='group').reset_index(drop=True),
    )
    assert result[result.mouse_id == '5'].group.isna().all()
