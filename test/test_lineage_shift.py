""" Test the classification of lineage stable vs shifting
"""
import sys
import os
import pandas as pd
from numpy.testing import assert_array_equal
sys.path.append(os.getcwd())
import aggregate_functions as agg

lymphoid_shifting_clone = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 3,
    'month': [4, 15, 17],
    'group': ['E-MOLD'] * 3,
    'lineage_bias': [1, -1, -1],
    'code': ['lymphoid_shifting'] * 3
})
myeloid_shifting_clone = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 3,
    'month': [4, 15, 17],
    'group': ['E-MOLD'] * 3,
    'lineage_bias': [-1, 1, 1],
    'code': ['myeloid_shifting'] * 3
})
ideal_lineage_stable_clone = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 3,
    'month': [4, 15, 17],
    'group': ['E-MOLD'] * 3,
    'lineage_bias': [0, 0, 1],
    'code': ['lineage_stable_0'] * 3
})

lineage_stable_clone = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 3,
    'month': [4, 15, 17],
    'group': ['E-MOLD'] * 3,
    'lineage_bias': [0, 0.01, 1],
    'code': ['lineage_stable_01'] * 3
})



# Test clones for full bias change cutoff
n_stable = 90
n_myeloid = 5
n_lymphoid = 5
tests = [
    [ideal_lineage_stable_clone] * n_stable +
    [myeloid_shifting_clone] * n_myeloid +
    [lymphoid_shifting_clone] * n_lymphoid
]
i = 0
tests = tests[0]
ts = []
for df in tests:
    df = df.copy()
    df['type'] = df['code']
    df['code'] = df['code'].astype(str) + '-' + str(i)
    ts.append(df)
    i += 1
test_df = pd.concat(ts)

lymphoid_shift_abundance = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 2,
    'month': [15] * 2,
    'group': ['E-MOLD'] * 2,
    'percent_engraftment': [.1, .2],
    'cell_type': ['gr', 'b'],
    'code': ['lymphoid_shifting-' + str(n_stable + n_myeloid)] * 2
})
myeloid_shift_abundance = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 2,
    'month': [15] * 2,
    'group': ['E-MOLD'] * 2,
    'percent_engraftment': [.2, .3],
    'cell_type': ['gr', 'b'],
    'code': ['myeloid_shifting-' + str(n_stable)] * 2
})
ideal_lineage_stable_abundance = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 2,
    'month': [15] * 2,
    'group': ['E-MOLD'] * 2,
    'percent_engraftment': [.7, .5],
    'cell_type': ['gr', 'b'],
    'code': ['lineage_stable_0-0'] * 2
})

def test_calculate_lymphoid_bias_change():
    result = agg.calculate_first_last_bias_change(
        lymphoid_shifting_clone,
        'month',
        True,
        True
    )
    print(result)
    assert result.code.nunique() == 1
    assert result.time_change.unique()[0] == 11
    assert result.bias_change.unique()[0] == -2

def test_calculate_myeloid_bias_change():
    result = agg.calculate_first_last_bias_change(
        myeloid_shifting_clone,
        'month',
        True,
        True
    )
    print(result)
    assert result.code.nunique() == 1
    assert result.time_change.unique()[0] == 11
    assert result.bias_change.unique()[0] == 2

def test_calculate_linstable_change():
    result = agg.calculate_first_last_bias_change(
        ideal_lineage_stable_clone,
        'month',
        True,
        True
    )
    print(result)
    assert result.code.nunique() == 1
    assert result.time_change.unique()[0] == 11
    assert result.bias_change.unique()[0] == 0

def test_bias_change_cutoff():
    bias_change_df = agg.calculate_first_last_bias_change(
        test_df,
        'month',
        True,
        True
    )

    # One entry per unique clone
    assert bias_change_df.shape[0] == (n_lymphoid + n_lymphoid + n_stable)
    x, y, y1, y2, x_c, y_c, kde = agg.calculate_bias_change_cutoff(
        bias_change_df,
        min_time_difference=3,
        timepoint=None,
    )

    # Outputs same kde
    assert_array_equal(kde.pdf(x), y)

    # cutoff should be ~0.5
    assert round(x_c[0], 1) == 0.5


def test_mark_change():
    bias_change_df = agg.calculate_first_last_bias_change(
        test_df,
        'month',
        True,
        True
    )
    result = agg.mark_changed(
        test_df,
        bias_change_df,
        3,
    )
    change_status = result.groupby('change_status').code.nunique()
    assert change_status['Unchanged'] == n_stable
    assert change_status['Changed'] == (n_myeloid + n_lymphoid)

    change_type = result.groupby('change_type').code.nunique()
    assert change_type['Unchanged'] == n_stable
    assert change_type['Lymphoid'] == n_lymphoid
    assert change_type['Myeloid'] == n_myeloid

def test_changed_contribution():
    abundance_data = pd.concat([
        myeloid_shift_abundance,
        lymphoid_shift_abundance,
        ideal_lineage_stable_abundance,
    ])
    bias_change_df = agg.calculate_first_last_bias_change(
        test_df,
        'month',
        True,
        True
    )
    change_marked = agg.mark_changed(
        abundance_data,
        bias_change_df,
        3,
    )
    result = agg.sum_abundance_by_change(
        change_marked,
        percent_of_total=True,
        timepoint_col='month'
    )

    print(result)
    # Check that each cell type adds to 100%
    assert result.groupby('cell_type').percent_engraftment.sum().tolist() == [100, 100]

    assert result[
        (result.change_type == 'Myeloid') &
        (result.cell_type == 'gr')
    ]['percent_engraftment'].values[0] == 20

    assert result[
        (result.change_type == 'Lymphoid') &
        (result.cell_type == 'gr')
    ]['percent_engraftment'].values[0] == 10

    assert result[
        (result.change_type == 'Unchanged') &
        (result.cell_type == 'gr')
    ]['percent_engraftment'].values[0] == 70
