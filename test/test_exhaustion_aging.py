""" Test calculations for clonal exhastions work as expected for aging over time

Exhausted clones exist >= 0.01 at M4 and M9, but NOT in last time point for a
mouse. This means a clone must have its last time point before the mouse
stopped existing to be counted as exhausted. Persistent clones exist at M4, 9
and the last timepoint for a mouse
"""

import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
import aggregate_functions as agg

persistent_clone = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 8,
    'month': [4, 9, 12, 15] * 2,
    'group': ['E-MOLD'] * 8,
    'percent_engraftment': [0.011] * 8,
    'cell_type': ['gr'] * 4 + ['b'] * 4,
    'code': ['persistent_clone'] * 8
})

persistent_clone_m15 = pd.DataFrame.from_dict({
    'mouse_id': ['test_last15'] * 6,
    'month': [4, 9, 15] * 2,
    'group': ['E-MOLD'] * 6,
    'percent_engraftment': [0.011] * 6,
    'cell_type': ['gr'] * 3 + ['b'] * 3,
    'code': ['persistent_clone_15'] * 6
})
persistent_clone_m12 = pd.DataFrame.from_dict({
    'mouse_id': ['test_last12'] * 6,
    'month': [4, 9, 12] * 2,
    'group': ['E-MOLD'] * 6,
    'percent_engraftment': [0.011] * 6,
    'cell_type': ['gr'] * 3 + ['b'] * 3,
    'code': ['persistent_clone_12'] * 6
})

exhaust_clone_by_presence_thresh = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 8,
    'month': [4, 9, 12, 15] * 2,
    'group': ['E-MOLD'] * 8,
    'percent_engraftment': [0.01, 0.01, 0.009, 0.009] * 2,
    'cell_type': ['gr'] * 4 + ['b'] * 4,
    'code': ['exhausted_clone'] * 8
})

no_label_MPP = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 8,
    'month': [4, 9, 12, 15] * 2,
    'group': ['E-MOLD'] * 8,
    'percent_engraftment': [0.01, 0.0, 0.01, 0.01] * 2,
    'cell_type': ['gr'] * 4 + ['b'] * 4,
    'code': ['mpp'] * 8
})

def test_finds_persistent_clone():
    result = agg.exhausted_clones_without_MPPs(
        persistent_clone,
        'month',
    )
    persisting = result[result.survived == 'Survived']
    assert persisting.code.nunique() == 1
    assert result[result.survived != 'Survived'].empty


def test_finds_persistent_clone_m15():
    result = agg.exhausted_clones_without_MPPs(
        persistent_clone_m15,
        'month',
    )
    persisting = result[result.survived == 'Survived']
    assert persisting.code.nunique() == 1
    assert result[result.survived != 'Survived'].empty

def test_finds_persistent_clone_m12():
    result = agg.exhausted_clones_without_MPPs(
        persistent_clone_m12,
        'month',
    )
    persisting = result[result.survived == 'Survived']
    assert persisting.code.nunique() == 1
    assert result[result.survived != 'Survived'].empty

def test_not_present_clone_exhausted():
    result = agg.exhausted_clones_without_MPPs(
        exhaust_clone_by_presence_thresh,
        'month',
    )
    exhaust = result[result.survived == 'Exhausted']
    assert exhaust.code.nunique() == 1
    assert result[result.survived != 'Exhausted'].empty

def test_mpp_not_added():
    result = agg.exhausted_clones_without_MPPs(
        no_label_MPP,
        'month',
    )
    assert result.empty

def test_label_return_both_exhaust():
    result = agg.label_exhausted_clones(
        None,
        pd.concat([
            persistent_clone,
            persistent_clone_m12,
            exhaust_clone_by_presence_thresh,
            no_label_MPP
        ]),
        'month'
    )
    assert result.shape[0] == 18
    assert result.code.nunique() == 3
