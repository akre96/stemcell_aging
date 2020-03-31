
""" Test calculations for clonal exhastions work as expected
for serial transplant data

For serial transplant, a clone is exhausted if it exists in a time point
before generation 7 but NOT in generation 7 or 8. A clone is persistetn
if it is present at both generation 7 AND 8. Clones present in just 7 or
just 8 are not considered.
"""

import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
import aggregate_functions as agg


persistent_clone = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 4,
    'gen': [7, 8] * 2,
    'group': ['E-MOLD'] * 4,
    'percent_engraftment': [0.011] * 4,
    'cell_type': ['gr'] * 2 + ['b'] * 2,
    'code': ['persistent_clone'] * 4
})

exhausted_by_presence_thresh = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 4,
    'gen': [6, 7] * 2,
    'group': ['E-MOLD'] * 4,
    'percent_engraftment': [0.011, 0.009] * 2,
    'cell_type': ['gr'] * 2 + ['b'] * 2,
    'code': ['exhausted'] * 4
})

no_label_by_gen_7_not_8 = pd.DataFrame.from_dict({
    'mouse_id': ['test'] * 4,
    'gen': [7, 8] * 2,
    'group': ['E-MOLD'] * 4,
    'percent_engraftment': [0.011, 0.009] * 2,
    'cell_type': ['gr'] * 2 + ['b'] * 2,
    'code': ['no_label'] * 4
})

def test_finds_persistent_clone():
    result = agg.exhausted_clones_serial_transplant(
        persistent_clone,
        'gen',
    )
    persisting = result[result.survived == 'Survived']
    assert persisting.code.nunique() == 1
    assert result[result.survived != 'Survived'].empty

def test_exhausted_by_presence_thresh():
    result = agg.exhausted_clones_serial_transplant(
        exhausted_by_presence_thresh,
        'gen',
    )
    exhaust = result[result.survived == 'Exhausted']
    assert exhaust.code.nunique() == 1
    assert result[result.survived != 'Exhausted'].empty


def test_no_label_gen_7_not_8():
    result = agg.exhausted_clones_serial_transplant(
        no_label_by_gen_7_not_8,
        'gen',
    )
    assert result.empty

def test_return_both_exhaust_types():
    """ Test that exhaust and persistent clones labeled,
    data returned only for time points passing presence threshold
    """
    result = agg.exhausted_clones_serial_transplant(
        pd.concat([
            persistent_clone,
            exhausted_by_presence_thresh,
            no_label_by_gen_7_not_8
        ]),
        'gen'
    )
    assert result.shape[0] == 6
    assert result.code.nunique() == 2

def test_label_return_both_exhaust():
    result = agg.label_exhausted_clones(
        None,
        pd.concat([
            persistent_clone,
            exhausted_by_presence_thresh,
            no_label_by_gen_7_not_8
        ]),
        'gen'
    )
    assert result.shape[0] == 6
    assert result.code.nunique() == 2
