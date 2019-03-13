"""
    Tests multi_index_data functions
"""
import os
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy.testing as npt
from ..step7_to_long_format import parse_column_metadata, transform_row_wide_to_long

# Necessary to specify path this way for tests to run across different platforms
TEST_DATA_LOCATION = os.getcwd() + os.sep + 'test' + os.sep + 'test_data' + os.sep 
TEST_STEP7_DF = pd.read_csv(TEST_DATA_LOCATION + 'Ania_M3000_percent-engraftment_100818.txt', sep='\t')

def test_parse_column_metadata():
    test_input = 'Ania_M3000D122Cgr'
    expected_output = {
        'user': 'Ania',
        'mouse_id': 'M3000',
        'day': 122,
        'cell_type': 'gr',
    }
    assert parse_column_metadata(test_input) == expected_output

def test_transform_row_wide_to_long():
    test_row = TEST_STEP7_DF.loc[1]
    expanded_row = transform_row_wide_to_long(test_row)
    expected_output = pd.read_csv(TEST_DATA_LOCATION + 'test_row_to_long.csv')
    print(expected_output)
    print(expanded_row)

    assert len(expanded_row) == (len(TEST_STEP7_DF.columns) - 1)
    for col in expected_output.columns:
        print(col)
        if col in ['percent_engraftment']:
            npt.assert_allclose(expanded_row[col], expected_output[col], atol=1e-10)
        else:
            npt.assert_array_equal(expanded_row[col], expected_output[col])
