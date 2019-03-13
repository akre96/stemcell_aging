"""
Initial exploration of step 7 processed data.
"""
import typing
import re
import pandas as pd


def parse_column_metadata(col_name: str) -> typing.Dict[str, str]:
    """
    Extracts user, mouse_id, day, and cell_type metadata from column string

    :param col_name: The column name
    :returns: Dictionary with user, mouse_id, day, cell_type keys
    """
    match_pattern: str = r'(\w+)_(M\d+)D(\d+)C(\w+)' # Finds [user]_[mouse_id]D[days]C[cell_type]
    matches: re.match = re.match(match_pattern, col_name)

    if len(matches.groups()) <= 3:
        print("Error: Did not match all groups")
        print("Matched string: " + matches.group(0))
        print(matches.groups())

    return {
        'user': matches.group(1),
        'mouse_id': matches.group(2),
        'day': int(matches.group(3)),
        'cell_type': matches.group(4),
    }

def transform_row_wide_to_long(row: pd.DataFrame) -> pd.DataFrame:
    """
    Turns one wide format entry from step 7 to long format
    Assumes the row has first column 'code' and following columns in format
    '[user]_[mouse_id]D[day]C[cell_type]'

    :param row: row from step 7 processed data output
    :returns: data frame expanded to long format
    """
    long_df: pd.DataFrame = pd.DataFrame(columns=['code', 'user', 'mouse_id', 'day', 'cell_type', 'percent_engraftment'])
    code: str = row['code']
    for col in row.index[1:]:
        new_row: typing.Dict = parse_column_metadata(col)
        new_row['code'] = code
        new_row['percent_engraftment'] = float(row[col])
        long_df = long_df.append(new_row, ignore_index=True)
    return long_df

def step7_out_to_long_format(step7_output: str) -> pd.DataFrame:
    """
    Takes the output of step 7 and transforms it in to long format with for
    code, user, mouse_id, day, cell_type, and percent_engraftment

    :param step7_output: path to the output from step7 to be transformed
    :returns: data frame of data from step7 in long format
    """

def main():
    input_file_path = "/mnt/d/data/aging_proc_03122019/Ania_M3000_percent-engraftment_100818.txt"
    input_df = pd.read_csv(input_file_path, sep='\t')
    print(parse_column_metadata(input_df.columns[2]))
    transform_row_wide_to_long(input_df.loc[1]).to_csv('test/test_data/test_row_to_long.csv',index=False)

if __name__ == "__main__":
    main()
