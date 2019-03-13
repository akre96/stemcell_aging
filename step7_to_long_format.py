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
    out_columns: typing.List[str] = ['code', 'user', 'mouse_id', 'day', 'cell_type', 'percent_engraftment']
    long_df: pd.DataFrame = pd.DataFrame(columns=out_columns)
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
    out_columns: typing.List[str] = ['code', 'user', 'mouse_id', 'day', 'cell_type', 'percent_engraftment']
    step7_df = pd.read_csv(step7_output, sep='\t')
    step7_long_df = pd.DataFrame(columns=out_columns)

    for index_row_tuple in step7_df.iterrows():
        row_long = transform_row_wide_to_long(index_row_tuple[1])
        step7_long_df = step7_long_df.append(row_long, ignore_index=True)

    return step7_long_df




def main():
    input_file_path = "/mnt/d/data/aging_proc_03122019/Ania_M3000_percent-engraftment_100818.txt"
    results = step7_out_to_long_format(input_file_path)
    print(results)

if __name__ == "__main__":
    main()
