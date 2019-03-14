"""
Takes output of step 7 and converts to long format for easier data manipulation
Saves output to specified output_directory with same input file name + '_long'

Example Usage:
python step7_to_long_format.py -i test/test_data -o output122019 -o output

"""
import typing
import sys
import os
import glob
import multiprocessing as mp
import re
import argparse
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
    out_columns: typing.List[str] = ['mouse_id', 'code', 'user', 'day', 'cell_type', 'percent_engraftment']
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
    out_columns: typing.List[str] = ['mouse_id', 'code', 'user', 'day', 'cell_type', 'percent_engraftment']
    step7_df = pd.read_csv(step7_output, sep='\t')
    step7_long_df = pd.DataFrame(columns=out_columns)

    for index_row_tuple in step7_df.iterrows():
        row_long = transform_row_wide_to_long(index_row_tuple[1])
        step7_long_df = step7_long_df.append(row_long, ignore_index=True)

    return step7_long_df

def step7_out_to_long_format_write(inputs: typing.Tuple)  -> None:
    """
    runs step7_out_to_long_format such that it writes out csv files

    :param inputs: tuple of the input file name and output file dir
    """
    step7_output, write_dir = inputs

    print('Transforming file: ' + step7_output)
    step7_long_df = step7_out_to_long_format(step7_output)
    step7_output_title = os.path.splitext(os.path.basename(step7_output))[0]
    step7_long_df.to_csv(write_dir + os.sep + step7_output_title + '_long.csv', index=False)
    print('Written to: ' + write_dir + os.sep + step7_output_title + '_long.csv')




def main():
    """ Transforms inputted files from wide to long format
    Parses input arguments and finds either a single file or entire folder.
    Input is then transformed from wide to long format and saved to specified output directory.

    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transform input files from wide to long format")
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing only data to be formatted or a single file to format', required=True)
    parser.add_argument('-p', '--prefix', dest='prefix', help='Optional prefix to filter by when finding input files from folder', default=False)
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='.')

    args = parser.parse_args()

    # If directory supplied, find all inputs
    if os.path.isdir(args.input):
        print('Inputs being parsed from folder: ' + args.input)

        # Filters by prefix if supplied
        if args.prefix:
            input_files: typing.List[str] = glob.glob(args.input + os.sep + args.prefix + "*.txt")
        else:
            input_files: typing.List[str] = glob.glob(args.input + os.sep + "*.txt")

        # Inputs zipped to allow parallelization
        input_tuple = zip(input_files, [args.output_dir for i in range(len(input_files))])

        print('Found the following input files: ')
        for file_name in input_files:
            print(file_name)

        print('Beginning to transform input files from wide to long format')
        pool = mp.Pool()
        pool.map(step7_out_to_long_format_write, input_tuple)
        print('Done.')

    # If single file, runs just once
    elif os.path.isfile(args.input):
        print('Input File: ' + args.input)
        print('Beginning to transform input files from wide to long format')
        step7_out_to_long_format_write((args.output_dir, args.input))
        print('Done.')

    else:
        print('Error: Input file or folder not found')
        sys.exit(1)



if __name__ == "__main__":
    main()
