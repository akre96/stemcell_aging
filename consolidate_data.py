import os
import glob
import re
import typing
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Consolidate all long format transformed data")
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing long formatted step7 outputs', required=True)
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='')

    args = parser.parse_args()

    input_files: typing.List[str] = glob.glob(args.input + os.sep + "*_long.csv")

    if len(input_files):
        all_data_df = pd.DataFrame()
        for mouse_data_file in input_files:
            mouse_data_df = pd.read_csv(mouse_data_file)
            all_data_df = all_data_df.append(mouse_data_df, ignore_index=True)
    first_file_title = os.path.splitext(os.path.basename(input_files[0]))[0]
    outfile_name = re.sub(r'M\d{1,4}', 'M_all', first_file_title)
    all_data_df.to_csv(args.output_dir + outfile_name + '.csv', index=False)
    


if __name__ == "__main__":
    main()
