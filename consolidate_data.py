""" Creates single file from long format output data of step_7_long_format.py

Example usage:
python consolidate_data.py -i /home/sakre/Data/aging_proc_03122019_long -o output -g /home/sakre/Data/mouse_id_group.csv

Real Example:
python consolidate_data.py -i /home/sakre/Data/stemcell_aging/aging_proc_03122019_long -o ~/Data/stemcell_aging/aging_proc_03122019_long_consolidated -g ~/Data/stemcell_aging/mouse_id_group.csv

"""

import os
import sys
import glob
import re
import typing
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Consolidate all long format transformed data")
    parser.add_argument('-i', '--input', dest='input', help='Path to folder containing long formatted step7 outputs', required=True)
    parser.add_argument('-g', '--groups', dest='groups', help='Path to csv file with mouse phenotype data', required=False)
    parser.add_argument('-o', '--output-dir', dest='output_dir', help='Directory to send output files to', default='')

    args = parser.parse_args()

    input_files: typing.List[str] = glob.glob(args.input + os.sep + "*_long.csv")
    if args.groups:
        group_info_df = pd.read_csv(args.groups)
    if not os.path.isdir(args.output_dir):
        print('Output directory: ' + args.out_dir + ' does not exist')
        sys.exit(1)

    if input_files:
        print('Processing Files: ')
        all_data_df = pd.DataFrame()
        for mouse_data_file in input_files:
            print(mouse_data_file)
            mouse_data_df = pd.read_csv(mouse_data_file)
            all_data_df = all_data_df.append(mouse_data_df, ignore_index=True)

        if args.groups:
            appended_df = pd.merge(all_data_df, group_info_df, how='left', on=['mouse_id'])
        else:
            appended_df = all_data_df
    else:
        print('No Input Files Found at: ' + args.input)
        sys.exit(1)
    print('Done.')

    first_file_title = os.path.splitext(os.path.basename(input_files[0]))[0]
    outfile_name = re.sub(r'M\d{1,4}', 'M_all', first_file_title)

    print('Saving Output To: ')
    print(args.output_dir + os.sep + outfile_name + '.csv')

    appended_df.to_csv(args.output_dir + os.sep + outfile_name + '.csv', index=False)

    print('Done.')
    



if __name__ == "__main__":
    main()
