from typing import List
import re
import pandas as pd

def parse_wbc_count_file(wbc_count_file_path: str, analyzed_cell_types: List[str] = ['gr', 'b'], sep: str = '\t', data_type: str = 'cell_count') -> pd.DataFrame:
    """ Parses white blood cell count file to format as dataframe

    Arguments:
        wbc_count_file_path {str} -- path to count file

    Keyword Arguments:
        analyzed_cell_types {List[str]} -- cell types to parse for (default: {['gr', 'b']})
        sep {str} -- String used to seperate cells in file

    Returns:
        pd.DataFrame -- dataframe of mouse_id, cell_type, day, cell_count
    """

    count_data_raw = pd.read_csv(wbc_count_file_path, sep=sep)
    parsed_counts = pd.DataFrame()
    col_names = count_data_raw.columns
    end_cols = [i for i, x in enumerate(col_names.tolist()) if x.find('Unnamed') != -1]
    end_cols.append(len(col_names))
    for i, end_col_index in enumerate(end_cols):
        parsed_timepoint_data: pd.DataFrame = pd.DataFrame()
        if i == 0:
            start_index = 0
        else:
            start_index = end_cols[i-1] + 1
        one_timepoint_data = count_data_raw[col_names[start_index:end_col_index]]
        one_timepoint_cols = one_timepoint_data.columns
        if one_timepoint_cols.empty:
            continue
        day = int(one_timepoint_cols[0][1:])
        month = int(round(day/30))

        for cell_type in analyzed_cell_types:
            cell_type_col_name = [re.match(cell_type.lower(), x.lower()) is not None for x in one_timepoint_cols]

            # Check if cell exists at time point
            if not any(cell_type_col_name):
                continue

            cell_type_timepoint_data = pd.DataFrame(columns=['mouse_id','day','month','cell_type',data_type])
            cell_type_timepoint_data['mouse_id'] = one_timepoint_data[one_timepoint_cols[0]].str.replace(" ", "")
            cell_type_timepoint_data['day'] = day
            cell_type_timepoint_data['month'] = month
            cell_type_timepoint_data['cell_type'] = cell_type.lower()
            cell_type_col = one_timepoint_cols[cell_type_col_name]
            cell_type_timepoint_data[data_type] = one_timepoint_data[cell_type_col]
            parsed_timepoint_data = parsed_timepoint_data.append(cell_type_timepoint_data, ignore_index=True)

        if parsed_timepoint_data.empty:
            continue
        no_nan_mouse_ids = parsed_timepoint_data[~parsed_timepoint_data.mouse_id.isnull()]
        parsed_counts = parsed_counts.append(no_nan_mouse_ids, ignore_index=True)

    return parsed_counts
