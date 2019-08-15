import os
from typing import List
import pandas as pd
import numpy as np
import aggregate_functions as agg


root_dir = '/home/sakre/Data/aging_and_10x'
data = pd.read_csv(root_dir + '/Ania_M_all OT2.0 rerun_percent-engraftment_NO filter_080819_long.csv')
timepoint_col = 'month'
data['month'] = agg.day_to_month(data['day'])

p_anywhere_filter = 0.01
change_p = 0.05

data = agg.remove_month_17_and_6(
    data,
    timepoint_col,
)
data = agg.filter_mice_with_n_timepoints(
    data,
    n_timepoints=4
)
onep_anywhere = data[data.percent_engraftment > p_anywhere_filter]
onep_anywhere = onep_anywhere[['code', 'mouse_id']].drop_duplicates()
filt_data = data.merge(
    onep_anywhere,
    how='inner',
    on=['code', 'mouse_id'],
    validate='m:1'
)
for cell_type in ['gr', 'b']:
    for group in ['aging_phenotype', 'no_change']:
        group_data = filt_data[filt_data.group == group]
        cell_data = group_data[group_data.cell_type == cell_type]

        ## Make change relative to first >= change_p in abundance
        first_tp = cell_data[cell_data.day == cell_data['day'].min()].rename(columns={'percent_engraftment': 'first_abund'})[['code', 'mouse_id', 'first_abund']]
        not_first_tp = cell_data[cell_data.day != cell_data['day'].min()]
        changes_codes: List[str] = []
        for time, tp_df in not_first_tp.groupby('day'):
            t_diff_df = first_tp.merge(
                tp_df,
                how='inner',
                validate='1:1'
            )
            pass_filt = t_diff_df[np.abs(t_diff_df.percent_engraftment - t_diff_df.first_abund) >= change_p]
            changes_codes = changes_codes + pass_filt.code.unique().tolist()

        filt_change_df = cell_data[cell_data.code.isin(changes_codes)]



        piv = filt_change_df.pivot_table(
            index=['code'],
            columns=[timepoint_col],
            values='percent_engraftment'
        )
        save_dir = \
            root_dir \
            + '/STEM_Clustering/' \
            + 'data_filt_' \
            + str(p_anywhere_filter).replace('.', '-') \
            + '_change_' \
            + str(change_p).replace('.', '-') 

        if not os.path.exists(save_dir):
            print('Creating', save_dir)
            os.makedirs(save_dir)

        file_name = save_dir \
            + '/' \
            + cell_type \
            + '_' + group \
            + '_data_filt_' \
            + str(p_anywhere_filter).replace('.', '-') \
            + '_change_' \
            + str(change_p).replace('.', '-') \
            + '.tsv'
        print('Saving File:', file_name)
        piv.to_csv(
            file_name,
            sep='\t'
            )
