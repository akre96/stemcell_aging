import pandas as pd
root_dir = '~/Data/serial_transplant_data'
data = pd.read_csv('~/Data/serial_transplant_data/M_allAniaAnia serial transpl_percent-engraftment_121018_long.csv')
root_dir = '~/Data/stemcell_aging'
data = pd.read_csv('~/Data/stemcell_aging/Ania_M_allAnia_percent-engraftment_052219_long.csv')

onep_anywhere = data[data.percent_engraftment > .01]
onep_anywhere = onep_anywhere[['code', 'mouse_id']].drop_duplicates()
filt_data = data.merge(
    onep_anywhere,
    how='inner',
    on=['code', 'mouse_id'],
    validate='m:1'
)


cell_type = 'gr'
cell_data = filt_data[filt_data.cell_type == cell_type]
piv = cell_data.pivot_table(
    index=['code'],
    columns=['day'],
    values='percent_engraftment'
)
piv.to_csv(root_dir + '/STEM_Clustering/'+ cell_type +'_data_filt-0-01p-anywhere.tsv', sep='\t')
