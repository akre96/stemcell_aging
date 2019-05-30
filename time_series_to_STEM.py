import pandas as pd
data = pd.read_csv('~/Data/serial_transplant_data/M_allAniaAnia serial transpl_percent-engraftment_121018_long.csv')

gr_data = data[data.cell_type == 'b']
piv = gr_data.pivot_table(
    index=['code'],
    columns=['day'],
    values='percent_engraftment'
)

piv.to_csv('~/Data/serial_transplant_data/STEM_Clustering/b_data.tsv', sep='\t')
