import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


charlie_data = pd.read_csv(
    '~/Data/charlie_data/M_all_percent-engraftment_052019-1_long.csv'
)


for (cell_type, mouse_id), c_df in charlie_data.groupby(['cell_type', 'mouse_id']):
    plt.figure()
    sorted_df = c_df.sort_values(by=['day','percent_engraftment'])
    piv = sorted_df.pivot_table(
        index='code',
        columns='day',
        values='percent_engraftment',
        fill_value=0,
    )
    plt.stackplot(
        piv.columns,
        piv.values
    )
    plt.title(cell_type + mouse_id)
    plt.savefig('/home/sakre/Data/charlie_data/Graphs/'+cell_type+'-'+mouse_id+'.png')


