#%% [markdown]
## Import files

#%%
import os
import json
from collections import OrderedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import aggregate_functions as agg
import plotting_functions as pf

#%% [markdown]
# # Young Vs Old Mouse HSC Abundance

#%%
young_data_root = '~/Data/young_mouse_hscs'
young_lb_file = os.path.join(young_data_root, 'lineage_bias/lineage_bias_t0-0_from-counts.csv')
young_abundance = pd.read_csv(
    os.path.join(young_data_root, "M_all_Ania_percent-engraftment_filter 0.0001 070919_long.csv")
)
young_lineage_bias = pd.read_csv(young_lb_file)
young_hsc_data = young_abundance[young_abundance.cell_type == 'hsc']

old_data_root = '~/Data/stemcell_aging'
old_lb_file = os.path.join(old_data_root, 'lineage_bias/lineage_bias_t0-0_from-counts.csv')
old_abundance = pd.read_csv(
    os.path.join(old_data_root, "Ania_M_allAnia_percent-engraftment_052219_long.csv")
)
old_lineage_bias = pd.read_csv(old_lb_file)
old_hsc_data = old_abundance[old_abundance.cell_type == 'hsc']
#%%
last_young_clones = agg.find_last_clones(
    young_lineage_bias,
    'day'
)
labeled_young_last_clones = agg.add_bias_category(
    last_young_clones
)
bias_young_hsc_abundance_df = young_hsc_data.merge(
    labeled_young_last_clones[['code', 'mouse_id', 'bias_category_short']],
    how='inner',
    validate='1:1'
).assign(age='young')
last_old_clones = agg.find_last_clones(
    old_lineage_bias,
    'day'
)
labeled_old_last_clones = agg.add_bias_category(
    last_old_clones
)
bias_old_hsc_abundance_df = old_hsc_data.merge(
    labeled_old_last_clones[['code', 'mouse_id', 'bias_category_short']],
    how='inner',
    validate='1:1'
).assign(age='old')

hsc_bias_abundance = bias_old_hsc_abundance_df.append(bias_young_hsc_abundance_df)
#%%
sns.set_context(
    'paper',
    rc={
        'lines.linewidth': 3,
        'axes.linewidth': 4,
        'axes.labelsize': 20,
        'xtick.major.width': 5,
        'ytick.major.width': 5,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.titlesize': 'small',
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.facecolor': 'white'
    }

    )
COLOR_PALETTES = json.load(open('lib/color_palettes.json', 'r'))
cats_order = [
    'Ly Biased',
    'Balanced',
    'My Biased',
]
unique_cats = list(OrderedDict.fromkeys(cats_order))


plt.figure(figsize=(12,8))
ax = sns.barplot(
    y='percent_engraftment',
    x='bias_category_short',
    order=unique_cats,
    data=hsc_bias_abundance,
    hue='age',
)
ax.set(yscale='log')
plt.xlabel('')
plt.ylabel('HSC Abundance (% HSCs)')
plt.title(
    'HSC Abundance by Bias at Last Time Point'
)


save_format='png'
fname = '/home/sakre/Data/young_mouse_hscs/Graphs/' \
    + 'hsc_abundance_yvo' \
    + '.' + save_format

plt.savefig(
    fname,
    format=save_format,
    bbox_inches='tight',
)
#%%


#%%
