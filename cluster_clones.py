import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from plotting_functions import save_plot

save=True
COLOR_PALETTES = json.load(open('color_palettes.json', 'r'))
categorical_cols = ['mouse_id', 'group']
categorical_cols = ['mouse_id']
numerical_cols = ['gr_percent_engraftment', 'b_percent_engraftment', 'lineage_bias']
n_clusters=5

encoder = LabelEncoder()
#scaler = StandardScaler()
#scaler_type = 'standard'

scaler = RobustScaler()
scaler_type = 'robust'

initials = [x[:3] for x in numerical_cols+categorical_cols]
save_folder = '_'.join(initials) + '_' + str(n_clusters) + '-clusters' + '_s-' + scaler_type

# Uncomment for Aging Data
#train = pd.read_csv('~/Data/stemcell_aging/lineage_bias/lineage_bias_t0-0_from-counts.csv')
#timepoint_col='month'
#save_path = '/home/sakre/Data/stemcell_aging/Graphs/Cluster_Exploration/'+save_folder

# Uncomment for Serial Transplant Data
train = pd.read_csv('~/Data/serial_transplant_data/lineage_bias/lineage_bias_t0-0_from-counts.csv')
timepoint_col='day'
save_path = '/home/sakre/Data/serial_transplant_data/Graphs/Cluster_Exploration/'+save_folder

data = train[train.group.isin(['aging_phenotype', 'no_change'])]
train = train[train.group.isin(['aging_phenotype', 'no_change'])]

for cat in categorical_cols:
    train[cat] = encoder.fit_transform(train[cat])

train[numerical_cols] = scaler.fit_transform(train[numerical_cols])

# Transform to shape [code, time, nfeatures]
X_train = np.zeros(
    (len(train[['code', 'mouse_id']].drop_duplicates()),
    train[timepoint_col].nunique(),
    len(numerical_cols) + len(categorical_cols))
)
i = 0
for _, clone_group in train.groupby(['code', 'mouse_id']):
    j = 0
    for _, time_group in clone_group.groupby(timepoint_col):
        X_train[i][j] = time_group.loc[:, numerical_cols + categorical_cols].to_numpy()
        j = j + 1
    i = i+1
sz = X_train.shape[1]

seed=100

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

with_label_df = pd.DataFrame()
pd.options.mode.chained_assignment = None
for _, clone_group in data.groupby(['code', 'mouse_id']):
    j = 0
    clone_test = np.zeros((
        1,
        train[timepoint_col].nunique(),
        len(numerical_cols) + len(categorical_cols)
    ))
    for _, time_group in clone_group.groupby(timepoint_col):
        for cat in categorical_cols:
            time_group[cat] = encoder.transform(time_group[cat])
        time_group[numerical_cols] = scaler.transform(time_group[numerical_cols])
        clone_test[0][j] = time_group.loc[:, numerical_cols + categorical_cols].to_numpy()
        j = j + 1
    
    label = km.predict(clone_test)
    if len(label) > 1:
        print(label)
    clone_group = clone_group.assign(label=str(label[0]))
    with_label_df = with_label_df.append(clone_group)

pd.options.mode.chained_assignment = 'warn'
lb_mean_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id',timepoint_col,'label', 'group']).lineage_bias.mean()
).reset_index().rename(columns={'lineage_bias': 'Average Lineage Bias'})
b_sum_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id',timepoint_col,'label', 'group']).b_percent_engraftment.sum()
).reset_index().rename(columns={'b_percent_engraftment':'Sum B Abundance'})
gr_sum_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id',timepoint_col,'label', 'group']).gr_percent_engraftment.sum()
).reset_index().rename(columns={'gr_percent_engraftment':'Sum Gr Abundance'})
b_mean_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id',timepoint_col,'label', 'group']).b_percent_engraftment.mean()
).reset_index().rename(columns={'b_percent_engraftment':'Average B Abundance'})
gr_mean_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id',timepoint_col,'label', 'group']).gr_percent_engraftment.mean()
).reset_index().rename(columns={'gr_percent_engraftment':'Average Gr Abundance'})
sum_data_df = lb_mean_data_df.merge(b_sum_data_df, on=['mouse_id', timepoint_col, 'label', 'group'])
sum_data_df = sum_data_df.merge(gr_sum_data_df, on=['mouse_id', timepoint_col, 'label', 'group'])
sum_data_df = sum_data_df.merge(gr_mean_data_df, on=['mouse_id', timepoint_col, 'label', 'group'])
sum_data_df = sum_data_df.merge(b_mean_data_df, on=['mouse_id', timepoint_col, 'label', 'group'])


for y_col in ['Sum B Abundance', 'Sum Gr Abundance', 'Average Gr Abundance', 'Average B Abundance', 'Average Lineage Bias']:
    plt.figure(figsize=(8, 14))
    sns.lineplot(
        x=timepoint_col,
        y=y_col,
        data=sum_data_df,
        hue='label',
        palette=COLOR_PALETTES['cluster_label'],
        style='group',
        err_style=None,
        linewidth=4,
    )
    plt.title(y_col)
    fname = save_path + os.sep + y_col.lower().replace(' ', '_') + '.png'
    save_plot(fname, save, 'png')

for name, cluster in with_label_df.groupby(['group', 'label']):
    print(
        'Group: ' \
        + name[0] \
        + ', Label: ' \
        + name[1] \
        + ', Unique Mice: ' \
        + str(cluster.mouse_id.nunique()) \
        + ', Unique Clones: ' \
        + str(cluster.code.nunique()) \

    )
if not save:
    plt.show()