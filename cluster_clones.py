import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

COLOR_PALETTES = json.load(open('color_palettes.json', 'r'))
train = pd.read_csv('~/Data/stemcell_aging/lineage_bias/lineage_bias_t0-0_from-counts.csv')
train = train[train.group.isin(['aging_phenotype','no_change'])]

categorical_cols = ['mouse_id', 'group']
numerical_cols = ['gr_percent_engraftment', 'b_percent_engraftment', 'lineage_bias']
numerical_cols = ['b_percent_engraftment']

#scaler = StandardScaler()
#train[numerical_cols] = scaler.fit_transform(train[numerical_cols])

encoded = pd.get_dummies(train[categorical_cols+numerical_cols])

# Transform to shape [code, time, nfeatures]
X_train = np.zeros((len(train[['code', 'mouse_id']].drop_duplicates()), 4, 3))
i = 0
for _, clone_group in train.groupby(['code', 'mouse_id']):
    j = 0
    for _, time_group in clone_group.groupby('month'):
        X_train[i][j] = time_group[numerical_cols].to_numpy()
        j = j + 1
    i = i+1
sz = X_train.shape[1]

seed=100
n_clusters=5

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

with_label_df = pd.DataFrame()
for _, clone_group in train.groupby(['code', 'mouse_id']):
    j = 0
    clone_test = np.zeros((1, 4, 3))
    for _, time_group in clone_group.groupby('month'):
        clone_test[0][j] = time_group[numerical_cols].to_numpy()
        j = j + 1
    
    label = km.predict(clone_test)
    if len(label) > 1:
        print(label)
    clone_group = clone_group.assign(label=str(label[0]))
    with_label_df = with_label_df.append(clone_group)

lb_sum_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id','month','label', 'group']).lineage_bias.mean()
).reset_index()
b_sum_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id','month','label', 'group']).b_percent_engraftment.sum()
).reset_index()
gr_sum_data_df = pd.DataFrame(
    with_label_df.groupby(['mouse_id','month','label', 'group']).gr_percent_engraftment.sum()
).reset_index()
sum_data_df = lb_sum_data_df.merge(b_sum_data_df, on=['mouse_id', 'month', 'label', 'group'])
sum_data_df = sum_data_df.merge(gr_sum_data_df, on=['mouse_id', 'month', 'label', 'group'])


for gname, group_df in sum_data_df.groupby('group'):
    plt.figure()
    plt.subplot(3, 1, 1)
    sns.lineplot(
        x='month',
        y='lineage_bias',
        data=group_df,
        hue='label',
        legend=None,
        palette=COLOR_PALETTES['cluster_label'],
    )
    plt.ylabel('Lineage Bias Average by Mouse')

    plt.subplot(3, 1, 2)
    sns.lineplot(
        x='month',
        y='b_percent_engraftment',
        data=group_df,
        hue='label',
        legend=None,
        palette=COLOR_PALETTES['cluster_label'],
    )
    plt.ylabel('B Abundance (%WBC) Sum Per Mouse')

    plt.subplot(3, 1, 3)
    sns.lineplot(
        x='month',
        y='gr_percent_engraftment',
        data=group_df,
        hue='label',
        palette=COLOR_PALETTES['cluster_label'],
    )
    plt.ylabel('Gr Abundance (%WBC) Sum Per Mouse')
    plt.suptitle('Group: ' + gname.title())

    plt.figure()
    plt.subplot(3, 1, 1)
    sns.lineplot(
        x='month',
        y='lineage_bias',
        data=group_df,
        hue='label',
        legend=None,
        units='mouse_id',
        estimator=None,
        palette=COLOR_PALETTES['cluster_label'],
    )
    plt.ylabel('Lineage Bias Average by Mouse')

    plt.subplot(3, 1, 2)
    sns.lineplot(
        x='month',
        y='b_percent_engraftment',
        data=group_df,
        hue='label',
        legend=None,
        units='mouse_id',
        estimator=None,
        palette=COLOR_PALETTES['cluster_label'],
    )
    plt.ylabel('B Abundance (%WBC) Sum Per Mouse')

    plt.subplot(3, 1, 3)
    sns.lineplot(
        x='month',
        y='gr_percent_engraftment',
        data=group_df,
        hue='label',
        units='mouse_id',
        estimator=None,
        palette=COLOR_PALETTES['cluster_label'],
    )
    plt.ylabel('Gr Abundance (%WBC) Sum Per Mouse')
    plt.suptitle('Group: ' + gname.title())

for name, cluster in with_label_df.groupby(['group', 'label']):
    print(
        'Group: ' \
        + name[0] \
        + ' Label: ' \
        + name[1] \
        + ' Unique Clones: ' \
        + str(cluster.code.nunique())
    )
plt.show()