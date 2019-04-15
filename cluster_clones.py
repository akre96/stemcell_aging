import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AffinityPropagation


train = pd.read_csv('~/Data/stemcell_aging/lineage_bias/lineage_bias_t0-0_from-counts.csv')

train_4m = train[(train.month == 4) & ~(train.group.isna())]

categorical_cols = ['mouse_id', 'group']
numerical_cols = ['gr_percent_engraftment', 'b_percent_engraftment', 'lineage_bias']

X = pd.get_dummies(train_4m[categorical_cols+numerical_cols])
af = AffinityPropagation().fit(X)
print(af)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

X = X.to_numpy()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


