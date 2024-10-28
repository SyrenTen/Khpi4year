import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, DBSCAN, AffinityPropagation
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

# Завантаження і стандартизація даних
wine = load_wine()
X = wine.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Метрики для оцінювання
metrics = {
    "ARI": adjusted_rand_score,
    "NMI": normalized_mutual_info_score,
    "FMI": fowlkes_mallows_score
}

# Істинні мітки
y_true = wine.target

# 4.3.1 Агломеративна кластеризація
linkages = ['ward', 'complete', 'average', 'single']
n_clusters_range = range(2, 11)
agglo_results = {metric: [] for metric in metrics}

for linkage in linkages:
    for n_clusters in n_clusters_range:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X_scaled)
        labels = clustering.labels_
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_true, labels)
            agglo_results[metric_name].append((n_clusters, linkage, score))

# Візуалізація точкових діаграм для агломеративної кластеризації
for metric_name in metrics:
    plt.figure()
    for linkage in linkages:
        values = [(x[0], x[2]) for x in agglo_results[metric_name] if x[1] == linkage]
        x_vals, y_vals = zip(*values)
        plt.scatter(x_vals, y_vals, label=linkage)
    plt.title(f'Agglomerative Clustering ({metric_name})')
    plt.xlabel('n_clusters')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# 4.3.2 K-means кластеризація
kmeans_results = {metric: [] for metric in metrics}
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
    labels = kmeans.labels_
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        kmeans_results[metric_name].append((n_clusters, score))

# Візуалізація графіків для K-means
for metric_name in metrics:
    plt.figure()
    values = [(x[0], x[1]) for x in kmeans_results[metric_name]]
    x_vals, y_vals = zip(*values)
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(f'K-means Clustering ({metric_name})')
    plt.xlabel('n_clusters')
    plt.ylabel(metric_name)
    plt.show()

# 4.3.3 Mean Shift кластеризація
bandwidth_range = np.linspace(0.1, 2.0, 10)
mean_shift_results = {metric: [] for metric in metrics}
for bandwidth in bandwidth_range:
    mean_shift = MeanShift(bandwidth=bandwidth).fit(X_scaled)
    labels = mean_shift.labels_
    n_clusters = len(np.unique(labels))
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        mean_shift_results[metric_name].append((bandwidth, n_clusters, score))

# Візуалізація графіків для Mean Shift
for metric_name in metrics:
    plt.figure()
    values = [(x[0], x[2]) for x in mean_shift_results[metric_name]]
    x_vals, y_vals = zip(*values)
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(f'Mean Shift Clustering ({metric_name})')
    plt.xlabel('bandwidth')
    plt.ylabel(metric_name)
    plt.show()

# 4.3.4 DBSCAN кластеризація
eps_range = np.linspace(0.1, 2.0, 10)
min_samples_range = range(2, 11)
dbscan_results = {metric: [] for metric in metrics}
for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = dbscan.labels_
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_true, labels)
            dbscan_results[metric_name].append((eps, min_samples, n_clusters, score))

# Візуалізація точкових діаграм для DBSCAN
for metric_name in metrics:
    plt.figure()
    for min_samples in min_samples_range:
        values = [(x[0], x[3]) for x in dbscan_results[metric_name] if x[1] == min_samples]
        x_vals, y_vals = zip(*values)
        plt.scatter(x_vals, y_vals, label=f'min_samples={min_samples}')
    plt.title(f'DBSCAN Clustering ({metric_name})')
    plt.xlabel('eps')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

# 4.3.5 Affinity Propagation кластеризація
damping_range = np.linspace(0.5, 0.95, 10)
affinity_results = {metric: [] for metric in metrics}
for damping in damping_range:
    affinity = AffinityPropagation(damping=damping, random_state=0).fit(X_scaled)
    labels = affinity.labels_
    n_clusters = len(np.unique(labels))
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        affinity_results[metric_name].append((damping, n_clusters, score))

# Візуалізація графіків для Affinity Propagation
for metric_name in metrics:
    plt.figure()
    values = [(x[0], x[2]) for x in affinity_results[metric_name]]
    x_vals, y_vals = zip(*values)
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(f'Affinity Propagation Clustering ({metric_name})')
    plt.xlabel('damping')
    plt.ylabel(metric_name)
    plt.show()
