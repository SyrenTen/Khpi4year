import numpy as np
import pandas as pd
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

# Ініціалізація словників для зберігання результатів
agglo_results = {metric: [] for metric in metrics}
kmeans_results = {metric: [] for metric in metrics}
mean_shift_results = {metric: [] for metric in metrics}
dbscan_results = {metric: [] for metric in metrics}
affinity_results = {metric: [] for metric in metrics}

# 4.3.1 Агломеративна кластеризація
linkages = ['ward', 'complete', 'average', 'single']
n_clusters_range = range(2, 11)

for linkage in linkages:
    for n_clusters in n_clusters_range:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X_scaled)
        labels = clustering.labels_
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_true, labels)
            agglo_results[metric_name].append((n_clusters, linkage, score))

# 4.3.2 K-means кластеризація
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
    labels = kmeans.labels_
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        kmeans_results[metric_name].append((n_clusters, score))

# 4.3.3 Mean Shift кластеризація
bandwidth_range = np.linspace(0.1, 2.0, 10)
for bandwidth in bandwidth_range:
    mean_shift = MeanShift(bandwidth=bandwidth).fit(X_scaled)
    labels = mean_shift.labels_
    n_clusters = len(np.unique(labels))
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        mean_shift_results[metric_name].append((bandwidth, n_clusters, score))

# 4.3.4 DBSCAN кластеризація
eps_range = np.linspace(0.1, 2.0, 10)
min_samples_range = range(2, 11)
for eps in eps_range:
    for min_samples in min_samples_range:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = dbscan.labels_
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        for metric_name, metric_func in metrics.items():
            score = metric_func(y_true, labels)
            dbscan_results[metric_name].append((eps, min_samples, n_clusters, score))

# 4.3.5 Affinity Propagation кластеризація
damping_range = np.linspace(0.5, 0.95, 10)
for damping in damping_range:
    affinity = AffinityPropagation(damping=damping, random_state=0).fit(X_scaled)
    labels = affinity.labels_
    n_clusters = len(np.unique(labels))
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        affinity_results[metric_name].append((damping, n_clusters, score))

# Створення DataFrames для кожного алгоритму і метрики

# 1. Agglomerative Clustering DataFrames
agglo_tables = {metric: pd.DataFrame(columns=['n_clusters', 'linkage', metric]) for metric in metrics}
for metric_name in metrics:
    for entry in agglo_results[metric_name]:
        n_clusters, linkage, score = entry
        agglo_tables[metric_name] = pd.concat([agglo_tables[metric_name],
                                               pd.DataFrame({'n_clusters': [n_clusters], 'linkage': [linkage], metric_name: [score]})], ignore_index=True)

# 2. K-means Clustering DataFrames
kmeans_tables = {metric: pd.DataFrame(columns=['n_clusters', metric]) for metric in metrics}
for metric_name in metrics:
    for entry in kmeans_results[metric_name]:
        n_clusters, score = entry
        kmeans_tables[metric_name] = pd.concat([kmeans_tables[metric_name],
                                                pd.DataFrame({'n_clusters': [n_clusters], metric_name: [score]})], ignore_index=True)

# 3. Mean Shift Clustering DataFrames
mean_shift_tables = {metric: pd.DataFrame(columns=['bandwidth', 'n_clusters', metric]) for metric in metrics}
for metric_name in metrics:
    for entry in mean_shift_results[metric_name]:
        bandwidth, n_clusters, score = entry
        mean_shift_tables[metric_name] = pd.concat([mean_shift_tables[metric_name],
                                                    pd.DataFrame({'bandwidth': [bandwidth], 'n_clusters': [n_clusters], metric_name: [score]})], ignore_index=True)

# 4. DBSCAN Clustering DataFrames
dbscan_tables = {metric: pd.DataFrame(columns=['eps', 'min_samples', 'n_clusters', metric]) for metric in metrics}
for metric_name in metrics:
    for entry in dbscan_results[metric_name]:
        eps, min_samples, n_clusters, score = entry
        dbscan_tables[metric_name] = pd.concat([dbscan_tables[metric_name],
                                                pd.DataFrame({'eps': [eps], 'min_samples': [min_samples], 'n_clusters': [n_clusters], metric_name: [score]})], ignore_index=True)

# 5. Affinity Propagation Clustering DataFrames
affinity_tables = {metric: pd.DataFrame(columns=['damping', 'n_clusters', metric]) for metric in metrics}
for metric_name in metrics:
    for entry in affinity_results[metric_name]:
        damping, n_clusters, score = entry
        affinity_tables[metric_name] = pd.concat([affinity_tables[metric_name],
                                                  pd.DataFrame({'damping': [damping], 'n_clusters': [n_clusters], metric_name: [score]})], ignore_index=True)

# Відображення таблиць у консолі
print("Agglomerative Clustering (ARI):")
print(agglo_tables['ARI'])
print("\nK-means Clustering (ARI):")
print(kmeans_tables['ARI'])
print("\nMean Shift Clustering (ARI):")
print(mean_shift_tables['ARI'])
print("\nDBSCAN Clustering (ARI):")
print(dbscan_tables['ARI'])
print("\nAffinity Propagation Clustering (ARI):")
print(affinity_tables['ARI'])

# Збереження таблиць у Excel файли
with pd.ExcelWriter('clustering_results.xlsx') as writer:
    for metric_name in metrics:
        agglo_tables[metric_name].to_excel(writer, sheet_name=f'Agglomerative_{metric_name}')
        kmeans_tables[metric_name].to_excel(writer, sheet_name=f'KMeans_{metric_name}')
        mean_shift_tables[metric_name].to_excel(writer, sheet_name=f'MeanShift_{metric_name}')
        dbscan_tables[metric_name].to_excel(writer, sheet_name=f'DBSCAN_{metric_name}')
        affinity_tables[metric_name].to_excel(writer, sheet_name=f'Affinity_{metric_name}')
