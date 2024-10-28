# Re-import necessary libraries after environment reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# 4.3.1 Agglomerative Clustering
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

# 4.3.2 K-means Clustering
kmeans_results = {metric: [] for metric in metrics}
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
    labels = kmeans.labels_
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        kmeans_results[metric_name].append((n_clusters, score))

# 4.3.3 Mean Shift Clustering
bandwidth_range = np.linspace(0.1, 2.0, 10)
mean_shift_results = {metric: [] for metric in metrics}
for bandwidth in bandwidth_range:
    mean_shift = MeanShift(bandwidth=bandwidth).fit(X_scaled)
    labels = mean_shift.labels_
    n_clusters = len(np.unique(labels))
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        mean_shift_results[metric_name].append((bandwidth, n_clusters, score))

# 4.3.4 DBSCAN Clustering
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

# 4.3.5 Affinity Propagation Clustering
damping_range = np.linspace(0.5, 0.95, 10)
affinity_results = {metric: [] for metric in metrics}
for damping in damping_range:
    affinity = AffinityPropagation(damping=damping, random_state=0).fit(X_scaled)
    labels = affinity.labels_
    n_clusters = len(np.unique(labels))
    for metric_name, metric_func in metrics.items():
        score = metric_func(y_true, labels)
        affinity_results[metric_name].append((damping, n_clusters, score))

# Summary visualization
summary_results = []

# Collecting data for agglomerative clustering
for metric_name in metrics:
    for linkage in linkages:
        for n_clusters in n_clusters_range:
            values = [(x[0], x[2]) for x in agglo_results[metric_name] if x[1] == linkage and x[0] == n_clusters]
            if values:
                score = values[0][1]
                summary_results.append({
                    'Algorithm': 'Agglomerative',
                    'Metric': metric_name,
                    'Clusters': n_clusters,
                    'Score': score,
                    'Combination': f'Agglomerative ({linkage}) - {metric_name}'
                })

# Collecting data for K-means clustering
for metric_name in metrics:
    for n_clusters in n_clusters_range:
        score = next((x[1] for x in kmeans_results[metric_name] if x[0] == n_clusters), None)
        if score is not None:
            summary_results.append({
                'Algorithm': 'K-means',
                'Metric': metric_name,
                'Clusters': n_clusters,
                'Score': score,
                'Combination': f'K-means - {metric_name}'
            })

# Collecting data for Mean Shift clustering
for metric_name in metrics:
    for entry in mean_shift_results[metric_name]:
        bandwidth, n_clusters, score = entry
        summary_results.append({
            'Algorithm': 'Mean Shift',
            'Metric': metric_name,
            'Clusters': n_clusters,
            'Score': score,
            'Combination': f'Mean Shift - {metric_name}'
        })

# Collecting data for DBSCAN clustering
for metric_name in metrics:
    for entry in dbscan_results[metric_name]:
        eps, min_samples, n_clusters, score = entry
        summary_results.append({
            'Algorithm': 'DBSCAN',
            'Metric': metric_name,
            'Clusters': n_clusters,
            'Score': score,
            'Combination': f'DBSCAN - {metric_name}'
        })

# Collecting data for Affinity Propagation clustering
for metric_name in metrics:
    for entry in affinity_results[metric_name]:
        damping, n_clusters, score = entry
        summary_results.append({
            'Algorithm': 'Affinity Propagation',
            'Metric': metric_name,
            'Clusters': n_clusters,
            'Score': score,
            'Combination': f'Affinity Propagation - {metric_name}'
        })

# Convert collected data to DataFrame for visualization
df_summary = pd.DataFrame(summary_results)

# Visualization of summarized clustering results
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_summary, x='Clusters', y='Score', hue='Combination', palette='viridis')
plt.title('Порiвняльна дiаграма алгоритмiв кластеризацiї')
plt.xlabel('Number of Clusters')
plt.ylabel('Quality Metric Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small', title='Algorithm - Metric')
plt.tight_layout()
plt.show()



# Підсумковий аналіз результатів кластеризації та вибір найкращої метрики та алгоритму

# Аналіз кількості знайдених кластерів та метрик для кожного алгоритму

true_num_clusters = len(np.unique(y_true))

# Функція для перевірки, чи кількість знайдених кластерів відповідає реальній
def is_correct_num_clusters(found_clusters, true_clusters=true_num_clusters):
    return found_clusters == true_clusters

# Підрахунок кількості збігів та середніх значень метрик по всім алгоритмам
results_summary = []

# Аналіз агломеративної кластеризації
for metric_name in metrics:
    for linkage in linkages:
        correct_clusters = sum(1 for x in agglo_results[metric_name] if is_correct_num_clusters(x[0]))
        avg_score = np.mean([x[2] for x in agglo_results[metric_name]])
        results_summary.append({
            'Algorithm': f'Agglomerative ({linkage})',
            'Metric': metric_name,
            'Correct Cluster Count': correct_clusters,
            'Average Score': avg_score
        })

# Аналіз K-means кластеризації
for metric_name in metrics:
    correct_clusters = sum(1 for x in kmeans_results[metric_name] if is_correct_num_clusters(x[0]))
    avg_score = np.mean([x[1] for x in kmeans_results[metric_name]])
    results_summary.append({
        'Algorithm': 'K-means',
        'Metric': metric_name,
        'Correct Cluster Count': correct_clusters,
        'Average Score': avg_score
    })

# Аналіз Mean Shift кластеризації
for metric_name in metrics:
    correct_clusters = sum(1 for x in mean_shift_results[metric_name] if is_correct_num_clusters(x[1]))
    avg_score = np.mean([x[2] for x in mean_shift_results[metric_name]])
    results_summary.append({
        'Algorithm': 'Mean Shift',
        'Metric': metric_name,
        'Correct Cluster Count': correct_clusters,
        'Average Score': avg_score
    })

# Аналіз DBSCAN кластеризації
for metric_name in metrics:
    correct_clusters = sum(1 for x in dbscan_results[metric_name] if is_correct_num_clusters(x[2]))
    avg_score = np.mean([x[3] for x in dbscan_results[metric_name]])
    results_summary.append({
        'Algorithm': 'DBSCAN',
        'Metric': metric_name,
        'Correct Cluster Count': correct_clusters,
        'Average Score': avg_score
    })

# Аналіз Affinity Propagation кластеризації
for metric_name in metrics:
    correct_clusters = sum(1 for x in affinity_results[metric_name] if is_correct_num_clusters(x[1]))
    avg_score = np.mean([x[2] for x in affinity_results[metric_name]])
    results_summary.append({
        'Algorithm': 'Affinity Propagation',
        'Metric': metric_name,
        'Correct Cluster Count': correct_clusters,
        'Average Score': avg_score
    })

# Створення DataFrame для аналізу результатів
df_results_summary = pd.DataFrame(results_summary)

# Пошук найкращої метрики
best_metric = df_results_summary.groupby('Metric')['Average Score'].mean().idxmax()
best_metric_score = df_results_summary.groupby('Metric')['Average Score'].mean().max()

# Пошук найкращого алгоритму
best_algorithm = df_results_summary.groupby('Algorithm')['Average Score'].mean().idxmax()
best_algorithm_score = df_results_summary.groupby('Algorithm')['Average Score'].mean().max()

# Виведення результатів
print("Підсумок:")
print(df_results_summary)
print("\nНайкраща метрика в цiлому : ", best_metric, "з середнім значенням: ", best_metric_score)
print("Найкращий алгоритм кластеризацiї: ", best_algorithm, "з середнім значенням: ", best_algorithm_score)
