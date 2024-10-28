import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import MDS, Isomap, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження та нормалізація даних з відбором підмножини для зменшення обчислень
digits = load_digits()
data = digits.data[:500]  # Зменшення кількості зразків до 500
target = digits.target[:500]
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(data)

# Реальна кількість класів
true_num_clusters = len(np.unique(target))

# Діапазон розмірностей для аналізу
dimensions = range(3, 8)  # Менший діапазон для швидшого виконання

# Зберігаємо результати
results = []


# Функція для аналізу зниження розмірності та кластеризації
def analyze_method(method_name, embedding_func, X, dimensions, true_labels):
    for dim in dimensions:
        # Зниження розмірності
        X_embedded = embedding_func(X, dim)

        # Кластеризація
        kmeans = KMeans(n_clusters=true_num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(X_embedded)

        # Оцінка якості кластеризації
        ari_score = adjusted_rand_score(true_labels, cluster_labels)

        # Зберігаємо результат
        results.append({
            'Method': method_name,
            'Dimensions': dim,
            'Clusters Found': len(np.unique(cluster_labels)),
            'ARI Score': ari_score
        })


# Визначення функцій для кожного методу зниження розмірності
def apply_pca(X, dim):
    return PCA(n_components=dim).fit_transform(X)


def apply_rp(X, dim):
    return GaussianRandomProjection(n_components=dim, random_state=0).fit_transform(X)


def apply_mds(X, dim):
    return MDS(n_components=dim, random_state=0, max_iter=100).fit_transform(X)  # Обмеження ітерацій MDS


def apply_isomap(X, dim):
    return Isomap(n_components=dim, n_neighbors=10).fit_transform(X)


def apply_spectral(X, dim):
    return SpectralEmbedding(n_components=dim, random_state=0).fit_transform(X)


def apply_lle(X, dim):
    return LocallyLinearEmbedding(n_components=dim, n_neighbors=10, method='standard').fit_transform(X)


# Лінійні методи
analyze_method("PCA", apply_pca, X_normalized, dimensions, target)
analyze_method("Random Projection", apply_rp, X_normalized, dimensions, target)

# Нелінійні методи
analyze_method("MDS", apply_mds, X_normalized, dimensions, target)
analyze_method("Isomap", apply_isomap, X_normalized, dimensions, target)
analyze_method("Spectral Embedding", apply_spectral, X_normalized, dimensions, target)
analyze_method("Locally Linear Embedding", apply_lle, X_normalized, dimensions, target)

# Виведення результатів
import pandas as pd

df_results = pd.DataFrame(results)
print(df_results)

# Візуалізація результатів
plt.figure(figsize=(12, 8))
for method in df_results['Method'].unique():
    subset = df_results[df_results['Method'] == method]
    plt.plot(subset['Dimensions'], subset['ARI Score'], marker='o', label=method)

plt.title("Comparison of Dimensionality Reduction Methods for Clustering Quality (ARI Score)")
plt.xlabel("Number of Dimensions")
plt.ylabel("ARI Score")
plt.legend()
plt.grid(True)
plt.show()

# Висновок
best_result = df_results.loc[df_results['ARI Score'].idxmax()]
print("\nBest Method:", best_result['Method'])
print("Optimal Dimensions:", best_result['Dimensions'])
print("Clusters Found:", best_result['Clusters Found'])
print("ARI Score:", best_result['ARI Score'])





# Узагальнена таблиця результатів кластеризації
results_summary = pd.DataFrame(results)  # Замініть 'results' на вашу змінну з результатами
results_summary = results_summary[['Method', 'Dimensions', 'Clusters Found', 'ARI Score']]

# Виведення узагальненої таблиці
print("Узагальнена таблиця результатів кластеризації:")
print(results_summary)

# Побудова графіка залежності якості кластеризації від розмірності
plt.figure(figsize=(12, 8))
for method in results_summary['Method'].unique():
    subset = results_summary[results_summary['Method'] == method]
    plt.plot(subset['Dimensions'], subset['ARI Score'], marker='o', label=method)

plt.title("Залежність якості кластеризації (ARI) від розмірності для різних методів зниження розмірності")
plt.xlabel("Кількість вимірів")
plt.ylabel("ARI (Adjusted Rand Index)")
plt.legend()
plt.grid(True)
plt.show()

# Пошук найкращого методу та розмірності за ARI
best_result = results_summary.loc[results_summary['ARI Score'].idxmax()]
print("\nНайкращий метод кластеризації за ARI:")
print("Метод:", best_result['Method'])
print("Розмірність:", best_result['Dimensions'])
print("Знайдена кількість кластерів:", best_result['Clusters Found'])
print("ARI:", best_result['ARI Score'])
