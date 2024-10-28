import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

# Завантаження та нормалізація даних
digits = load_digits()
X = digits.data
y = digits.target
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Список методів для аналізу
methods = {
    "PCA": PCA(n_components=2),
    "Random Projection": GaussianRandomProjection(n_components=2, random_state=0),
    "Isomap": Isomap(n_components=2, n_neighbors=10),
    "MDS": MDS(n_components=2, random_state=0),
    "Spectral Embedding": SpectralEmbedding(n_components=2),
    "Locally Linear Embedding": LocallyLinearEmbedding(n_components=2, n_neighbors=10, method='standard')
}


# Функція для візуалізації та оцінки результатів кластеризації
def analyze_and_plot(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar()
    plt.show()

    # Кластеризація з KMeans
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=0).fit(data)
    kmeans_labels = kmeans.labels_

    # Оцінка якості кластеризації з ARI
    ari_score = adjusted_rand_score(y, kmeans_labels)
    print(f"{title} - Adjusted Rand Index (ARI): {ari_score:.4f}")


# Проходження по всім методам, зниження розмірності, візуалізація та оцінка
for method_name, method in methods.items():
    X_transformed = method.fit_transform(X_normalized)
    analyze_and_plot(X_transformed, y, method_name)
