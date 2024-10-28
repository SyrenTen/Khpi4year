import numpy as np
from sklearn.decomposition import PCA
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

# Діапазон розмірностей для зниження
dimensions = range(3, 13)

# Зберігання результатів
results = []

# Цикл для зниження розмірності та кластеризації для кожної розмірності
for dim in dimensions:
    # Зниження розмірності методом PCA
    pca = PCA(n_components=dim)
    X_reduced = pca.fit_transform(X_normalized)

    # Кластеризація з KMeans, де кількість кластерів дорівнює кількості класів
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=0)
    kmeans_labels = kmeans.fit_predict(X_reduced)

    # Оцінка якості кластеризації за метрикою Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(y, kmeans_labels)

    # Фіксація кількості знайдених кластерів
    num_clusters = len(np.unique(kmeans_labels))

    # Зберігаємо результати для кожної розмірності
    results.append({
        'Dimension': dim,
        'ARI Score': ari_score,
        'Found Clusters': num_clusters
    })

# Виведення результатів
for result in results:
    print(
        f"Dimension: {result['Dimension']}, ARI Score: {result['ARI Score']:.4f}, Found Clusters: {result['Found Clusters']}")

# Короткий висновок
print("\nВисновок: Метод PCA зі збільшенням розмірності (3-12) показує, як змінюється якість кластеризації.")
print(
    "Оптимальний баланс досягається на певній розмірності, де значення ARI максимальне, що вказує на найбільш "
    "відповідний простір.")
print(
    "Зокрема, KMeans у знижених просторах зберігає точність за класифікацією, коли кількість кластерів близька до "
    "кількості класів.")
