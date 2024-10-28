# Імпортуємо необхідні бібліотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

# 4.1 Завантаження даних "Wine dataset" без міток класів та стандартизація
wine = load_wine()
X = wine.data

# Стандартизація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Визначаємо три метрики зовнішнього оцінювання
metrics = {
    "Adjusted Rand Index": adjusted_rand_score,
    "Normalized Mutual Information": normalized_mutual_info_score,
    "Fowlkes-Mallows Index": fowlkes_mallows_score
}

# 4.2 Побудова дендрограм для всіх значень параметру method функції linkage()
methods = ['ward', 'complete', 'average', 'single']
plt.figure(figsize=(18, 10))

# Побудова дендрограм для кожного з методів
for i, method in enumerate(methods, 1):
    plt.subplot(2, 2, i)
    Z = linkage(X_scaled, method=method)
    dendrogram(Z)
    plt.title(f'Dendrogram ({method} linkage)')
    plt.xlabel('Data points')
    plt.ylabel('Distance')

plt.tight_layout()
plt.show()
