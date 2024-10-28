# Re-import necessary libraries after environment reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import Isomap, TSNE

# 5.1 Завантаження набору даних "Digits" та відбір по 100 елементів з класів [0, 1, 2, 5, 6, 8], починаючи з індексу 0
digits = load_digits()
data = digits.data
target = digits.target

# Відбираємо по 100 елементів з кожного класу (0, 1, 2, 5, 6, 8) починаючи з індексу 0
selected_classes = [0, 1, 2, 5, 6, 8]
selected_indices = []

for class_label in selected_classes:
    # Відбираємо перші 100 індексів для кожного класу
    class_indices = np.where(target == class_label)[0][:100]
    selected_indices.extend(class_indices)

# Створюємо підмножину даних і міток для вибраних класів
X_selected = data[selected_indices]
y_selected = target[selected_indices]

# Первинна візуалізація 2-3 представників кожного класу
plt.figure(figsize=(10, 6))
for i, class_label in enumerate(selected_classes):
    # Відображаємо перші 3 зображення кожного класу
    for j in range(3):
        plt.subplot(len(selected_classes), 3, i * 3 + j + 1)
        plt.imshow(X_selected[i * 100 + j].reshape(8, 8), cmap='gray')
        plt.title(f'Class {class_label}')
        plt.axis('off')

plt.tight_layout()
plt.show()

# 5.2 Нормалізація даних
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_selected)

# Перевірка нормалізації (виводимо мінімальні і максимальні значення для кожної ознаки)
min_values = np.min(X_normalized, axis=0)
max_values = np.max(X_normalized, axis=0)

min_values, max_values


# ЧАСТИНА 5.3

# Лінійні методи зниження розмірності
# 1. Метод головних компонент (PCA)
pca_2d = PCA(n_components=2).fit_transform(X_normalized)
pca_3d = PCA(n_components=3).fit_transform(X_normalized)

# 2. Метод випадкових проекцій (Random Projection)
rp_2d = GaussianRandomProjection(n_components=2, random_state=0).fit_transform(X_normalized)
rp_3d = GaussianRandomProjection(n_components=3, random_state=0).fit_transform(X_normalized)

# Нелінійні методи зниження розмірності
# 3. Нелінійний метод Isomap (збільшення n_neighbors)
isomap_2d = Isomap(n_components=2, n_neighbors=10).fit_transform(X_normalized)
isomap_3d = Isomap(n_components=3, n_neighbors=10).fit_transform(X_normalized)

# 4. Алгоритм t-SNE
tsne_2d = TSNE(n_components=2, random_state=0).fit_transform(X_normalized)
tsne_3d = TSNE(n_components=3, random_state=0).fit_transform(X_normalized)


# Функція для візуалізації
def plot_2d(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.xlim([-25, 25])
    plt.ylim([-25, 25])
    plt.show()


def plot_3d(data, labels, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=10)
    ax.set_title(title)
    fig.colorbar(scatter)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([-25, 25])
    plt.show()


# Візуалізація для вкладення в 2D
plot_2d(pca_2d, y_selected, "Метод головних компонент (2D)")
plot_2d(rp_2d, y_selected, "Метод випадкових проєкцій (2D)")
plot_2d(isomap_2d, y_selected, "Isomap (2D)")
plot_2d(tsne_2d, y_selected, "t-SNE (2D)")

# Візуалізація для вкладення в 3D
plot_3d(pca_3d, y_selected, "Метод головних компонент (3D)")
plot_3d(rp_3d, y_selected, "Метод випадкових проєкцій (3D)")
plot_3d(isomap_3d, y_selected, "Isomap (3D)")
plot_3d(tsne_3d, y_selected, "t-SNE (3D)")
