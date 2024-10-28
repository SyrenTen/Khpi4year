# Re-import necessary libraries after environment reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

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
