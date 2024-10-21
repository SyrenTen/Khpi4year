import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Завантаження даних Wine
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Стандартизація та нормалізація даних
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X)

# Параметри за варіантом
k1 = 2
k2 = 12
T = 0.4
random_state = 0
k_range = range(k1, k2 + 1)

# Розбивка на навчальні та тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=T, random_state=random_state)
X_standard_train, X_standard_test, _, _ = train_test_split(X_standard, y, test_size=T, random_state=random_state)
X_normalized_train, X_normalized_test, _, _ = train_test_split(X_normalized, y, test_size=T, random_state=random_state)

# Масиви для збереження точності
accuracy_original_train = []
accuracy_original_test = []
accuracy_standard_train = []
accuracy_standard_test = []
accuracy_normalized_train = []
accuracy_normalized_test = []

# Цикл по k для кожної вибірки
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

    # Навчання на оригінальних даних
    knn.fit(X_train, y_train)
    accuracy_original_train.append(accuracy_score(y_train, knn.predict(X_train)))
    accuracy_original_test.append(accuracy_score(y_test, knn.predict(X_test)))

    # Навчання на стандартизованих даних
    knn.fit(X_standard_train, y_train)
    accuracy_standard_train.append(accuracy_score(y_train, knn.predict(X_standard_train)))
    accuracy_standard_test.append(accuracy_score(y_test, knn.predict(X_standard_test)))

    # Навчання на нормалізованих даних
    knn.fit(X_normalized_train, y_train)
    accuracy_normalized_train.append(accuracy_score(y_train, knn.predict(X_normalized_train)))
    accuracy_normalized_test.append(accuracy_score(y_test, knn.predict(X_normalized_test)))

# Формування таблиць результатів для кожної вибірки
results_original = pd.DataFrame({
    'k': list(k_range),
    'Train Accuracy': accuracy_original_train,
    'Test Accuracy': accuracy_original_test
})

results_standard = pd.DataFrame({
    'k': list(k_range),
    'Train Accuracy': accuracy_standard_train,
    'Test Accuracy': accuracy_standard_test
})

results_normalized = pd.DataFrame({
    'k': list(k_range),
    'Train Accuracy': accuracy_normalized_train,
    'Test Accuracy': accuracy_normalized_test
})

# Виведення таблиць
print("Results for Original Data:")
print(results_original)
print("\nResults for Standardized Data:")
print(results_standard)
print("\nResults for Normalized Data:")
print(results_normalized)

# Візуалізація результатів
plt.figure(figsize=(12, 6))
sns.lineplot(x=k_range, y=accuracy_original_train, label='Original Train', marker='o')
sns.lineplot(x=k_range, y=accuracy_original_test, label='Original Test', marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Original Data: k-NN Classification Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x=k_range, y=accuracy_standard_train, label='Standardized Train', marker='o')
sns.lineplot(x=k_range, y=accuracy_standard_test, label='Standardized Test', marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Standardized Data: k-NN Classification Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x=k_range, y=accuracy_normalized_train, label='Normalized Train', marker='o')
sns.lineplot(x=k_range, y=accuracy_normalized_test, label='Normalized Test', marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Normalized Data: k-NN Classification Accuracy')
plt.legend()
plt.show()
