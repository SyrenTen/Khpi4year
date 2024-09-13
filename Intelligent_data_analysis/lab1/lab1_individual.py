from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np

# Завантаження Iris dataset
iris = datasets.load_iris(as_frame=True)
df = iris.frame

# Вибір індексів
S = 44
N1 = 0
N2 = 3

# Вибір 5 об'єктів з кожного класу
df_subset = pd.concat([df.iloc[S:S+5], df.iloc[S+50:S+55], df.iloc[S+100:S+105]])

# Збереження таблиці у CSV
df_subset.to_csv('subset.csv')

# Виведення статистичних показників
print(df_subset.describe())

# Scatterplot
sns.scatterplot(data=df_subset, x='petal length (cm)', y='petal width (cm)', hue='target', palette='Greys')
plt.show()

# Pairplot
sns.pairplot(df_subset, hue='target')
plt.show()

# Heatmap кореляції
correlation_matrix = df_subset.drop('target', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Стандартизація
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(df_subset.drop('target', axis=1))

df_scaled = pd.DataFrame(X_scaled, columns=df_subset.columns[:-1])
df_scaled['target'] = df_subset['target'].values

# Виведення статистичних показників
print(df_scaled.describe())

# Запис у CSV
df_scaled.to_csv('scaled_subset.csv')

# Нормалізація
X_normalized = preprocessing.normalize(df_subset.drop('target', axis=1))

df_normalized = pd.DataFrame(X_normalized, columns=df_subset.columns[:-1])
df_normalized['target'] = df_subset['target'].values

# Виведення статистичних показників
print(df_normalized.describe())

# Запис у CSV
df_normalized.to_csv('normalized_subset.csv')

# Додавання пропусків у стовпці за індексами N1 та N2
df_with_nans = df.copy()
df_with_nans.iloc[S:S+2, N1] = np.nan
df_with_nans.iloc[S:S+3, N2] = np.nan

print(df_with_nans)
df_with_nans.to_csv('df_with_nans.csv')

# Видалення рядків з пропусками
df_dropna = df_with_nans.dropna()

# Заповнення пропусків середнім
df_fillna = df_with_nans.fillna(df_with_nans.mean(numeric_only=True))

# Заповнення за допомогою SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_with_nans_numeric = df_with_nans.drop('target', axis=1)  # Виключаємо колонку 'target'
df_imputed = pd.DataFrame(imputer.fit_transform(df_with_nans_numeric), columns=df_with_nans_numeric.columns)

df_imputed['target'] = df_with_nans['target'].values

# Виведення статистичних показників
print(df_dropna.describe())
print(df_fillna.describe())
print(df_imputed.describe())

# Запис у файли
df_dropna.to_csv('df_dropna.csv')
df_fillna.to_csv('df_fillna.csv')
df_imputed.to_csv('df_imputed.csv')

# Стандартизація відновлених датасетів
df_fillna_scaled = pd.DataFrame(scaler.fit_transform(df_fillna.drop('target', axis=1)), columns=df_fillna.columns[:-1])
df_fillna_scaled['target'] = df_fillna['target'].values  # Додаємо стовпець 'target'

df_imputed_scaled = pd.DataFrame(scaler.fit_transform(df_imputed.drop('target', axis=1)), columns=df_imputed.columns[:-1])
df_imputed_scaled['target'] = df_imputed['target'].values  # Додаємо стовпець 'target'

# Виведення статистики
print(df_fillna_scaled.describe())
print(df_imputed_scaled.describe())

# Побудова pairplot
sns.pairplot(df_imputed_scaled, hue='target')
plt.show()
