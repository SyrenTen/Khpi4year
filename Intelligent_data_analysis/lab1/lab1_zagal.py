# Iмпортуємо модуль роботи iз датасетом
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np

# Завантажуємо датасет
iris = datasets.load_iris()
# Виводимо заголовки даних датасету
print(iris.keys())
# Виводимо назви ознак
print(iris['feature_names'])
# Виводимо значення ознак
print(iris['data'])
# Виводимо наявнi мiтки класiв
print(iris['target_names'])
# Виводимо номери класiв
print(iris['target'])
# Сформуємо масиви ( NumPy ) ознак X та мiток y
X = iris.data
# або X = iris [ ’ data ’]
y = iris.target
# або y = iris [ ’ target ’]
setosa = X[y == 0]
versicolor = X[y == 1]
virginica = X[y == 2]
print(setosa)
print(versicolor)
print(virginica)
df = pd.DataFrame(X)
iris = datasets.load_iris(as_frame=True)
df = iris.frame

print(df.head())
print(df.tail())
print(df.head(3))
print(df.tail(7))
print(df.index)
print(df.columns)
print(df.dtypes)

data = df.to_numpy()

df_transposed = df.T
print(df_transposed)

print(df.sort_index(axis=1, ascending=True))
print(df.sort_index(axis=1, ascending=False))
print(df.sort_index(axis=0, ascending=False))
print(df.sort_values(by='sepal length (cm)', ascending=False))
print(df.sort_values(by=['sepal length (cm)', 'petal width (cm)']))

# Беремо усi стовпцi, окрiм вказаного
X = df.drop('target', axis=1)
# Обираємо вказаний стовпець
y = df['target']
print(X)
print(y)

# Першi чотири рядки
print(df[:4])
# Рядки з 10 по 39
print(df[10:40])

# Усi рядки заданого стовпця
print(df.loc[:, 'sepal length (cm)'])
# Усi рядки декiлькох стовпцiв
print(df.loc[:, ['sepal length (cm)', 'petal length (cm)']])
# Задана кiлькiсть рядкiв декiлькох стовпцiв
print(df.loc[10:40, ['sepal length (cm)', 'petal length (cm)']])

# Рядок за iндексом
print(df.iloc[149])
# Зрiз рядкiв та стовпцiв
print(df.iloc[3:5, 0:2])
# Перелiк рядкiв та стовпцiв
print(df.iloc[[1, 2, 4], [0, 2]])
print(df.describe())

# Середнє арифметичне
print(df.mean())
# Максимум
print(df.max())
# Мiнiмум
print(df.min())
# Дисперсiя
print(df.var())
# Стандартне вiдхилення
print(df.std())

# Середнє арифметичне вздовж рядкiв
print(df.mean(axis=1))
print(df.value_counts())
print(df.sum())
print(df.sum(axis=1))

# Запис у файл
df.to_csv("table_name.csv")
# Зчитування файлу
df = pd.read_csv("table_name.csv")

df.to_excel("table_name.xlsx", sheet_name="Sheet1")
df = pd.read_excel("table_name.xlsx", "Sheet1", index_col=None, na_values=[" NA "])

sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='target', palette='Greys')
plt.show()

sns.pairplot(df, hue='target')
plt.show()

confusion_matrix = df.drop('target', axis=1).corr()
sns.heatmap(confusion_matrix)

scaler = preprocessing.StandardScaler().fit(X)
print(scaler.mean_)
print(scaler.var_)
print(scaler.scale_)
X_scaled = scaler.transform(X)
print(X_scaled)
print(np.mean(X_scaled))
print(np.std(X_scaled))

X_scaled = preprocessing.StandardScaler().fit_transform(X)

X_normalized = preprocessing.normalize(X, norm='l2')
# X_normalized = preprocessing.Normalizer().fit_transform ( X )

df = df.dropna()
print(df)
df = df.fillna(value=5)
print(df)

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_corrupted = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]

# Масив для тренування імп'ютера з 3 ознаками
X_train = [[7, 2, 3], [4, 5, 6], [10, 5, 9]]
mean_imputer.fit(X_train)
X_imputed = mean_imputer.transform(X_corrupted)
print(X_imputed)
