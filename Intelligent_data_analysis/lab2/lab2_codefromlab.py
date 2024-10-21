# Iмпортуємо бiблiотеки вiзуалiзацiї
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Iмпортуємо функцiю розбиття датасету на навчальну та тестову вибiрки
from sklearn.model_selection import train_test_split

# Iмпортуємо модулi роботи iз датасетом та класифiкатором
from sklearn import datasets, neighbors

k = 1  # Задаємо кiлькiсть сусiдiв

iris = datasets.load_iris()  # Завантажуємо датасет

print(iris.keys())  # Виводимо заголовки даних датасету

print(iris['feature_names'])  # Виводимо назви ознак

print(iris['data'])  # Виводимо значення ознак

print(iris['target_names'])  # Виводимо наявнi мiтки класiв

print(iris['target'])  # Виводимо номери класiв

# В тестовому прикладi обираємо тiльки 2 ознаки iз х4
X = iris.data[:, 2:4]
y = iris.target

# Розбиваємо датасет на вибiрки параметри train_size або test_size задають вiдсоток розбиття та є комплементарними
# один до одного: train_size + test_size = 1 тож достатньо задати тiльки один iз цих параметрiв параметр random_state
# слiд задавати цiлим числом для вiдтворення результатiв пiд час множинних запускiв алгоритму
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, random_state=0)

# Перевiряємо розмiрностi отриманих вибiрок
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

cmap = ["darkorange", "c", "darkblue"] # Задаємо кольорову палiтру для вiзуалiзацiї

# Створюємо та навчаємо класифiкатор
knn = neighbors.KNeighborsClassifier(k)
knn.fit(x_train, y_train)

# Вiзуалiзацiя точок даних в виглядi точкової дiаграми на площинi кожен клас представлений власним кольором iз
# заданої палiтри
_, ax = plt.subplots()
sns.scatterplot(
    x=X[:, 0],
    y=X[:, 1],
    hue=iris.target_names[y],
    palette=cmap,
    alpha=1.0,
    edgecolor="black", )
plt.title("3 - Class classification ( k = %i )" % k)
plt.show()

# Проводимо оцiнювання точностi на тестовiй вибiрцi виводимо результат
score = knn.score(x_test, y_test)
print(score)
