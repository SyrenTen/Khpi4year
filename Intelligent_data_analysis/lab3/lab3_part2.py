import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score

# Завантажуємо датасет
wine = load_wine()
X = wine.data
y = wine.target

# Обираємо індекси S1, S2, S3 для вибірки
S1, S2, S3 = 2, 13, 3
X_train_1, y_train_1 = X[y == 0][S1:S1+40], y[y == 0][S1:S1+40]
X_train_2, y_train_2 = X[y == 1][S2:S2+40], y[y == 1][S2:S2+40]
X_train_3, y_train_3 = X[y == 2][S3:S3+40], y[y == 2][S3:S3+40]

# Формуємо навчальну вибірку
X_train = np.vstack((X_train_1, X_train_2, X_train_3))
y_train = np.concatenate((y_train_1, y_train_2, y_train_3))

# Визначаємо діапазони значень для max_depth та min_samples_split
max_depth_range = range(1, 11)
min_samples_split_range = range(2, 11)

# Створюємо словник параметрів для сіткового пошуку
param_grid = {
    'max_depth': max_depth_range,
    'min_samples_split': min_samples_split_range,
    'criterion': ['log_loss']
}

# Налаштовуємо перехресне затверджування
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Визначаємо класифікатор
clf = DecisionTreeClassifier(random_state=42)

# Зберігаємо результати для трьох метрик: точність, влучність та повнота
scores = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro')
}
results = {score: [] for score in scores}

for score_name, scoring_function in scores.items():
    grid_search = GridSearchCV(clf, param_grid, cv=kf, scoring=scoring_function, return_train_score=True)
    grid_search.fit(X_train, y_train)
    results[score_name] = grid_search.cv_results_['mean_test_score'].reshape(len(max_depth_range), len(min_samples_split_range))

    # Виводимо результати у консоль для кожної метрики (п. 2.2)
    print(f"\nМатриця для метрики {score_name}:")
    print(results[score_name])

# Візуалізація теплових карт
plt.figure(figsize=(18, 5))
for i, score_name in enumerate(scores):
    plt.subplot(1, 3, i+1)
    sns.heatmap(results[score_name], annot=True, xticklabels=min_samples_split_range, yticklabels=max_depth_range, cmap='viridis')
    plt.title(f'Heatmap for {score_name}')
    plt.xlabel('min_samples_split')
    plt.ylabel('max_depth')
plt.tight_layout()
plt.show()

# Налаштовуємо GridSearchCV для визначення найкращих параметрів за точністю
grid_search_accuracy = GridSearchCV(clf, param_grid, cv=kf, scoring='accuracy', return_train_score=True)
grid_search_accuracy.fit(X_train, y_train)
best_params = grid_search_accuracy.best_params_
best_score = grid_search_accuracy.best_score_

# Виводимо найкращі параметри
best_params, best_score

# Дані для звіту
report_data = {
    'Диапазон max_depth': list(max_depth_range),
    'Диапазон min_samples_split': list(min_samples_split_range),
    'Параметри крос-валідації': {
        'n_splits': kf.n_splits,
        'shuffle': kf.shuffle,
        'random_state': kf.random_state
    },
    'Найкраща точність': best_score,
    'Найкращі параметри': best_params
}

# Виведення даних для звіту
report_data
