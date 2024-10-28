import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score

# Завантажуємо датасет
wine = load_wine()
X = wine.data
y = wine.target

# Обираємо індекси для вибірки
S1, S2, S3 = 2, 13, 3
X_train_1, y_train_1 = X[y == 0][S1:S1 + 40], y[y == 0][S1:S1 + 40]
X_train_2, y_train_2 = X[y == 1][S2:S2 + 40], y[y == 1][S2:S2 + 40]
X_train_3, y_train_3 = X[y == 2][S3:S3 + 40], y[y == 2][S3:S3 + 40]

# Формуємо навчальну вибірку
X_train = np.vstack((X_train_1, X_train_2, X_train_3))
y_train = np.concatenate((y_train_1, y_train_2, y_train_3))

# Налаштовуємо перехресне затверджування з меншою кількістю розрізів для швидшого виконання
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Діапазон значень для n_estimators (менший для пришвидшення обчислень)
n_estimators_range = range(2, 31)

# Ініціалізація словників для зберігання результатів
results_adaboost = {'accuracy': [], 'precision': [], 'recall': []}
results_randomforest = {'accuracy': [], 'precision': [], 'recall': []}

# Оцінка метрик для AdaBoost та RandomForest
for n_estimators in n_estimators_range:
    # AdaBoost з використанням алгоритму SAMME
    adaboost_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='log_loss', random_state=42),
                                      n_estimators=n_estimators, algorithm='SAMME', random_state=42)
    accuracy_scores = cross_val_score(adaboost_clf, X_train, y_train, cv=kf, scoring='accuracy')
    precision_scores = cross_val_score(adaboost_clf, X_train, y_train, cv=kf,
                                       scoring=make_scorer(precision_score, average='macro', zero_division=0))
    recall_scores = cross_val_score(adaboost_clf, X_train, y_train, cv=kf,
                                    scoring=make_scorer(recall_score, average='macro'))
    results_adaboost['accuracy'].append(np.mean(accuracy_scores))
    results_adaboost['precision'].append(np.mean(precision_scores))
    results_adaboost['recall'].append(np.mean(recall_scores))

    # RandomForest
    randomforest_clf = RandomForestClassifier(n_estimators=n_estimators, criterion='log_loss', random_state=42)
    accuracy_scores = cross_val_score(randomforest_clf, X_train, y_train, cv=kf, scoring='accuracy')
    precision_scores = cross_val_score(randomforest_clf, X_train, y_train, cv=kf,
                                       scoring=make_scorer(precision_score, average='macro', zero_division=0))
    recall_scores = cross_val_score(randomforest_clf, X_train, y_train, cv=kf,
                                    scoring=make_scorer(recall_score, average='macro'))
    results_randomforest['accuracy'].append(np.mean(accuracy_scores))
    results_randomforest['precision'].append(np.mean(precision_scores))
    results_randomforest['recall'].append(np.mean(recall_scores))

# Візуалізація результатів AdaBoost
plt.figure(figsize=(18, 5))
for i, metric in enumerate(results_adaboost):
    plt.subplot(1, 3, i + 1)
    plt.plot(n_estimators_range, results_adaboost[metric], label='AdaBoost')
    plt.title(f'AdaBoost - {metric}')
    plt.xlabel('n_estimators')
    plt.ylabel(metric)
plt.tight_layout()
plt.show()

# Візуалізація результатів RandomForest
plt.figure(figsize=(18, 5))
for i, metric in enumerate(results_randomforest):
    plt.subplot(1, 3, i + 1)
    plt.plot(n_estimators_range, results_randomforest[metric], label='RandomForest', color='orange')
    plt.title(f'RandomForest - {metric}')
    plt.xlabel('n_estimators')
    plt.ylabel(metric)
plt.tight_layout()
plt.show()

# Визначення найкращого значення n_estimators на основі метрики точність

# AdaBoost
best_n_estimators_adaboost = n_estimators_range[np.argmax(results_adaboost['accuracy'])]
best_accuracy_adaboost = max(results_adaboost['accuracy'])

# RandomForest
best_n_estimators_randomforest = n_estimators_range[np.argmax(results_randomforest['accuracy'])]
best_accuracy_randomforest = max(results_randomforest['accuracy'])

print(f"\nНайкраще значення точності для AdaBoost: {best_accuracy_adaboost}, n_estimators={best_n_estimators_adaboost}")
print(
    f"Найкраще значення точності для RandomForest: {best_accuracy_randomforest}, n_estimators={best_n_estimators_randomforest}")
