# Це код симплекс-методу

import numpy as np


def simplexmethod(c, A, b):
    # Кількість обмежень і змінних
    num_const, num_var = A.shape

    # Формуємо розширену симплекс-таблицю
    tab = np.hstack([A, np.eye(num_const), b.reshape(-1, 1)])
    c_row = np.hstack([c, np.zeros(num_const + 1)])
    tab = np.vstack([tab, c_row])

    while np.any(tab[-1, :-1] < 0):  # Поки є від’ємні коефіцієнти в останньому рядку
        # Визначаємо розв’язуючий стовпець (з найменшим значенням у Z-рядку)
        pivot_col = np.argmin(tab[-1, :-1])

        # Визначаємо розв’язуючий рядок
        ratios = tab[:-1, -1] / tab[:-1, pivot_col]
        valid_ratios = np.where(tab[:-1, pivot_col] > 0, ratios, np.inf)
        pivot_row = np.argmin(valid_ratios)

        # Нормалізуємо розв’язуючий рядок
        tab[pivot_row] /= tab[pivot_row, pivot_col]

        # Виконуємо виключення Гауса для всіх рядків
        for i in range(tab.shape[0]):
            if i != pivot_row:
                tab[i] -= tab[i, pivot_col] * tab[pivot_row]

    # Оптимальні значення змінних
    result = np.zeros(num_var)
    for i in range(num_var):
        col = tab[:-1, i]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1:
            row = np.argmax(col)
            result[i] = tab[row, -1]

    return result, tab[-1, -1]


# 12 варіант
c = np.array([-6, -8, 1, 3])
A = np.array([[2, 5, 1, 2], [12, 6, 2, 1]])
b = np.array([20, 72])

result, optimal_value = simplexmethod(c, A, b)

print(f'Оптимальні знач змінних (x1, x2, x3, x4): {result}')
print(f'Оптимальне знач функції: {optimal_value}')
