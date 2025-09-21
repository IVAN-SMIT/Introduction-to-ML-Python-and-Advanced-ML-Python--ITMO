import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from colorama import Fore, Style
import pyfiglet

# Загрузка данных
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X': [91, 13, 41, 27, 96, 21, 64, 13, 37, 74],
    'Y': [24, 45, 38, 23, 79, 85, 14, 21, 39, 87],
    'Class': [0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Новая точка — передаём как массив, чтобы избежать warnings
new_point = np.array([[45, 29]])

# Признаки и целевая переменная — тоже как массивы
X = df[['X', 'Y']].values  # <-- ключевое изменение: .values
y = df['Class'].values     # <-- ключевое изменение: .values

# --- Евклидова метрика (p=2) ---
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)
print("=== Евклидова метрика ===")

# Ближайший сосед (k=1)
knn_euc_1 = KNeighborsClassifier(n_neighbors=1, p=2)
knn_euc_1.fit(X, y)
dist_euc, idx_euc = knn_euc_1.kneighbors(new_point)
nearest_dist_euc = dist_euc[0][0]
nearest_id_euc = df.iloc[idx_euc[0][0]]['id']

print(f"Расстояние до ближайшего соседа: {nearest_dist_euc:.3f}")

# Три ближайших соседа (k=3)
knn_euc_3 = KNeighborsClassifier(n_neighbors=3, p=2)
knn_euc_3.fit(X, y)
distances_euc, indices_euc = knn_euc_3.kneighbors(new_point)

# Получаем id трех ближайших соседей
ids_euc = df.iloc[indices_euc[0]]['id'].tolist()
ids_str_euc = ','.join(map(str, ids_euc))
print(f"Идентификаторы трех ближайших точек: {ids_str_euc}")

# Предсказываем класс для k=3
pred_class_euc = knn_euc_3.predict(new_point)[0]
print(f"Класс нового объекта (k=3): {pred_class_euc}")

# --- Манхэттенская метрика (p=1) ---
print("\n=== Манхэттенская метрика ===")

# Ближайший сосед
knn_man_1 = KNeighborsClassifier(n_neighbors=1, p=1)
knn_man_1.fit(X, y)
dist_man, idx_man = knn_man_1.kneighbors(new_point)
nearest_dist_man = dist_man[0][0]
nearest_id_man = df.iloc[idx_man[0][0]]['id']

print(f"Расстояние до ближайшего соседа: {nearest_dist_man:.3f}")

# Три ближайших соседа
knn_man_3 = KNeighborsClassifier(n_neighbors=3, p=1)
knn_man_3.fit(X, y)
distances_man, indices_man = knn_man_3.kneighbors(new_point)

# Получаем id трех ближайших соседей
ids_man = df.iloc[indices_man[0]]['id'].tolist()
ids_str_man = ','.join(map(str, ids_man))
print(f"Идентификаторы трех ближайших точек: {ids_str_man}")

# Предсказываем класс для k=3
pred_class_man = knn_man_3.predict(new_point)[0]
print(f"Класс нового объекта (k=3): {pred_class_man}")

print("BY IVAN SMIT, 2025")
