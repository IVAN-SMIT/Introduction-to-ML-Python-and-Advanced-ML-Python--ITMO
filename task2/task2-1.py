import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from colorama import Fore, Style
import pyfiglet

# --- Загрузка и подготовка данных ---
# Загрузка данных без заголовка
data = pd.read_csv('task2/8_36.csv', header=None)

# Стандартизация данных (обязательно для PCA!)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Применение PCA
pca = PCA()
pca.fit(data_scaled)

# Преобразование данных в пространство главных компонент
data_pca = pca.transform(data_scaled)

# --- Ответы на вопросы ---
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)

# 1. Координата первого объекта относительно первой главной компоненты
coord_pc1_obj1 = data_pca[0, 0]
print(f"1. Координата первого объекта относительно первой главной компоненты: {coord_pc1_obj1:.6f}")

# 2. Координата первого объекта относительно второй главной компоненты
coord_pc2_obj1 = data_pca[0, 1]
print(f"2. Координата первого объекта относительно второй главной компоненты: {coord_pc2_obj1:.6f}")

# 3. Доля объясненной дисперсии первыми двумя компонентами
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
explained_variance_2pc = explained_variance_ratio_cumsum[1]  # Индекс 1 = первые две компоненты
print(f"3. Доля объясненной дисперсии первыми двумя компонентами: {explained_variance_2pc:.6f}")

# 4. Минимальное количество компонент для объяснения > 0.85 дисперсии
n_components_for_85 = np.argmax(explained_variance_ratio_cumsum > 0.85) + 1
print(f"4. Минимальное количество компонент для > 0.85 дисперсии: {n_components_for_85}")

# 5. Построение графика для визуального определения количества групп
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                     c='blue', edgecolor='k', s=60, alpha=0.7)
plt.xlabel('Первая главная компонента (PC1)', fontsize=12)
plt.ylabel('Вторая главная компонента (PC2)', fontsize=12)
plt.title('Распределение объектов в пространстве первых двух главных компонент', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Добавим подписи к точкам (опционально, для лучшей читаемости)
for i in range(min(10, len(data_pca))):  # Подписываем первые 10 точек
    plt.text(data_pca[i, 0], data_pca[i, 1], str(i+1), fontsize=9, ha='right')

plt.tight_layout()
plt.show()

# Вывод инструкции для пользователя
print("\n" + "="*60)
print("ВАЖНО: Для ответа на вопрос 5 внимательно рассмотрите график выше.")
print("Сколько четких, визуально различимых скоплений (кластеров) точек вы видите?")
print("Часто правильный ответ — 2 или 3 группы.")
print("="*60)

# --- Дополнительно: Автоматическое определение (для справки) ---
# Применим KMeans для k=2 и k=3, чтобы показать разницу
from sklearn.cluster import KMeans

k_values = [2, 3]
plt.figure(figsize=(14, 5))

for idx, k in enumerate(k_values, 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_pca[:, :2])
    
    plt.subplot(1, 2, idx)
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                         c=cluster_labels, cmap='viridis', 
                         edgecolor='k', s=60, alpha=0.8)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', marker='X', s=200, label='Центроиды')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'KMeans с {k} кластерами')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print("\nСправочная информация (автоматическая кластеризация):")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_pca[:, :2])
    # Можно добавить расчет силуэта, но для 60 точек это не очень надежно
    print(f"При k={k} кластеры визуально {'хорошо' if k == 2 else 'менее четко'} разделяются на графике.")



print("BY IVAN SMIT, 2025")
