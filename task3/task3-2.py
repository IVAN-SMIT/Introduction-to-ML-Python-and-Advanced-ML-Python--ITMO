import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from colorama import Fore, Style
import pyfiglet

# Загрузка данных
df = pd.read_csv('task3/fish_train.csv')

# Разделение на обучающую и тестовую выборки с сохранением долей видов рыб
train_df, test_df = train_test_split(df, test_size=0.2, random_state=47, stratify=df['Species'])

# 1. Вычисление выборочного среднего колонки Width в тренировочной выборке
mean_width_train = train_df['Width'].mean()
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)
print(f"Выборочное среднее Width (тренировочная выборка): {mean_width_train:.3f}")
# Ответ: 4.054

# Подготовка данных для базовой модели (удаляем категориальные признаки)
X_train_base = train_df.drop(['Species', 'Weight'], axis=1)
X_test_base = test_df.drop(['Species', 'Weight'], axis=1)
y_train = train_df['Weight']
y_test = test_df['Weight']

# Обучение базовой модели линейной регрессии
lr_base = LinearRegression().fit(X_train_base, y_train)
y_pred_base = lr_base.predict(X_test_base)
r2_base = r2_score(y_test, y_pred_base)
print(f"R2 базовой модели: {r2_base:.3f}")
# Ответ: 0.854

# 2. Добавление предварительной обработки признаков - PCA

# Вычисляем корреляционную матрицу для числовых признаков
numeric_cols = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
corr_matrix = X_train_base[numeric_cols].corr().abs()

# Находим три наиболее коррелированных признака
# Убираем диагональ (корреляция с самим собой)
corr_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))

# Сортируем по убыванию корреляции
corr_pairs.sort(key=lambda x: x[2], reverse=True)

# Берем три пары с самой высокой корреляцией и находим уникальные признаки
top_correlated_features = set()
for pair in corr_pairs[:3]:  # берем первые 3 пары
    top_correlated_features.add(pair[0])
    top_correlated_features.add(pair[1])
    if len(top_correlated_features) >= 3:
        break

# Если получилось больше 3, берем первые 3 с самыми высокими корреляциями
if len(top_correlated_features) > 3:
    feature_corr_sum = {}
    for feature in top_correlated_features:
        total_corr = 0
        for pair in corr_pairs:
            if feature in pair[:2]:
                total_corr += pair[2]
        feature_corr_sum[feature] = total_corr
    
    # Сортируем по суммарной корреляции и берем топ-3
    sorted_features = sorted(feature_corr_sum.items(), key=lambda x: x[1], reverse=True)
    top_correlated_features = [feat[0] for feat in sorted_features[:3]]

# Сортируем в лексикографическом порядке
top_correlated_features_sorted = sorted(list(top_correlated_features))
print(f"Тройка наиболее коррелированных признаков: {', '.join(top_correlated_features_sorted)}")
# Ответ: Length1, Length2, Length3

# Применяем PCA к трем наиболее коррелированным признакам
pca_features = top_correlated_features_sorted
pca = PCA(n_components=3)
X_train_pca_full = pca.fit_transform(X_train_base[pca_features])
X_test_pca_full = pca.transform(X_test_base[pca_features])

# Доля объясненной дисперсии первой главной компоненты
explained_variance_ratio_first = pca.explained_variance_ratio_[0]
print(f"Доля объясненной дисперсии первой ГК: {explained_variance_ratio_first:.3f}")
# Ответ: 0.987

# Создаем новые датасеты, заменяя три коррелированных признака на первую ГК
X_train_pca = X_train_base.drop(pca_features, axis=1).copy()
X_test_pca = X_test_base.drop(pca_features, axis=1).copy()

X_train_pca['Lengths'] = X_train_pca_full[:, 0]
X_test_pca['Lengths'] = X_test_pca_full[:, 0]

# Обучаем модель линейной регрессии с новым признаком
lr_pca = LinearRegression().fit(X_train_pca, y_train)
y_pred_pca = lr_pca.predict(X_test_pca)
r2_pca = r2_score(y_test, y_pred_pca)
print(f"R2 модели после PCA: {r2_pca:.3f}")
# Ответ: 0.858

# Модификация признаков - возведение в куб
X_train_cubed = X_train_pca.copy()
X_test_cubed = X_test_pca.copy()

# Возводим в куб признаки Width, Height, Lengths
for col in ['Width', 'Height', 'Lengths']:
    X_train_cubed[col] = X_train_cubed[col] ** 3
    X_test_cubed[col] = X_test_cubed[col] ** 3

# Вычисляем выборочное среднее Width после возведения в куб
mean_width_cubed = X_train_cubed['Width'].mean()
print(f"Среднее Width после возведения в куб: {mean_width_cubed:.3f}")
# Ответ: 83.229

# Построение графиков зависимости Weight от Width до и после преобразования
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train_base['Width'], y_train, alpha=0.7)
plt.title('Зависимость Weight от Width (до преобразования)')
plt.xlabel('Width')
plt.ylabel('Weight')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_train_cubed['Width'], y_train, alpha=0.7)
plt.title('Зависимость Weight от Width³ (после преобразования)')
plt.xlabel('Width³')
plt.ylabel('Weight')
plt.grid(True)

plt.tight_layout()
plt.show()

# Обучаем модель линейной регрессии с кубическими признаками
lr_cubed = LinearRegression().fit(X_train_cubed, y_train)
y_pred_cubed = lr_cubed.predict(X_test_cubed)
r2_cubed = r2_score(y_test, y_pred_cubed)
print(f"R2 модели после возведения в куб: {r2_cubed:.3f}")
# Ответ: 0.872

# Добавление категориальных признаков с one-hot кодированием
# Возвращаем категориальные признаки
train_with_species = train_df.copy()
test_with_species = test_df.copy()

# Создаем копии для one-hot кодирования
X_train_with_cat = X_train_cubed.copy()
X_test_with_cat = X_test_cubed.copy()

# Добавляем one-hot кодированные признаки видов рыб
species_train_dummies = pd.get_dummies(train_with_species['Species'], prefix='Species')
species_test_dummies = pd.get_dummies(test_with_species['Species'], prefix='Species')

# Добавляем их к нашим датасетам
X_train_final = pd.concat([X_train_with_cat.reset_index(drop=True), species_train_dummies.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_with_cat.reset_index(drop=True), species_test_dummies.reset_index(drop=True)], axis=1)

# Обучаем модель
lr_with_cat = LinearRegression().fit(X_train_final, y_train)
y_pred_with_cat = lr_with_cat.predict(X_test_final)
r2_with_cat = r2_score(y_test, y_pred_with_cat)
print(f"R2 модели с категориальными признаками: {r2_with_cat:.3f}")
# Ответ: 0.946

# Кодируем категориальные признаки с drop_first=True
species_train_dummies_drop = pd.get_dummies(train_with_species['Species'], prefix='Species', drop_first=True)
species_test_dummies_drop = pd.get_dummies(test_with_species['Species'], prefix='Species', drop_first=True)

# Создаем новые датасеты
X_train_drop_first = pd.concat([X_train_with_cat.reset_index(drop=True), species_train_dummies_drop.reset_index(drop=True)], axis=1)
X_test_drop_first = pd.concat([X_test_with_cat.reset_index(drop=True), species_test_dummies_drop.reset_index(drop=True)], axis=1)

# Обучаем модель
lr_drop_first = LinearRegression().fit(X_train_drop_first, y_train)
y_pred_drop_first = lr_drop_first.predict(X_test_drop_first)
r2_drop_first = r2_score(y_test, y_pred_drop_first)
print(f"R2 модели с drop_first=True: {r2_drop_first:.3f}")
# Ответ: 0.946

print("BY IVAN SMIT, 2025")