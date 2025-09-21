import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from colorama import Fore, Style
import pyfiglet

# Загрузка данных
train_df = pd.read_csv('task3/fish_train.csv')
test_df = pd.read_csv('task3/fish_reserved.csv')

# --- Шаг 1: Предобработка обучающих данных ---
X_train_num = train_df.drop(['Species', 'Weight'], axis=1)
y_train = train_df['Weight']

# Применяем PCA к Length1, Length2, Length3
pca_features = ['Length1', 'Length2', 'Length3']
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_num[pca_features])

# Создаем новый датасет с заменой Length1-3 на Lengths (первая ГК)
X_train_processed = X_train_num.drop(pca_features, axis=1).copy()
X_train_processed['Lengths'] = X_train_pca[:, 0]

# Возведение в куб: Width, Height, Lengths
for col in ['Width', 'Height', 'Lengths']:
    X_train_processed[col] = X_train_processed[col] ** 3

# One-hot кодирование Species БЕЗ drop_first=True (важно!)
species_dummies = pd.get_dummies(train_df['Species'], prefix='Species')
X_train_final = pd.concat([X_train_processed.reset_index(drop=True), species_dummies.reset_index(drop=True)], axis=1)

# Обучаем модель
model = LinearRegression().fit(X_train_final, y_train)

# --- Шаг 2: Предобработка тестовых данных ---
X_test_num = test_df.drop(['Species'], axis=1)

# Применяем PCA (тот же объект)
X_test_pca = pca.transform(X_test_num[pca_features])

# Создаем аналогичный датасет
X_test_processed = X_test_num.drop(pca_features, axis=1).copy()
X_test_processed['Lengths'] = X_test_pca[:, 0]

# Возведение в куб
for col in ['Width', 'Height', 'Lengths']:
    X_test_processed[col] = X_test_processed[col] ** 3

# One-hot кодирование Species (без drop_first!)
species_dummies_test = pd.get_dummies(test_df['Species'], prefix='Species')

# Убедимся, что столбцы совпадают с обучающей выборкой
for col in X_train_final.columns:
    if col not in X_test_processed.columns and col not in species_dummies_test.columns:
        if col.startswith('Species_'):
            species_dummies_test[col] = 0

X_test_final = pd.concat([X_test_processed.reset_index(drop=True), species_dummies_test.reset_index(drop=True)], axis=1)

# Выравниваем порядок столбцов
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

# --- Шаг 3: Предсказания ---
predictions = model.predict(X_test_final)

# Выводим результат в виде списка
prediction_list = predictions.tolist()
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)
print(prediction_list)
print("BY IVAN SMIT, 2025")