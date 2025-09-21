import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from colorama import Fore, Style
import pyfiglet

# Загрузка обучающего датасета
df_train = pd.read_csv('task4/adult_data_train.csv')

# --- Подготовка обучающих данных ---

# Удаляем признаки education и marital-status
df_train = df_train.drop(['education', 'marital-status'], axis=1)

# Определяем категориальные и числовые столбцы
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
if 'label' in categorical_cols:
    categorical_cols.remove('label')

df_train = df_train.replace('?', np.nan)

# Заполняем пропуски модой в категориальных столбцах (на основе train!)
for col in categorical_cols:
    mode_value = df_train[col].mode()[0]
    df_train[col] = df_train[col].fillna(mode_value)

# Разделяем признаки и целевую переменную
X_train_full = df_train.drop('label', axis=1)
y_train_full = df_train['label']

# Применяем one-hot кодирование ТОЛЬКО к категориальным признакам
X_train_dummies = pd.get_dummies(X_train_full, columns=categorical_cols, drop_first=True)

# Обучаем scaler и модель на ВСЕХ данных
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_dummies)

# Обучаем финальную модель
final_model = KNeighborsClassifier()
final_model.fit(X_train_scaled, y_train_full)

# --- Загрузка и предобработка тестового датасета ---

# Загружаем зарезервированный датасет
df_test = pd.read_csv('task4/adult_data_reserved.csv')  # ← Убедитесь, что имя файла верное!

# Удаляем те же признаки
df_test = df_test.drop(['education', 'marital-status'], axis=1)

df_test = df_test.replace('?', np.nan)

# Заполняем пропуски модой (используем моду из ТРЕНИРОВОЧНОГО датасета!)
for col in categorical_cols:
    mode_value = df_train[col].mode()[0]  # Берем моду из train!
    df_test[col] = df_test[col].fillna(mode_value)

# Применяем one-hot кодирование
X_test_dummies = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# Выравниваем столбцы с обучающим набором
for col in X_train_dummies.columns:
    if col not in X_test_dummies.columns:
        X_test_dummies[col] = 0

# Приводим к одинаковому порядку столбцов
X_test_dummies = X_test_dummies[X_train_dummies.columns]

# Масштабируем тестовые данные (используем scaler, обученный на train)
X_test_scaled = scaler.transform(X_test_dummies)

# Получаем предсказания
predictions = final_model.predict(X_test_scaled)

# Выводим результат в виде списка
prediction_list = predictions.tolist()
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)
print(prediction_list)

print("BY IVAN SMIT, 2025")
