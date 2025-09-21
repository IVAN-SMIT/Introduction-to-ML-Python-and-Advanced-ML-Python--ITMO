import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style
import pyfiglet

# Загрузка данных
df = pd.read_csv('task4/adult_data_train.csv')

# --- Шаг 0: Удаление признаков education и marital-status ---
df = df.drop(['education', 'marital-status'], axis=1)

# Определение числовых и нечисловых признаков
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if 'label' in categorical_cols:
    categorical_cols.remove('label')
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)
print(f"Количество нечисловых признаков: {len(categorical_cols)}")
# Ответ: 7

# --- Построение гистограммы распределения объектов по классам ---
plt.figure(figsize=(6, 4))
df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Распределение объектов по классам')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Вычисление доли объектов класса 0
class_0_ratio = (df['label'] == 0).mean()
print(f"Доля объектов класса 0: {class_0_ratio:.3f}")
# Ответ: 0.759

# --- 1. Построение базовой модели ---

# Отбираем только числовые признаки
X_num = df[numeric_cols]
y = df['label']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y, test_size=0.2, random_state=5, stratify=y
)

# Выборочное среднее fnlwgt в тренировочной выборке
mean_fnlwgt_train = X_train['fnlwgt'].mean()
print(f"Среднее fnlwgt (тренировка): {mean_fnlwgt_train:.3f}")
# Ответ: 190368.243

# Обучение базовой модели KNN
knn_base = KNeighborsClassifier()
knn_base.fit(X_train, y_train)
y_pred_base = knn_base.predict(X_test)

f1_base = f1_score(y_test, y_pred_base)
print(f"F1-score базовой модели: {f1_base:.3f}")
# Ответ: 0.583

# --- Масштабирование признаков ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Среднее fnlwgt после масштабирования (на основе масштабированных данных)
# fnlwgt — это первый столбец в numeric_cols, если не меняли порядок
fnlwgt_col_index = X_train.columns.get_loc('fnlwgt')
mean_fnlwgt_scaled = X_train_scaled[:, fnlwgt_col_index].mean()
print(f"Среднее fnlwgt после масштабирования: {mean_fnlwgt_scaled:.3f}")
# Ответ: 0.499

# Обучение модели KNN на масштабированных данных
knn_scaled = KNeighborsClassifier()
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

f1_scaled = f1_score(y_test, y_pred_scaled)
print(f"F1-score после масштабирования: {f1_scaled}")
# Ответ: 0.627

# --- 2. Работа с нечисловыми признаками ---

# --- Визуализация нечисловых признаков ---
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    if i < len(axes):
        value_counts = df[col].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
        axes[i].set_title(f'Распределение значений: {col}')
        axes[i].tick_params(axis='x', rotation=45)

for j in range(len(categorical_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# --- Удаление пропущенных значений ---
rows_with_question_mark = (df == '?').any(axis=1).sum()
print(f"Число строк с '?': {rows_with_question_mark}")
# Ответ: 2399

df_cleaned = df.replace('?', np.nan).dropna().reset_index(drop=True)

# Разделяем на признаки и целевую переменную ДО one-hot кодирования
X_cleaned = df_cleaned.drop('label', axis=1)
y_cleaned = df_cleaned['label']

# Разделение на train/test
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_cleaned, y_cleaned, test_size=0.2, random_state=5, stratify=y_cleaned
)

# Применяем one-hot кодирование ОТДЕЛЬНО к train и test
X_train_clean_dummies = pd.get_dummies(X_train_clean, columns=categorical_cols, drop_first=True)
X_test_clean_dummies = pd.get_dummies(X_test_clean, columns=categorical_cols, drop_first=True)

# Выравнивание столбцов
for col in X_train_clean_dummies.columns:
    if col not in X_test_clean_dummies.columns:
        X_test_clean_dummies[col] = 0
for col in X_test_clean_dummies.columns:
    if col not in X_train_clean_dummies.columns:
        X_train_clean_dummies[col] = 0

X_test_clean_dummies = X_test_clean_dummies[X_train_clean_dummies.columns]

# Масштабирование
scaler_clean = MinMaxScaler()
X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean_dummies)
X_test_clean_scaled = scaler_clean.transform(X_test_clean_dummies)

# Обучение модели
knn_clean = KNeighborsClassifier()
knn_clean.fit(X_train_clean_scaled, y_train_clean)
y_pred_clean = knn_clean.predict(X_test_clean_scaled)

f1_clean = f1_score(y_test_clean, y_pred_clean)
print(f"Общее число признаков после one-hot: {X_train_clean_dummies.shape[1]}")
# Ответ: 87
print(f"F1-score после удаления пропусков: {f1_clean:.3f}")
# Ответ: 0.644

# --- Заполнение пропущенных значений ---
df_filled = df.copy()
df_filled = df_filled.replace('?', np.nan)

# Заполняем модой только категориальные признаки
for col in categorical_cols:
    mode_value = df_filled[col].mode()[0]
    df_filled[col] = df_filled[col].fillna(mode_value)

# Разделяем на признаки и целевую переменную
X_filled = df_filled.drop('label', axis=1)
y_filled = df_filled['label']

# Разделение на train/test
X_train_fill, X_test_fill, y_train_fill, y_test_fill = train_test_split(
    X_filled, y_filled, test_size=0.2, random_state=5, stratify=y_filled  # Исправлено: 0.2 вместо 00.2
)

# One-hot кодирование
X_train_fill_dummies = pd.get_dummies(X_train_fill, columns=categorical_cols, drop_first=True)
X_test_fill_dummies = pd.get_dummies(X_test_fill, columns=categorical_cols, drop_first=True)

# Выравнивание столбцов
for col in X_train_fill_dummies.columns:
    if col not in X_test_fill_dummies.columns:
        X_test_fill_dummies[col] = 0
for col in X_test_fill_dummies.columns:
    if col not in X_train_fill_dummies.columns:
        X_train_fill_dummies[col] = 0

X_test_fill_dummies = X_test_fill_dummies[X_train_fill_dummies.columns]

# Масштабирование
scaler_fill = MinMaxScaler()
X_train_fill_scaled = scaler_fill.fit_transform(X_train_fill_dummies)
X_test_fill_scaled = scaler_fill.transform(X_test_fill_dummies)

# Обучение модели
knn_fill = KNeighborsClassifier()
knn_fill.fit(X_train_fill_scaled, y_train_fill)
y_pred_fill = knn_fill.predict(X_test_fill_scaled)

f1_fill = f1_score(y_test_fill, y_pred_fill)
print(f"F1-score после заполнения пропусков: {f1_fill:.3f}")
# Ответ: 0.641

print("BY IVAN SMIT, 2025")