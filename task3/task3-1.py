import numpy as np
from sklearn.linear_model import LinearRegression
from colorama import Fore, Style
import pyfiglet

# Данные (X - количество людей в очереди, Y - длительность ожидания)
data = np.array([
    [1,  4],
    [16, 44],
    [4,  7],
    [25, 54],
    [13, 24],
    [5,  14],
    [22, 45],
    [2,  4],
    [18, 41],
    [8,  25]
])

# Разделяем на признаки X и целевую переменную Y
X = data[:, 0].reshape(-1, 1)  # sklearn требует 2D массив для X
Y = data[:, 1]

# Выборочные средние
mean_X = np.mean(X)
mean_Y = np.mean(Y)

print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), font="starwars") + Style.RESET_ALL)
print(f"Выборочное среднее X: {mean_X:.6f}")
print(f"Выборочное среднее Y: {mean_Y:.6f}")

# Создаём и обучаем модель линейной регрессии
model = LinearRegression().fit(X, Y)

# Коэффициенты модели
theta1 = model.coef_[0]   # наклон
theta0 = model.intercept_ # сдвиг

print(f"Коэффициент θ1: {theta1:.6f}")
print(f"Коэффициент θ0: {theta0:.6f}")

# Прогнозы модели
Y_pred = model.predict(X)

# Вычисляем R² вручную (как требует задание — не используем model.score)
SS_res = np.sum((Y - Y_pred) ** 2)      # сумма квадратов остатков
SS_tot = np.sum((Y - mean_Y) ** 2)      # общая сумма квадратов
R_squared = 1 - (SS_res / SS_tot)

print(f"R² статистика: {R_squared:.6f}")

print("BY IVAN SMIT, 2025")
