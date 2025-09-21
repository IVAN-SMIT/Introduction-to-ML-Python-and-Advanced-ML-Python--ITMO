import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
import pyfiglet

# Загрузка данных с указанием разделителя ";"
X_reduced = pd.read_csv('task2/X_reduced_561.csv', sep=';', header=None).values.astype(float)
X_loadings = pd.read_csv('task2/X_loadings_561.csv', sep=';', header=None).values.astype(float)

# Восстановление исходного изображения: X_original ≈ X_reduced @ X_loadings.T
X_restored = X_reduced @ X_loadings.T  # Результат: матрица (561, 561)

# Отображение изображения
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), font="starwars") + Style.RESET_ALL)
plt.figure(figsize=(8, 8))
plt.imshow(X_restored, cmap='gray')
plt.axis('off')  # Скрыть оси для лучшего вида логотипа
plt.title('Восстановленный логотип новогоднего корпоратива')
plt.show()

print("BY IVAN SMIT, 2025")