import numpy as np
from colorama import Fore, Style
import pyfiglet

# Начальные значения
V_s1 = 0.0
V_s2 = 0.0
V_s3 = 0.0
V_s4 = 0.0

gamma = 0.8
tolerance = 1e-10
max_iter = 1000

for _ in range(max_iter):
    # Сохраняем старые значения для сравнения
    V_s1_old, V_s2_old, V_s3_old, V_s4_old = V_s1, V_s2, V_s3, V_s4
    
    # Обновляем V(s1)
    V_s1 = (
        0.2 * (0.1 * (2.0 + gamma * V_s2) + 0.9 * (3.0 + gamma * V_s3)) +
        0.8 * (5.0 + gamma * V_s3)
    )
    
    # Обновляем V(s2)
    V_s2 = 3.0 + gamma * V_s1
    
    # Обновляем V(s3)
    V_s3 = (
        0.1 * (-3.0 + gamma * V_s1) +
        0.9 * (0.2 * (1.0 + gamma * V_s3) + 0.8 * (6.0 + gamma * V_s4))
    )
    
    # Обновляем V(s4)
    V_s4 = (
        0.6 * (5.0 + gamma * V_s1) +
        0.4 * (-3.0 + gamma * V_s2)
    )
    
    # Проверка сходимости
    if (abs(V_s1 - V_s1_old) < tolerance and
        abs(V_s2 - V_s2_old) < tolerance and
        abs(V_s3 - V_s3_old) < tolerance and
        abs(V_s4 - V_s4_old) < tolerance):
        break

# Округляем до тысячных
V_s1 = round(V_s1, 3)
V_s2 = round(V_s2, 3)
V_s3 = round(V_s3, 3)
V_s4 = round(V_s4, 3)


print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)
print(f"V(s1) = {V_s1}")
print(f"V(s2) = {V_s2}")
print(f"V(s3) = {V_s3}")
print(f"V(s4) = {V_s4}")


print("BY IVAN SMIT, 2025")
