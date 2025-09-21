import gymnasium as gym  
import numpy as np
import time
from colorama import Fore, Style
import pyfiglet
from IPython.display import clear_output
from tqdm import tqdm

# Параметры обучения
epsilon = 0.05
gamma = 0.8
random_seed = 10
lr_rate = 0.9
time_delay = 1

# Генерация случайной карты
def generate_random_map(size, p, sd):
    valid = False
    np.random.seed(sd)

    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if res[r_new][c_new] not in '#H':
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]

# Создаем карту
random_map = generate_random_map(size=6, p=0.8, sd=random_seed)
env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False, render_mode="ansi")  
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)
print("Ваша карта")
env.reset()  
env.render()

# Функция выбора действия
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.random.choice(np.argwhere(Q[state, :] == np.max(Q[state, :])).flatten())
    return action

# Обновление Q-значений
def learn(state, state2, reward, action, done):
    if done:
        Q[state, action] += lr_rate * (reward - Q[state, action])
    else:
        Q[state, action] += lr_rate * (reward + gamma * np.max(Q[state2, :]) - Q[state, action])

# Инициализация
np.random.seed(random_seed)
total_games = 10000
max_steps = 100
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Список для хранения результатов игр
wins = []
consecutive_wins = 0
first_five_in_a_row = None

# Основной цикл обучения
for game in tqdm(range(total_games)):
    state, _ = env.reset()  # ← ИЗМЕНЕНО: возвращает (state, info)
    t = 0
    game_win = False
    while t < max_steps:
        t += 1
        action = choose_action(state)
        state2, reward, terminated, truncated, info = env.step(action)  # ← ИЗМЕНЕНО: 5 значений
        done = terminated or truncated  # ← ИЗМЕНЕНО: done = terminated OR truncated

        if t == max_steps:
            done = True

        learn(state, state2, reward, action, done)
        state = state2

        if done and reward == 1:
            game_win = True
            consecutive_wins += 1
            if first_five_in_a_row is None and consecutive_wins >= 5:
                first_five_in_a_row = game + 1
            break
        elif done:
            consecutive_wins = 0
            break

    wins.append(game_win)

# Подсчет общего числа побед
total_wins = sum(wins)

# Вывод ответов
print("Количество побед в серии из 10 000 игр: ", total_wins)
print("Пять побед подряд впервые было одержано в игре ", first_five_in_a_row)

# После обучения — демонстрация одной игры
def choose_action_one_game(state):
    action = np.random.choice(np.argwhere(Q[state, :] == np.max(Q[state, :])).flatten())
    return action

states = []
t = 0
state, _ = env.reset() 
wn = 0
while t < 100:
    env.render()
    time.sleep(time_delay)
    clear_output(wait=True)
    action = choose_action_one_game(state)
    state2, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  
    states.append(state)
    state = state2
    t += 1
    if done and reward == 1:
        wn = 1
    if done:
        break

if wn == 1:
    print("!!!Победа!!!")

# Визуализация пути
def make_maze_pic(maze):
    maze_pic = []
    for i in range(len(maze)):
        row = []
        for j in range(len(maze[i])):
            if maze[i][j] == 'S' or maze[i][j] == 'F' or maze[i][j] == 'G':
                row.append(0)
            elif maze[i][j] == 'H':
                row.append(1)
        maze_pic.append(row)
    return np.array(maze_pic)

maze_pic = make_maze_pic(random_map)
nrows, ncols = maze_pic.shape

rw = np.remainder(states, nrows)
cl = np.floor_divide(states, nrows)

if wn == 1:
    rw = np.append(rw, [nrows - 1])
    cl = np.append(cl, [ncols - 1])

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(1, 1, tight_layout=True)
ax1.clear()
ax1.set_xticks(np.arange(0.5, nrows, step=1))
ax1.set_xticklabels([])
ax1.set_yticks(np.arange(0.5, ncols, step=1))
ax1.set_yticklabels([])
ax1.grid(True)
ax1.plot([0], [0], "gs", markersize=40)
ax1.text(0, 0.2, "Start", ha="center", va="center", color="white", fontsize=12)
ax1.plot([nrows-1], [ncols-1], "rs", markersize=40)
ax1.text(nrows-1, ncols-1+0.2, "Finish", ha="center", va="center", color="white", fontsize=12)
ax1.plot(rw, cl, ls='-', color='blue')
ax1.plot(rw, cl, "bo")
ax1.imshow(maze_pic, cmap="binary")
plt.show()

print("BY IVAN SMIT, 2025")
