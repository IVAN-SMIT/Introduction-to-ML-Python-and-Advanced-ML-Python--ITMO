import numpy as np
from colorama import Fore, Style
import pyfiglet

# --- Исходные данные ---
spam_emails = 27
non_spam_emails = 20
total_emails = spam_emails + non_spam_emails

# Общее количество слов
F_spam = 99    
F_non_spam = 132 

# Частоты слов (из таблицы)
word_counts = {
    "Unlimited":   {"спам": 1,  "не спам": 4},
    "Purchase":    {"спам": 0,  "не спам": 0},
    "Million":     {"спам": 4,  "не спам": 2},
    "Membership":  {"спам": 8,  "не спам": 5},
    "Money":       {"спам": 8,  "не спам": 14},
    "Refund":      {"спам": 24, "не спам": 9},
    "Free":        {"спам": 6,  "не спам": 35},
    "Gift":        {"спам": 7,  "не спам": 44},
    "Bill":        {"спам": 41, "не спам": 3},
    "Offer":       {"спам": 0,  "не спам": 16}
}

# Письмо для классификации
email_words = ["Coupon", "Refund", "Gift", "Membership", "Prize", "Bill", "Purchase"]

# Количество уникальных слов в словаре (V)
V = len(word_counts)  # 10 уникальных слов

# --- 1. Вероятность того, что письмо является спамом ---
P_spam = spam_emails / total_emails
print(Fore.RED + pyfiglet.figlet_format(''.join([chr(code) for code in [98, 121, 32, 73, 49]]), 
          font="starwars") + Style.RESET_ALL)

print(f"Вероятность спама: {P_spam:.3f}")
# Ответ: 0.574

# --- 2. F(«спам») и F(«не спам») ---
print(f"F(спам): {F_spam:.3f}")
print(f"F(не спам): {F_non_spam:.3f}")
# Ответы могут быть неправильные, мб доработаю этот момент

# --- 3. Вероятность P(Класс = «спам» / Письмо) ---

# Априорные вероятности
log_P_spam = np.log(P_spam)
log_P_non_spam = np.log(1 - P_spam)

# Инициализируем логарифмы правдоподобий
log_likelihood_spam = 0.0
log_likelihood_non_spam = 0.0

# Для каждого слова в письме
for word in email_words:
    # Если слово есть в нашем словаре — берем его частоту
    if word in word_counts:
        count_spam = word_counts[word]["спам"]
        count_non_spam = word_counts[word]["не спам"]
    else:
        # Если слова нет — считаем частоту = 0
        count_spam = 0
        count_non_spam = 0
    
    # Применяем сглаживание Лапласа
    P_word_given_spam = (count_spam + 1) / (F_spam + V)
    P_word_given_non_spam = (count_non_spam + 1) / (F_non_spam + V)
    
    # Добавляем логарифмы вероятностей
    log_likelihood_spam += np.log(P_word_given_spam)
    log_likelihood_non_spam += np.log(P_word_given_non_spam)

# Вычисляем логарифмы апостериорных вероятностей
log_posterior_spam = log_P_spam + log_likelihood_spam
log_posterior_non_spam = log_P_non_spam + log_likelihood_non_spam

# Применяем softmax для получения вероятности спама
log_diff = log_posterior_non_spam - log_posterior_spam
P_spam_given_email = 1 / (1 + np.exp(log_diff))  # Это P(спам | email)

print(f"P(спам | письмо): {P_spam_given_email:.3f}")
# Ответ: 0.984


print("BY IVAN SMIT, 2025")
