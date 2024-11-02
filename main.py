import os
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

# Загружаем API-ключи из файла .env
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GEMINI_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Один или оба API-ключа не найдены! Проверьте файл .env.")

# Функция для отправки данных на Gemini API с ограничением одного запроса в минуту
def send_data_to_gemini(data):
    url = "https://api.gemini.com/v1/order/new"
    headers = {"X-GEMINI-APIKEY": GEMINI_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=data)
    time.sleep(60)  # Ожидаем 60 секунд, чтобы не превышать лимит

    if response.status_code == 200:
        print("Данные успешно отправлены на Gemini:", response.json())
    else:
        print("Ошибка отправки данных на Gemini:", response.status_code, response.text)

# Загрузка и обработка данных из датасетов
# Загрузка данных из новых датасетов
dataset_1 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет #1')
dataset_2 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет 2')
dataset_3 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет 3')
dataset_4 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет 4')


dataset_1['arrival_time'] = pd.to_datetime(dataset_1['arrival_time'], format='%H:%M:%S')
dataset_1['departure_time'] = pd.to_datetime(dataset_1['departure_time'], format='%H:%M:%S')
dataset_1['travel_time'] = (dataset_1['departure_time'] - dataset_1['arrival_time']).dt.total_seconds()

dataset_1['bus_stop'] = dataset_1['bus_stop'].astype(str)
dataset_2['stop_id'] = dataset_2['stop_id'].astype(str)

dataset_1 = dataset_1.merge(dataset_2, how='left', left_on='bus_stop', right_on='stop_id')

# Рассчитываем среднее время прибытия для каждой остановки
average_travel_times = dataset_1.groupby('bus_stop')['travel_time'].mean()

# Прогноз прибытия на следующую остановку
start_date = datetime.now()

# Создаем прогноз на основе среднего времени для каждой остановки
forecast_entries = []
for bus_stop, avg_time in average_travel_times.items():
    next_arrival = start_date + timedelta(seconds=avg_time)
    forecast_entries.append({
        'Time': next_arrival,
        'Bus Stop': bus_stop,
        'Forecasted Travel Time': avg_time
    })

forecast_df = pd.DataFrame(forecast_entries).set_index('Time')

# Выводим прогноз на экран
print("Прогноз прибытия следующего автобуса:")
print(forecast_df)

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(forecast_df.index, forecast_df['Forecasted Travel Time'], label='Прогноз прибытия')
plt.title("Прогноз времени прибытия следующего автобуса на остановку")
plt.xlabel("Время")
plt.ylabel("Среднее время в пути (секунды)")
plt.legend()
plt.show()

# Отправляем прогноз на Gemini API, ограничивая частоту до одного запроса в минуту
for _, entry in forecast_df.reset_index().iterrows():
    gemini_entry = {
        "Time": entry['Time'].isoformat(),
        "Bus Stop": entry['Bus Stop'],
        "Forecasted Travel Time": entry['Forecasted Travel Time']
    }
    send_data_to_gemini(gemini_entry)
