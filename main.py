import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

# Загружаем API-ключи из файла .env
load_dotenv()  # Загружает ключи из .env по умолчанию


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Отладочный вывод для проверки
print("GEMINI_API_KEY:", GEMINI_API_KEY)
print("GOOGLE_API_KEY:", GOOGLE_API_KEY)

if not GEMINI_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Один или оба API-ключа не найдены! Проверьте файл .env.")


# Функция для отправки данных на Gemini API
def send_data_to_gemini(data):
    url = f"https://api.gemini.com/v1/order/new"  # Примерный URL для отправки данных
    headers = {"X-GEMINI-APIKEY": GEMINI_API_KEY, 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("Данные успешно отправлены на Gemini:", response.json())
    else:
        print("Ошибка отправки данных на Gemini:", response.status_code, response.text)


# Загрузка данных из новых датасетов
dataset_1 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет #1')
dataset_2 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет 2')

# Обработка данных из Датасета 1 и Датасета 2
dataset_1['arrival_time'] = pd.to_datetime(dataset_1['arrival_time'], format='%H:%M:%S')
dataset_1['departure_time'] = pd.to_datetime(dataset_1['departure_time'], format='%H:%M:%S')
dataset_1['travel_time'] = (dataset_1['departure_time'] - dataset_1['arrival_time']).dt.total_seconds()

# Преобразуем типы данных для объединения
dataset_1['bus_stop'] = dataset_1['bus_stop'].astype(str)
dataset_2['stop_id'] = dataset_2['stop_id'].astype(str)

# Объединяем с географическими данными
dataset_1 = dataset_1.merge(dataset_2, how='left', left_on='bus_stop', right_on='stop_id')

# Рассчитываем среднее время в пути для прогноза
average_travel_time = dataset_1['travel_time'].mean()

# Устанавливаем начальную дату прогноза как текущую дату и время
start_date = datetime.now()

# Создаем базовый прогноз на 24 часа на основе среднего значения
forecast_df = pd.DataFrame({
    'Forecasted Travel Time': [average_travel_time] * 24,
    'Bus Number': ['Bus 101'] * 24,  # Добавляем пример номера автобуса
    'Bus Stop': ['Stop A'] * 24  # Добавляем пример остановки
}, index=pd.date_range(start=start_date, periods=24, freq='H'))

print("Прогноз на основе среднего значения времени в пути с остановками и номером автобуса:")
print(forecast_df)

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(forecast_df.index, forecast_df['Forecasted Travel Time'], label='Прогноз на основе среднего значения')
plt.title("Прогноз времени в пути на основе среднего значения с остановками и номером автобуса")
plt.xlabel("Время")
plt.ylabel("Среднее время в пути (секунды)")
plt.legend()
plt.show()

# Преобразуем данные прогноза в формат для отправки на Gemini
forecast_data = forecast_df.reset_index().rename(columns={'index': 'Time'}).to_dict(orient='records')

# Преобразуем временные метки в строки для отправки
for entry in forecast_data:
    entry['Time'] = entry['Time'].isoformat()  # Преобразуем Timestamp в ISO формат строки

# Отправляем каждый прогноз на Gemini API
for entry in forecast_data:
    send_data_to_gemini(entry)
