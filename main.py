import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Загрузка переменных окружения
load_dotenv(".env")  # Загрузка API ключа из файла
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Проверка API-ключей
if not GEMINI_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("Один или оба API-ключа не найдены! Проверьте файл api.env.")

# Функция для получения "загруженности" из Gemini API
def get_current_usage_from_gemini(symbol="btcusd"):
    """Получение текущей загруженности (в качестве примера) через Gemini API."""
    url = f"https://api.gemini.com/v1/pubticker/{symbol}"
    headers = {"X-GEMINI-APIKEY": GEMINI_API_KEY}
    response = requests.get(url, headers=headers)
    data = response.json()
    current_usage = float(data["last"])
    print(f"Текущая загруженность (данные Gemini) для {symbol}: {current_usage}")
    return current_usage

# Функция для отправки запроса к Google Language API
def ask_google_language_model(question="Explain how AI works"):
    """Отправка запроса к Google Language API с вопросом."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": question
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("Ответ от Google Language API:", response.json())
    else:
        print("Ошибка:", response.status_code, response.text)

# Загрузка данных из новых датасетов
dataset_1 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет #1')
dataset_2 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет 2')
dataset_3 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет 3')
dataset_4 = pd.read_csv('C:/Users/mukha/Documents/GitHub/CTS/Датасет 4')

# Обработка Датасета 1: Расчет времени в пути между остановками
dataset_1['arrival_time'] = pd.to_datetime(dataset_1['arrival_time'], format='%H:%M:%S')
dataset_1['departure_time'] = pd.to_datetime(dataset_1['departure_time'], format='%H:%M:%S')
dataset_1['travel_time'] = dataset_1.apply(
    lambda row: (row['arrival_time'] - dataset_1.loc[row.name - 1, 'arrival_time']).total_seconds()
    if row.name > 0 and row['trip_id'] == dataset_1.loc[row.name - 1, 'trip_id'] else None,
    axis=1
)

# Преобразование типов для объединения
dataset_1['bus_stop'] = dataset_1['bus_stop'].astype(str)
dataset_2['stop_id'] = dataset_2['stop_id'].astype(str)

# Объединение с Датасетом 2
dataset_1 = dataset_1.merge(dataset_2, how='left', left_on='bus_stop', right_on='stop_id')

# Обработка Датасета 3: Время поездок по сегментам
dataset_3['start_time'] = pd.to_datetime(dataset_3['start_time'], format='%H:%M:%S')
dataset_3['end_time'] = pd.to_datetime(dataset_3['end_time'], format='%H:%M:%S')
dataset_3['segment_duration'] = (dataset_3['end_time'] - dataset_3['start_time']).dt.total_seconds()
dataset_1 = dataset_1.merge(dataset_3[['trip_id', 'deviceid', 'segment', 'segment_duration']],
                            how='left', on=['trip_id', 'deviceid'])

# Обработка Датасета 4: Общая длительность поездок
dataset_4['start_time'] = pd.to_datetime(dataset_4['start_time'], format='%H:%M:%S')
dataset_4['end_time'] = pd.to_datetime(dataset_4['end_time'], format='%H:%M:%S')
dataset_4['total_trip_duration'] = (dataset_4['end_time'] - dataset_4['start_time']).dt.total_seconds()
dataset_1 = dataset_1.merge(dataset_4[['trip_id', 'deviceid', 'start_terminal', 'end_terminal', 'total_trip_duration']],
                            how='left', on=['trip_id', 'deviceid'])

# Прогнозирование с использованием Exponential Smoothing
hourly_data = dataset_1.set_index('arrival_time').resample('H')['travel_time'].mean().fillna(method='ffill')
model = ExponentialSmoothing(hourly_data, trend='add', seasonal='add', seasonal_periods=24)
fit = model.fit()
forecast = fit.forecast(24)

# Получение текущей загруженности и корректировка расписания
current_usage = get_current_usage_from_gemini("btcusd")
peak_threshold = 1200
arrival_interval = forecast.apply(lambda x: 20 if x > peak_threshold else 10)

if current_usage > peak_threshold:
    arrival_interval += 5

arrival_times = pd.date_range(start=hourly_data.index[-1], periods=24, freq="H") + pd.to_timedelta(arrival_interval, unit='m')
arrival_schedule = pd.DataFrame({
    "Forecast Usage": forecast,
    "Arrival Interval (min)": arrival_interval,
    "Expected Arrival Time": arrival_times
})



# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(hourly_data.index[-100:], hourly_data[-100:], label='Исторические данные')
plt.plot(pd.date_range(hourly_data.index[-1], periods=24, freq='H'), forecast, label='Прогноз')
plt.title("Прогноз использования автобуса")
plt.xlabel("Время")
plt.ylabel("Использование автобуса")
plt.legend()
plt.show()

print("Расчетное расписание прибытия с учетом текущих данных Gemini:")
print(arrival_schedule)

# Пример запроса к Google Language API
ask_google_language_model("Explain how AI works")
