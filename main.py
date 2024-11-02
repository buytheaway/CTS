import requests
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# ВСТАВИТЬ API - ФУНКЦИЯ ДЛЯ ЗАПРОСА К ГЕМИНИ
def get_current_usage_from_gemini(symbol="btcusd"):
    """Получение текущего аналога загруженности через API Gemini."""
    url = f"https://api.gemini.com/v1/pubticker/{symbol}"
    response = requests.get(url)
    data = response.json()
    current_usage = float(data["last"])
    print(f"Текущая загруженность (аналоги данных Gemini) для {symbol}: {current_usage}")
    return current_usage


# Загрузка данных из локального CSV
df = pd.read_csv("municipality_bus_utilization.csv", parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Агрегация данных по часу
hourly_data = df.resample('H').mean().fillna(method='ffill')

# Проверка данных
if 'usage' in hourly_data.columns and not hourly_data['usage'].isnull().any():
    # Основной прогноз модели
    model = ExponentialSmoothing(hourly_data['usage'], trend='add', seasonal='add', seasonal_periods=24)
    fit = model.fit()
    forecast = fit.forecast(24)  # Прогноз на 24 часа

    # Визуализация прогноза
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_data.index[-100:], hourly_data['usage'][-100:], label='Исторические данные')
    plt.plot(pd.date_range(hourly_data.index[-1], periods=24, freq='H'), forecast, label='Прогноз')
    plt.title("Прогноз использования автобуса")
    plt.xlabel("Время")
    plt.ylabel("Использование автобуса")
    plt.legend()
    plt.show()

    # Определение часов пик
    peak_threshold = 1200  # Порог для часов пик
    forecast_peak_hours = forecast[forecast > peak_threshold]
    print("Часы пик:")
    print(forecast_peak_hours)

    # ВСТАВИТЬ API - ИСПОЛЬЗУЕМ ЗНАЧЕНИЕ С ГЕМИНИ ДЛЯ АКТУАЛЬНОЙ КОРРЕКЦИИ ВРЕМЕНИ ПРИБЫТИЯ
    current_usage = get_current_usage_from_gemini("btcusd")

    # Коррекция расписания на основе текущей загруженности
    arrival_interval = forecast.apply(lambda x: 20 if x > peak_threshold else 10)
    # Если текущая загруженность больше порога, добавляем задержку
    if current_usage > peak_threshold:
        arrival_interval += 5  # Увеличение интервала на 5 минут в случае высокой "загруженности"

    # Вычисление времени прибытия
    arrival_times = pd.date_range(start=hourly_data.index[-1], periods=24, freq="H") + pd.to_timedelta(arrival_interval,
                                                                                                       unit='m')

    # Создание расписания прибытия
    arrival_schedule = pd.DataFrame({
        "Forecast Usage": forecast,
        "Arrival Interval (min)": arrival_interval,
        "Expected Arrival Time": arrival_times
    })

    print("Расчетное расписание прибытия с учетом текущих данных Gemini:")
    print(arrival_schedule)
else:
    print("Проблема с данными: 'usage' содержит NaN или отсутствует.")
