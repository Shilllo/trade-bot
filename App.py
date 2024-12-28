import streamlit as st
import pandas as pd
import os
from indicators import dataloader, simulation

# Путь к папке с данными по тикерам
DATA_PATH = "tickers_data/"
MODELS = ["Random Forest", "Logistic Regression"]

# Функция для получения списка доступных тикеров
def get_ticker_list():
    if not os.path.exists(DATA_PATH):
        st.error("Папка с данными не найдена. Проверьте путь к данным.")
        return []
    files = os.listdir(DATA_PATH)
    tickers = [file.split('.')[0] for file in files if file.endswith('.csv')]
    return tickers

# Функция для загрузки данных по выбранному тикеру
def load_ticker_data(ticker):
    file_path = os.path.join(DATA_PATH, f"{ticker}.csv")
    if not os.path.exists(file_path):
        st.error(f"Файл данных для тикера {ticker} не найден.")
        return pd.DataFrame()
    data = dataloader(file_path)
    return data

# Интерфейс Streamlit
st.title("💹 Торговый бот на основе технических индикаторов")
st.write("Выберите параметры для симуляции торговли.")

# Выбор начального баланса
initial_balance = st.number_input("Введите начальный баланс (единиц валюты):", min_value=0, value=100)

# Загрузка списка тикеров
ticker_list = get_ticker_list()
if not ticker_list:
    st.stop()

# Выбор тикера из списка
selected_ticker = st.selectbox("Выберите тикер:", ["-"] + ticker_list)

# Проверка, что тикер выбран
if selected_ticker == "-":
    st.warning("Пожалуйста, выберите тикер из списка.")
    st.stop()

# Загрузка данных для выбранного тикера
data = load_ticker_data(selected_ticker)
if data.empty:
    st.warning("Не удалось загрузить данные. Проверьте файл данных.")
    st.stop()

# Отображение первых строк данных
st.subheader(f"Данные для тикера: {selected_ticker}")
st.dataframe(data.head())

# Настройка параметров индикаторов через боковую панель
st.sidebar.header("🤖 Выбор ML модели")

# Параметры ML модели
choose_ML_model = st.sidebar.selectbox(
    "Выберите модель для симуляции:",
    MODELS
)

# Настройка параметров индикаторов через боковую панель
st.sidebar.header("⚙️ Настройки индикаторов")

# Параметры RSI
rsi_window = st.sidebar.slider("Окно RSI", min_value=5, max_value=30, value=14)

# Параметры MACD
macd_short = st.sidebar.slider("Короткое окно MACD", min_value=5, max_value=30, value=12)
macd_long = st.sidebar.slider("Длинное окно MACD", min_value=10, max_value=60, value=26)
macd_signal = st.sidebar.slider("Сигнальное окно MACD", min_value=5, max_value=30, value=9)

# Параметры Bollinger Bands
bb_window = st.sidebar.slider("Окно Bollinger Bands", min_value=10, max_value=50, value=20)
bb_num_std = st.sidebar.slider("Количество стандартных отклонений", min_value=1.0, max_value=3.0, value=2.0)

# Параметры Stochastic Oscillator
st_k_period = st.sidebar.slider("K-период Stochastic", min_value=5, max_value=30, value=14)
st_d_period = st.sidebar.slider("D-период Stochastic", min_value=3, max_value=15, value=3)

# Параметры Aroon
aroon_window = st.sidebar.slider("Окно Aroon", min_value=5, max_value=50, value=14)

# Параметры Parabolic SAR
psar_step = st.sidebar.slider("Шаг Parabolic SAR", min_value=0.01, max_value=0.1, value=0.02)
psar_max_step = st.sidebar.slider("Максимальный шаг Parabolic SAR", min_value=0.1, max_value=0.5, value=0.2)

# Запуск симуляции
if st.button("Запустить симуляцию"):
    result = simulation(
        data,
        rsi_window,
        [macd_short, macd_long, macd_signal],
        [bb_window, bb_num_std, 2, 2],
        [st_k_period, st_d_period, 3, 3],
        aroon_window,
        [psar_step, psar_max_step],
        balance=initial_balance
    )

    st.success(f"Итоговый баланс: {result:.2f}")

    # Отображение результатов
    st.write(f"Симуляция завершена. Итоговый капитал: {result:.2f} единиц валюты.")

# Дополнительная статистика по данным
if st.checkbox("Показать статистику по данным"):
    st.write(data.describe())
