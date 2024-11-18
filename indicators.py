import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st

# Отключаем предупреждения
import warnings
warnings.filterwarnings('ignore')

### 1. Функция для загрузки данных
import pandas as pd

def dataloader(name):
    # Загрузка данных с явным указанием, что первая строка — это заголовок
    data = pd.read_csv(name, delimiter=';', encoding='cp1251', header=0)
    data = data.drop(columns=['<TICKER>', '<PER>', '<DATE>', '<VOL>', '<TIME>', '<OPENINT>', '<OPEN>'])

    # Выводим имена всех колонок для проверки
    print("Колонки загруженного файла:", data.columns.tolist())
    
    # Пытаемся переименовать колонки
    # data = data.rename(columns={
    #     '<HIGH>': 'High',
    #     '<LOW>': 'Low',
    #     '<CLOSE>': 'Close',
    #     '<ОТКРЫТИЕ>': 'Open',    # Если есть русские названия колонок
    #     '<ЗАКРЫТИЕ>': 'Close'    # Русское название колонки
    # })
    
    # Выводим имена колонок после переименования
    # print("Колонки после переименования:", data.columns.tolist())
    
    # Проверяем первые строки данных
    print(data.head())
    
    return data


### 2. Индикатор RSI
def rsi(data, window=14):
    delta = data['<CLOSE>'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def rsi_signal(data, window=14):
    res_rsi = rsi(data, window).iloc[-1]
    if res_rsi > 70:
        return 'SELL'
    elif res_rsi < 30:
        return 'BUY'
    else:
        return 'HOLD'

### 3. Индикатор MACD
def macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['<CLOSE>'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['<CLOSE>'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

def macd_signal(data, short_window=12, long_window=26, signal_window=9):
    macd_line, signal_line = macd(data, short_window, long_window, signal_window)
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        return 'BUY'
    else:
        return 'SELL'

### 4. Индикатор Bollinger Bands
def bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['<CLOSE>'].rolling(window=window).mean()
    rolling_std = data['<CLOSE>'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def bollinger_bands_signal(data, window=20, num_std=2):
    upper_band, lower_band = bollinger_bands(data, window, num_std)
    if data['<CLOSE>'].iloc[-1] > upper_band.iloc[-1]:
        return 'SELL'
    elif data['<CLOSE>'].iloc[-1] < lower_band.iloc[-1]:
        return 'BUY'
    else:
        return 'HOLD'

### 5. Стохастический осциллятор
def stochastic(data, k_period=14, d_period=3):
    low_min = data['<LOW>'].rolling(window=k_period).min()
    high_max = data['<HIGH>'].rolling(window=k_period).max()
    k = 100 * ((data['<CLOSE>'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def stochastic_signal(data, k_period=14, d_period=3):
    k, d = stochastic(data, k_period, d_period)
    if k.iloc[-1] > d.iloc[-1]:
        return 'BUY'
    else:
        return 'SELL'

### 6. Индикатор Aroon
def aroon(data, window=14):
    aroon_up = 100 * (window - data['<HIGH>'].rolling(window=window).apply(lambda x: window - np.argmax(x), raw=True)) / window
    aroon_down = 100 * (window - data['<LOW>'].rolling(window=window).apply(lambda x: window - np.argmin(x), raw=True)) / window
    return aroon_up, aroon_down

def aroon_signal(data, window=14):
    aroon_up, aroon_down = aroon(data, window)
    if aroon_up.iloc[-1] > aroon_down.iloc[-1]:
        return 'BUY'
    else:
        return 'SELL'

### 7. Индикатор Parabolic SAR
def parabolic_sar(data, step=0.02, max_step=0.2):
    psar = data['<CLOSE>'].copy()
    trend = 1  # 1 - восходящий, -1 - нисходящий
    ep = data['<LOW>'][0]
    af = step

    for i in range(1, len(data)):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])

        if trend == 1:
            if data['<LOW>'][i] < psar[i]:
                trend = -1
                psar[i] = ep
                af = step
                ep = data['<HIGH>'][i]
        else:
            if data['<LOW>'][i] > psar[i]:
                trend = 1
                psar[i] = ep
                af = step
                ep = data['<LOW>'][i]

        if trend == 1 and data['<HIGH>'][i] > ep:
            ep = data['<HIGH>'][i]
            af = min(af + step, max_step)
        elif trend == -1 and data['<LOW>'][i] < ep:
            ep = data['<LOW>'][i]
            af = min(af + step, max_step)

    return psar

def parabolic_sar_signal(data, step=0.02, max_step=0.2):
    psar = parabolic_sar(data, step, max_step)
    if data['<CLOSE>'].iloc[-1] > psar.iloc[-1]:
        return 'BUY'
    else:
        return 'SELL'

### 8. Функция симуляции
def simulation(data, rsi_window, macd_params, bb_params, st_params, aroon_window, parabolic_params, balance = 100):

    current_pos = ['HOLD', 0]
    progress_bar = st.progress(0)
    total_steps = len(data) - round(len(data) * 0.25)

    for i in range(round(len(data) * 0.25), len(data)):
        # Обновляем прогресс-бар
        progress = (i - round(len(data) * 0.25)) / total_steps
        progress_bar.progress(progress)

        # Выводим текущий шаг каждые 100 итераций
        if i % 100 == 0:
            st.write(f"Итерация {i}/{len(data)}: текущий баланс {balance:.2f}")

        reviews = [
            rsi_signal(data.iloc[:i], rsi_window),
            macd_signal(data.iloc[:i], *macd_params),
            bollinger_bands_signal(data.iloc[:i], bb_params[0], bb_params[1]),
            stochastic_signal(data.iloc[:i], st_params[0], st_params[1]),
            aroon_signal(data.iloc[:i], aroon_window),
            parabolic_sar_signal(data.iloc[:i], *parabolic_params)
        ]
        signal = Counter(reviews).most_common(1)[0][0]

        # Торговая логика с использованием оригинальных колонок
        if signal == 'BUY' and current_pos[0] != 'BUY':
            if current_pos[0] == 'SELL':
                balance *= (data.iloc[i]['<CLOSE>'] / current_pos[1])
            current_pos = ['BUY', data.iloc[i]['<CLOSE>']]
        elif signal == 'SELL' and current_pos[0] != 'SELL':
            if current_pos[0] == 'BUY':
                balance *= (current_pos[1] / data.iloc[i]['<CLOSE>'])
            current_pos = ['SELL', data.iloc[i]['<CLOSE>']]

    st.success(f"Симуляция завершена. Итоговый баланс: {balance:.2f}")
    return balance