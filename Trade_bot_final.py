import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import copy
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# data preparation
def get_stock_data(ticker, tf, num_of_sticks=50000, market='shares'):
    url = f"https://iss.moex.com/iss/engines/stock/markets/{market}/securities/{ticker}/candles.json"
    till_date = datetime.today()
    delta_mapping = {
        1: timedelta(minutes=1),
        10: timedelta(minutes=10),
        15: timedelta(minutes=15),
        30: timedelta(minutes=30),
        60: timedelta(hours=1),
        24: timedelta(days=1),
        7: timedelta(weeks=1)
    }
    time_delta = delta_mapping[tf] * num_of_sticks
    params = {
        'interval': 1,  # Таймфрейм: 24 часа (дневные свечи)
        'from': till_date - time_delta,  # Начальная дата
        'till': till_date  # Конечная дата
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Проверка на ошибки
    data = response.json()
    candles = data.get('candles', {}).get('data', [])
    columns = data.get('candles', {}).get('columns', [])
    data = pd.DataFrame(candles, columns=columns).drop(columns=['begin', 'end', 'value'])
    data.columns = ['Open', 'Close', 'High', 'Low', 'Volume']
    return data

def dataset_preparation(data, *indicators):
    df = copy.deepcopy(data)

    df['Target'] = 0.5
    df['Local_Max'] = df['Close'].iloc[argrelextrema(df['Close'].values, np.greater, order=5)[0]]
    df['Local_Min'] = df['Close'].iloc[argrelextrema(df['Close'].values, np.less, order=5)[0]]
    df.loc[df['Local_Min'].notna(), 'Target'] = 1
    df.loc[df['Local_Max'].notna(), 'Target'] = 0
    df = df.drop(columns=['Local_Min', 'Local_Max'])

    if indicators[0]:
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    if indicators[1]:
        ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD'], df['MACD_signal'] = macd, signal

    if indicators[2]:
        sma = df['Close'].rolling(window=14).mean()
        std = df['Close'].rolling(window=14).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        df['BB_upper'], df['BB_lower'] = upper_band, lower_band

    if indicators[3]:    
        high = df['High'].rolling(window=14).max()
        low = df['Low'].rolling(window=14).min()
        k = 100 * ((df['Close'] - low) / (high - low))
        d = k.rolling(window=3).mean()
        df['Stoch_K'], df['Stoch_D'] = k, d

    if indicators[4]:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        df['ATR'] = atr

    if indicators[5]:
        aroon_up = 100 * df['High'].rolling(window=14).apply(lambda x: x.argmax() / 14, raw=True)
        aroon_down = 100 * df['Low'].rolling(window=14).apply(lambda x: x.argmin() / 14, raw=True)
        df['Aroon_Up'], df['Aroon_Down'] = aroon_up, aroon_down

        af, max_af = 0.02, 0.2
        sar = df['Low'][0]
        ep = df['High'][0]
        trend = 1
        sar_list = []

    if indicators[6]:
        for i in range(1, len(data)):
            prev_sar = sar
            sar = prev_sar + af * (ep - prev_sar)

            if trend == 1:
                if df['Low'][i] < sar:
                    trend = -1
                    sar = ep
                    ep = df['Low'][i]
                    af = 0.02
                else:
                    if df['High'][i] > ep:
                        ep = df['High'][i]
                        af = min(af + 0.02, max_af)
            elif trend == -1:
                if df['High'][i] > sar:
                    trend = 1
                    sar = ep
                    ep = df['High'][i]
                    af = 0.02
                else:
                    if df['Low'][i] < ep:
                        ep = df['Low'][i]
                        af = min(af + 0.02, max_af)

            sar_list.append(sar)

        df['Parabolic_SAR'] = pd.Series(sar_list, index=data.index[1:])

    if indicators[7]:
        mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
        cmf = mfv.rolling(window=14).sum() / df['Volume'].rolling(window=14).sum()
        df['CMF'] = cmf

    df['Volatility'] = df['Close'].rolling(window=20).std()

    result = seasonal_decompose(df['Close'], model='additive', period=90)
    df['Trend'] = result.trend
    df['Seasonal'] = result.seasonal

    df = df.drop(columns=['Open', 'High', 'Low'])
    df.dropna(inplace=True)
    df = df.loc[df['Target'] != 0.5]
    return df


# models
def split_data(data):
    return (
        data.iloc[:int(len(data) * 0.8)].drop(columns=['Target', 'Close']),
        data.iloc[:int(len(data) * 0.8)]['Target'],
        data.iloc[int(len(data) * 0.8):].drop(columns=['Target', 'Close']),
        data.iloc[int(len(data) * 0.8):]['Target'],
        data.iloc[:int(len(data) * 0.8)]['Close'],
        data.iloc[int(len(data) * 0.8):]['Close']
    )

def fit_random_forest(train_data, train_target):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_data, train_target)
    return model

def fit_log_reg(train_data, train_target):
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(train_data, train_target)
    return model

def predict(data, model):
    return model.predict(data)

def probability(data, model):
    return model.predict_proba(data)

X_train, y_train, X_test, y_test, close_train, close_test = split_data(dataset_preparation(get_stock_data('AFLT', 1), 1, 1, 1, 1, 1, 1, 1, 1))

# simulation
def simulate_trading(data, close_prices, model, treshold=0.9, initial_balance=100000, lot_size=1):
    balance = initial_balance  # Начальный баланс
    position = 0  # Текущая позиция (количество акций)
    trade_history = []  # История сделок

    for i in range(len(data)):
        # Цена закрытия на текущий день
        close_price = close_prices.iloc[i]
        proba_key = probability(data.iloc[[i]], model)[0].argmax()
        proba_value = probability(data.iloc[[i]], model)[0][proba_key]
        # Действие в зависимости от предсказания
        if proba_value > treshold and proba_key == 1:  # Покупка
            if balance >= close_price * lot_size:
                position += lot_size
                balance -= close_price * lot_size
                trade_history.append({
                    'action': 'buy',
                    'price': close_price,
                    'balance': balance,
                    'position': position,
                    'index': i
                })

        elif proba_value > treshold and proba_key == 0:  # Продажа
            if position >= lot_size:
                position -= lot_size
                balance += close_price * lot_size
                trade_history.append({
                    'action': 'sell',
                    'price': close_price,
                    'balance': balance,
                    'position': position,
                    'index': i
                })

    # Финальный пересчет: реализация оставшихся активов
    if position > 0:
        balance += position * close_prices.iloc[-1]
        trade_history.append({
            'action': 'sell_all',
            'price': close_prices.iloc[-1],
            'balance': balance,
            'position': 0,
            'index': len(data) - 1
        })

    return balance, trade_history


# Пример использования
final_balance, trades = simulate_trading(
    data=X_test,
    close_prices=close_test,
    model=fit_log_reg(X_train, y_train),
    treshold=0.95,
    initial_balance=100,
    lot_size=1
)

print("Итоговый баланс:", final_balance)
print("История сделок:")
for trade in trades:
    print(trade)