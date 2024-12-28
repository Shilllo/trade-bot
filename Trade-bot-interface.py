import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Подключение торгового бота
from Trade_bot_final import (get_stock_data, dataset_preparation, split_data,
                         fit_random_forest, fit_log_reg, simulate_trading)

def main():
    st.title("Торговый бот")

    # Ввод данных
    st.header("Параметры загрузки данных")
    ticker = st.text_input("Тикер акции", "AFLT")
    timeframe = st.selectbox("Таймфрейм", options=[1, 10, 15, 30, 60, 24, 7], index=3)
    num_of_sticks = 50000

    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = None

    if 'model' not in st.session_state:
        st.session_state['model'] = None

    if st.button("Загрузить данные"):
        with st.spinner("Загрузка данных..."):
            try:
                st.session_state['data'] = get_stock_data(ticker, timeframe, num_of_sticks)
                st.success("Данные успешно загружены!")
                st.session_state['processed_data'] = None  # Сбрасываем обработанные данные при загрузке новых
                st.write(st.session_state['data'].head())

                # Визуализация данных
                st.subheader("График цен закрытия")
                fig, ax = plt.subplots()
                ax.plot(st.session_state['data']["Close"], label="Цена закрытия")
                ax.set_xlabel("Время")
                ax.set_ylabel("Цена")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Ошибка загрузки данных: {e}")

    # Настройка индикаторов
    st.header("Индикаторы")
    rsi = st.checkbox("RSI", value=True)
    macd = st.checkbox("MACD", value=True)
    bollinger_bands = st.checkbox("Bollinger Bands", value=True)
    stochastic = st.checkbox("Stochastic", value=True)
    atr = st.checkbox("ATR", value=True)
    aroon = st.checkbox("Aroon", value=True)
    parabolic_sar = st.checkbox("Parabolic SAR", value=True)
    cmf = st.checkbox("CMF", value=True)

    indicators = [rsi, macd, bollinger_bands, stochastic, atr, aroon, parabolic_sar, cmf]

    if st.button("Подготовить данные"):
        if st.session_state['data'] is None:
            st.error("Сначала загрузите данные!")
        else:
            with st.spinner("Обработка данных..."):
                try:
                    st.session_state['processed_data'] = dataset_preparation(st.session_state['data'], *indicators)
                    st.success("Данные успешно обработаны!")
                    st.write(st.session_state['processed_data'].head())
                except Exception as e:
                    st.error(f"Ошибка обработки данных: {e}")

    # Обучение модели
    st.header("Обучение модели")
    model_type = st.selectbox("Выберите модель", ["Logistic Regression", "Random Forest"])

    if st.button("Обучить модель"):
        if st.session_state['processed_data'] is None:
            st.error("Сначала подготовьте данные!")
        else:
            with st.spinner("Обучение модели..."):
                try:
                    X_train, y_train, X_test, y_test, close_train, close_test = split_data(st.session_state['processed_data'])
                    if model_type == "Logistic Regression":
                        st.session_state['model'] = fit_log_reg(X_train, y_train)
                    else:
                        st.session_state['model'] = fit_random_forest(X_train, y_train)
                    st.success("Модель успешно обучена!")
                    st.write("Точность модели:", st.session_state['model'].score(X_test, y_test))
                except Exception as e:
                    st.error(f"Ошибка обучения модели: {e}")

    # Симуляция торговли
    st.header("Симуляция торговли")
    threshold = st.slider("Порог вероятности", min_value=0.5, max_value=1.0, step=0.05, value=0.9)
    initial_balance = st.number_input("Начальный баланс", min_value=100, value=100000, step=1000)
    lot_size = st.number_input("Размер лота", min_value=1, value=1)

    if st.button("Симулировать торговлю"):
        if st.session_state['processed_data'] is None or st.session_state['model'] is None:
            st.error("Убедитесь, что данные подготовлены и модель обучена!")
        else:
            with st.spinner("Симуляция торговли..."):
                try:
                    X_train, y_train, X_test, y_test, close_train, close_test = split_data(st.session_state['processed_data'])
                    final_balance, trade_history = simulate_trading(X_test, close_test, st.session_state['model'], threshold, initial_balance, lot_size)
                    st.success(f"Симуляция завершена! Итоговый баланс: {final_balance}")

                    # Визуализация истории сделок
                    st.subheader("История сделок")
                    trade_df = pd.DataFrame(trade_history)
                    st.write(trade_df)

                    st.subheader("График баланса")
                    fig, ax = plt.subplots()
                    ax.plot(trade_df["index"], trade_df["balance"], label="Баланс")
                    ax.set_xlabel("Индекс сделки")
                    ax.set_ylabel("Баланс")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Ошибка симуляции торговли: {e}")

if __name__ == "__main__":
    main()
