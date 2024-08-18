import time

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Замер времени начала работы скрипта
start_time = time.time()


def make_predictions(train_data, test_data, predictors, model):
    model.fit(train_data[predictors], train_data["Target"])
    predicted_probs = model.predict_proba(test_data[predictors])[:, 1]
    predicted_labels = (predicted_probs >= 0.63).astype(int)
    predictions = pd.Series(predicted_labels, index=test_data.index, name="Predictions")
    combined_results = pd.concat([test_data["Target"], predictions], axis=1)
    return combined_results


def backtest_model(data, model, predictors, start_index=2500, step_size=250):
    all_predictions = []
    for i in range(start_index, data.shape[0], step_size):
        train_data = data.iloc[0:i].copy()
        test_data = data.iloc[i:(i + step_size)].copy()
        predictions = make_predictions(train_data, test_data, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# Загрузка данных для тикера ^GSPC (S&P 500)
ticker_symbol = "^GSPC"
stock_data = yf.Ticker(ticker_symbol)
stock_history = stock_data.history(period="max")
stock_history.index = pd.to_datetime(stock_history.index)
del stock_history["Dividends"]
del stock_history["Stock Splits"]
stock_history["Next_Close"] = stock_history["Close"].shift(-1)
stock_history["Target"] = (stock_history["Next_Close"] > stock_history["Close"]).astype(int)
stock_history = stock_history.loc["1990-01-01":].copy()

# Создание модели случайного леса
random_forest_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Создание предикторов на основе различных горизонтов
rolling_horizons = [2, 5, 60, 250, 1000]
new_features = []

for horizon in rolling_horizons:
    rolling_averages = stock_history.rolling(horizon).mean()

    ratio_col = f"Close_Ratio_{horizon}"
    stock_history[ratio_col] = stock_history["Close"] / rolling_averages["Close"]

    trend_col = f"Trend_{horizon}"
    stock_history[trend_col] = stock_history.shift(1).rolling(horizon).sum()["Target"]

    new_features += [ratio_col, trend_col]

# Удаление строк с отсутствующими значениями в предикторах
stock_history = stock_history.dropna(subset=new_features)

# Тестирование модели на исторических данных
predicted_results = backtest_model(stock_history, random_forest_model, new_features)

# Оценка результатов
precision_metric = precision_score(predicted_results["Target"], predicted_results["Predictions"])
last_predicted_value = predicted_results['Predictions'].iloc[-1]

# Вывод результатов
print(f"Precision Score: {precision_metric}")
print("Target Value Counts:")
print(predicted_results["Target"].value_counts() / predicted_results.shape[0])
print("Predictions Value Counts:")
print(predicted_results["Predictions"].value_counts())
print("Last Prediction:", last_predicted_value)

# Замер времени окончания работы скрипта и вывод общего времени работы
end_time = time.time()
print('Время работы скрипта:', end_time - start_time)
