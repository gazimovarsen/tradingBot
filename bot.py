import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import time


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= .63] = 1
    preds[preds < .63] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


t0 = time.time()
assets = open('assets.txt', 'r')
results = open('results.txt', 'w')
count = 0

for ticker in assets:
    ticker = str(ticker)[:-1]
    try:
        stock = yf.Ticker(ticker)
        stock = stock.history(period="max")
        stock.index = pd.to_datetime(stock.index)
        del stock["Dividends"]
        del stock["Stock Splits"]
        stock["Tomorrow"] = stock["Close"].shift(-1)
        stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)
        stock = stock.loc["1990-01-01":].copy()

        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

        predictors = ["Close", "Volume", "Open", "High", "Low"]
        # predictions = backtest(stock, model, predictors)
        # print(precision_score(predictions["Target"], predictions["Predictions"]))
        # print(predictions['Predictions'].iloc[-1])

        horizons = [2, 5, 60, 250, 1000]
        new_predictors = []

        for horizon in horizons:
            rolling_averages = stock.rolling(horizon).mean()

            ratio_column = f"Close_Ratio_{horizon}"
            stock[ratio_column] = stock["Close"] / rolling_averages["Close"]

            trend_column = f"Trend_{horizon}"
            stock[trend_column] = stock.shift(1).rolling(horizon).sum()["Target"]

            new_predictors += [ratio_column, trend_column]

        stock = stock.dropna(subset=stock.columns[stock.columns != "Tomorrow"])
        predictions = backtest(stock, model, new_predictors)

        count += 1
        print(count, ticker, precision_score(predictions["Target"], predictions["Predictions"]),
              predictions['Predictions'].iloc[-1])

        # results.write(str(count) + ' ' + ticker + ' ' + str(precision_score(predictions["Target"], predictions["Predictions"]))
        #              + ' ' + str(predictions['Predictions'].iloc[-1]) + '\n')
        print(predictions["Target"].value_counts() / predictions.shape[0])
        print(predictions["Predictions"].value_counts())
        print(predictions['Predictions'].iloc[-1])
        print()
        print()
        # f.write(predictions['Predictions'].to_string())
    except:
        pass

t1 = time.time()
print('Время работы скрипта: ' + str(t1 - t0))
