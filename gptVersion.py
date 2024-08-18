import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import time
import logging
from joblib import Parallel, delayed

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def process_ticker(ticker):
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

        precision = precision_score(predictions["Target"], predictions["Predictions"])
        last_prediction = predictions['Predictions'].iloc[-1]

        logging.info(f"{ticker}: Precision: {precision}, Last Prediction: {last_prediction}")

        return {
            'ticker': ticker,
            'precision': precision,
            'last_prediction': last_prediction,
            'target_distribution': predictions["Target"].value_counts() / predictions.shape[0],
            'prediction_distribution': predictions["Predictions"].value_counts()
        }
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}")
        return None


def main():
    t0 = time.time()
    with open('assets.txt', 'r') as assets_file:
        tickers = [line.strip() for line in assets_file.readlines()]

    results = Parallel(n_jobs=-1)(delayed(process_ticker)(ticker) for ticker in tickers)

    results = [result for result in results if result is not None]

    with open('results.txt', 'w') as results_file:
        for result in results:
            results_file.write(f"{result['ticker']} {result['precision']} {result['last_prediction']}\n")
            print(f"{result['ticker']} {result['precision']} {result['last_prediction']}\n")

    t1 = time.time()
    logging.info(f'Script execution time: {t1 - t0}')


if __name__ == "__main__":
    main()
