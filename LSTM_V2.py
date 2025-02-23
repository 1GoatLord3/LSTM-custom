pip install pandas numpy matplotlib ccxt tensorflow matplotlib scikit-learn

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import time

# Helper functions for technical indicators
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(df, period, column='close'):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def fetch_ohlcv(symbol, timeframe, limit):
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Calculate RSI, EMA50, EMA200, ATR, and nATR
        df['RSI'] = calculate_rsi(df)
        df['EMA50'] = calculate_ema(df, 50)
        df['EMA200'] = calculate_ema(df, 200)
        df['ATR'] = calculate_atr(df)
        df['nATR'] = df['ATR'] / df['close']  # Normalize ATR by dividing by close price

        # Drop NaN values that result from indicator calculations
        df.dropna(inplace=True)

        return df
    except ccxt.NetworkError as e:
        print(f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        print(f"Exchange error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    return None

def prepare_data(data, time_step):
    # Use all 9 features: open, high, low, close, volume, RSI, EMA50, EMA200, nATR
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'EMA50', 'EMA200', 'nATR']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature_columns])

    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), :])  # Use all 9 features for X
        y.append(scaled_data[i + time_step, 3])  # Predict 'close' price (index 3)
    return np.array(X), np.array(y), scaler

def create_model(X_train, y_train, time_step, feature_count):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(time_step, feature_count)),
        LSTM(units=100),
        Dense(units=50),
        Dense(units=1)  # Predict a single value (the 'close' price)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    return model

def make_predictions(model, X, scaler, future_steps, feature_count):
    # Start with the last batch from X
    current_batch = X[-1].reshape(1, X.shape[1], feature_count)
    predicted = []

    for _ in range(future_steps):
        # Predict the next close price
        current_pred = model.predict(current_batch)[0]
        predicted.append(current_pred[0])

        # Roll the batch and update it with the predicted close price
        current_batch = np.roll(current_batch, -1, axis=1)
        current_batch[0, -1, 3] = current_pred[0]  # Update the 'close' value in the batch

    # Create a dummy array for inverse scaling
    dummy_pred = np.zeros((future_steps, feature_count))
    dummy_pred[:, 3] = np.array(predicted)  # Only fill the 'close' column with predictions

    # Inverse transform the predicted data
    return scaler.inverse_transform(dummy_pred)[:, 3]  # Return only the 'close' column

def plot_results(original, historical_pred, future_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(original, label='Original Data', color='blue')
    plt.plot(range(len(original)-len(historical_pred), len(original)), historical_pred, label='Historical Prediction', color='green')
    plt.plot(range(len(original)-1, len(original) + len(future_pred) - 1), future_pred, label='Future Prediction', color='red')
    plt.axvline(x=len(original)-1, color='black', linestyle='--', label='Prediction Start')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# Main execution in a loop
symbols = ['SOL/USDT', '']  # Multiple symbols
timeframes = ['1m']
limit = 1000  # Limit the amount of data to something reasonable

while True:
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"Processing {symbol} - {timeframe} timeframe")

            # Fetch data
            data = fetch_ohlcv(symbol, timeframe, limit)
            if data is None:
                print(f"Failed to fetch data for {symbol} - {timeframe} timeframe. Skipping...")
                continue

            # Prepare data
            time_step = 60 if timeframe == '15m' else 24
            feature_count = 9  # Using 9 features (open, high, low, close, volume, RSI, EMA50, EMA200, nATR)
            X, y, scaler = prepare_data(data, time_step)

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Reshape input data
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], feature_count))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], feature_count))

            # Create and train model
            model = create_model(X_train, y_train, time_step, feature_count)

            # Make historical predictions
            historical_pred = model.predict(X_test)

            # Create a dummy array with the same number of columns as the original scaled data (9 features)
            dummy_pred = np.zeros((historical_pred.shape[0], feature_count))
            dummy_pred[:, 3] = historical_pred[:, 0]  # The 3rd index is the 'close' column

            # Inverse transform the dummy array
            historical_pred = scaler.inverse_transform(dummy_pred)[:, 3]  # Extract only the 'close' column

            # Make future predictions
            future_steps = 12
            future_pred = make_predictions(model, X, scaler, future_steps, feature_count)

            # Prepare data for plotting
            original_data = data['close'].values

            # Plot results
            plot_results(original_data, historical_pred, future_pred, f"{symbol} Price Prediction - {timeframe} Timeframe")

            # Pause for a specific duration (e.g., 15 minutes) before the next execution
            time.sleep(60)  # Sleep for 900 seconds (15 minutes)
