import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error




def create_training_data(close_prices: pd.Series, window_day_size: int = 5): # function creating data to use for training
    x_list = []
    y_list = []

    for i in range(window_day_size, len(close_prices)): # loop to get the days after 4 days to get the price prediction
        window = close_prices.iloc[i-window_day_size:i]
        label = close_prices.iloc[i]

        x_list.append(window.values)
        y_list.append(label)
    
    return np.array(x_list), np.array(y_list)

def training_model(close_prices: pd.Series, window_day_size= 5): # function training in linear regression
    x, y = create_training_data(close_prices, window_day_size)

    split = int(len(x) * 0.8) # splitting the training into 80% training 20% testing
    
    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]

    lr = LinearRegression()

    lr.fit(x_train, y_train)
   
    y_pred = lr.predict(x_test)
    print(len(y_pred), len(y_test))

    mae  = mean_squared_error(y_test, y_pred)  # returning mae and rsme
    rmse = np.sqrt(mae)                  
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    last_window = close_prices.iloc[-window_day_size:].to_numpy().reshape(1, -1)
    next_close_pred = lr.predict(last_window)[0]

    return {
        "model": lr,
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test,   "y_test": y_test,
        "y_pred": y_pred,
        "mae": float(mae),
        "rmse": float(rmse),
        "next_close_pred": float(next_close_pred),
        "window": int(window_day_size),
        "n_samples": int(len(x)),
    }




