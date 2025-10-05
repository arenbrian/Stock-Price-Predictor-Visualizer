import streamlit as st
from data_fetch import fetch_stock_data
import matplotlib.pyplot as plt
from predictor import training_model
import predictor
import importlib
importlib.reload(predictor)  # force reload the latest predictor.py

st.title("Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker", value="AAPL")
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "5y"])
window = st.slider("Window size (days)", min_value=3, max_value=30, value=5, step=1)



if ticker: # if statement to show data of stock
    data = fetch_stock_data(ticker, period=period)
    if data.empty:
        st.write("No data found for this ticker and period.")

    else:
        st.write(f"Showing data for {ticker} over the last {period}")

        st.line_chart(data['Close'])

        

        fig, ax = plt.subplots()
        data['Close'].plot(ax=ax, title=f"{ticker} Closing Prices")
        st.pyplot(fig)

  
        close_prices = data["Close"].dropna().astype("float64")

        result = training_model(close_prices, window_day_size=window)


        st.subheader("Model Evaluation")
        st.write(f"MAE: {result['mae']:.4f}")
        st.write(f"RMSE: {result['rmse']:.4f}")

        st.subheader("Next Close Prediction")
        st.write(float(result["next_close_pred"]))



