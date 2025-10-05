import yfinance as yf

def fetch_stock_data(ticker_symbol, period="3mo", interval="1d"): 
     #Function to get data from a stock

    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period, interval=interval)
    close_prices = data["Close"].dropna().astype("float64")

    return data

if __name__ == "__main__":
    df = fetch_stock_data("AAPL")
    print(df.head())

    

