import yfinance as yf
import pandas as pd
import xgboost as xgb

# tickers to gather time series data for
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 
           'TSLA', 'META', 'AVGO', 'ORCL', 'CSCO', 
           'CRM', 'ADBE', 'TXN', 'AMD', 'QCOM', 
           'NFLX', 'INTC', 'INTU', 'IBM', 'AMAT', 
           'BKNG', 'ADI', 'NOW', 'ADP', 'PYPL', 
           'ABNB', 'FISV', 'LRCX', 'UBER', 'MU', 
           'EQIX', 'ATVI', 'PANW', 'SNPS', 'CDNS', 
           'KLAC', 'ANET', 'VMW', 'WDAY', 'FTNT', 
           'SQ', 'SNOW', 'ROP', 'MCHP', 'ADSK', 
           'GFS', 'IQV', 'MRVL', 'DELL', 'HPQ']

# set the start and end dates
startdate = '2020-01-01'
enddate = '2022-12-31'

# get time series data for each stock
stock_data = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=startdate, end=enddate)
    data.rename(columns={"Adj Close": ticker}, inplace=True)
    stock_data = pd.concat([stock_data, data[ticker]], axis=1)

    print(stock_data.head())