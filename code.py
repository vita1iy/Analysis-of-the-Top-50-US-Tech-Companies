import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#dataset and getting time series data for the stocks in my dataset
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

startdate = '2020-01-01'
enddate = '2022-12-31'

data = pd.read_csv("C:\\Users\\luchk\\Analysis-of-the-Top-50-US-Tech-Companies\\Dataset.csv")

#organized the data set and made everything spaced correctrly
data.dropna(inplace=True)
data["Annual Revenue 2022-2023 (USD in Billions)"] = pd.to_numeric(data["Annual Revenue 2022-2023 (USD in Billions)"], errors='coerce')
data["Market Cap (USD in Trillions)"] = pd.to_numeric(data["Market Cap (USD in Trillions)"], errors='coerce')
data["Annual Income Tax in 2022-2023 (USD in Billions)"] = pd.to_numeric(data["Annual Income Tax in 2022-2023 (USD in Billions)"], errors='coerce')
data["Employee Size"] = pd.to_numeric(data["Employee Size"], errors='coerce')
data.set_index("Company Name", inplace=True)

#print first five rows to see the changes
print(data.head())

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# Pie chart of "Sector" and the % of the companies in the specific sector
sector_count = data['Sector'].value_counts()
ax1.pie(sector_count, labels=sector_count.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 5})
ax1.set_title('Pie Chart of Sector', fontsize=12)

# Pie chart of "HQ State" to show where these companies are located in
state_count = data['HQ State'].value_counts()
ax2.pie(state_count, labels=state_count.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 5})
ax2.set_title('Pie Chart of HQ State', fontsize=12)

# adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

plt.show()