import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# read in the original dataset
data = pd.read_csv("C:\\Users\\luchk\\Analysis-of-the-Top-50-US-Tech-Companies\\Dataset.csv")

# organize the dataset and make sure everything is spaced correctly
data.dropna(inplace=True)
data["Annual Revenue 2022-2023 (USD in Billions)"] = pd.to_numeric(data["Annual Revenue 2022-2023 (USD in Billions)"], errors='coerce')
data["Market Cap (USD in Trillions)"] = pd.to_numeric(data["Market Cap (USD in Trillions)"], errors='coerce')
data["Annual Income Tax in 2022-2023 (USD in Billions)"] = pd.to_numeric(data["Annual Income Tax in 2022-2023 (USD in Billions)"], errors='coerce')
data["Employee Size"] = pd.to_numeric(data["Employee Size"], errors='coerce')
data.set_index("Company Name", inplace=True)

# select relevant columns
relevant_cols = ["Industry", "Sector", "HQ State", "Founding Year", "Annual Revenue 2022-2023 (USD in Billions)", "Market Cap (USD in Trillions)", "Annual Income Tax in 2022-2023 (USD in Billions)", "Employee Size"]
data = data[relevant_cols]

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

#Interactive chart that has x variable as annual revenue and y variable as market cap
interactivefig = px.scatter(data, x="Annual Revenue 2022-2023 (USD in Billions)", y="Market Cap (USD in Trillions)", color="Sector", hover_name=data.index, hover_data=relevant_cols)

#the person can hover over each dot to show all the information about that company thats present in the dataset
interactivefig.show()

# adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

#show the pie charts
plt.close()
plt.close()
plt.close()
plt.show()
