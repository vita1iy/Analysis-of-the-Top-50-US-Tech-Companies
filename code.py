import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Pie chart of "Sector" and the %
sector_count = data['Sector'].value_counts()
ax1.pie(sector_count, labels=sector_count.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 5})
ax1.set_title('Pie Chart of Sector', fontsize=12)

# Pie chart of "HQ State"
state_count = data['HQ State'].value_counts()
ax2.pie(state_count, labels=state_count.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 5})
ax2.set_title('Pie Chart of HQ State', fontsize=12)

# adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

plt.show()
