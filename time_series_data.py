import yfinance as yf
import pandas as pd
import xgboost as xgb
import plotly.express as px

# List of tickers
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

# Get historical data from 2020 to 2022
start_date = '2020-01-01'
end_date = '2022-12-31'
historical_data = {}
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    historical_data[ticker] = data

# Create a dictionary to store the predicted performances
predicted_performances = {}

# Iterate through each ticker
for ticker in tickers:
    # Get the historical data for the current ticker
    data = historical_data[ticker]
    
    # Create a new DataFrame with the closing prices
    df = pd.DataFrame({'Close': data['Close']})
    
    # Create features
    for i in range(1, 4):
        col_name = f'lag_{i}'
        df[col_name] = df['Close'].shift(i)
        
    # Drop missing values
    df.dropna(inplace=True)
    
    # Set the target variable
    y = df['Close'].shift(-1).dropna()
    
    # Set the features
    X = df.drop(columns=['Close']).dropna()
    
    # Split the data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Train the XGBoost model
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, gamma=0, subsample=0.75,
                             colsample_bytree=1, max_depth=7)
    model.fit(X_train, y_train)
    
    # Make predictions for 2023
    future_data = yf.download(ticker, start='2023-01-01', end='2023-12-31')

    # Create a new DataFrame with the closing prices
    future_df = pd.DataFrame({'Close': future_data['Close']})

    # Create features
    for i in range(1, 4):
        col_name = f'lag_{i}'
        future_df[col_name] = future_df['Close'].shift(i)
    
    # Make predictions
    future_predictions = model.predict(future_df.drop(columns=['Close']))

    # Check if the number of dates in future_data matches the number of predictions
    if len(future_data) != len(future_predictions):
        print(f"Error: The number of dates in future_data ({len(future_data)}) does not match the number of predictions ({len(future_predictions)}).")
        print(future_data.shape, future_predictions.shape)

    # Create a DataFrame with the dates and predictions
    predicted_df = pd.DataFrame({'Date': future_data['Close'].index[1:], 'Prediction': future_predictions[:-1]})

    # Set the index to the date
    predicted_df.set_index('Date', inplace=True)

    # Add the predicted performance to the dictionary
    predicted_performances[ticker] = predicted_df

# Combine predicted performances into one DataFrame
combined_df = pd.concat(predicted_performances.values(), axis=1)
combined_df.columns = predicted_performances.keys()

#created an interactive chart using plotly
fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns, 
              title='Predicted Performances for 2023', 
              labels={'index': 'Date', 'value': 'Closing Price'},
              log_y=True)

fig.update_layout(
    xaxis=dict(
        tickfont_size=12,
    ),
    yaxis=dict(
        tickfont_size=12,
    ),
    legend=dict(
        title=None,
        font=dict(size=12),
    ),
    plot_bgcolor='white'
)

fig.show()