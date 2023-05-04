## Code.py ##

uses the Dataset.csv file which is the dataset from keggle (https://www.kaggle.com/datasets/lamiatabassum/top-50-us-tech-companies-2022-2023-dataset)
After running, the dataset will be cleaned and will be easily readable by the person viewing It
also creates two pie charts that show the percentage of companies that are in each sector and the % of the HQ state that the companies are located in
this provides visualization of what is the main technology that the companies are in and where these companies are mostly located
at the end it also provides an interactive chart with plotly that displays all the companies on a scatter plot with x axis as annual revenue and y variable as market cap
you are able to also see all the details about each company by putting your cursor above each companies dot that would display the information from the dataset

## ticker_list ##

Created manually to easily see the list of all the stock tickers

## time_series_data.py ##

Here Data is collected from yahoo finance over the span of 2 years(begining of 2020 to end of 2022). 
I then create a dataframe to show only closing prices. 
After i utilize xgboost library to start my prediction. 
Each ticker has lagged features which are the previos days closing price for the XGBoost model and also creates separate datas for training and testing. 
All the predictions are stored into a dataframe.
Then the predicted perfomances in this case are the closing prices are plotted on a plotly graph.
this graph allows the person who is viewing the graph to how each stock performs over 2023.
It also allows to disable/enable which stocks to be viewed by clicking on the stock in the legend.
and displays the specific data for each data by hovering the mouse cursor over the line

