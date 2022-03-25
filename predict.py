import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

stock = pd.read_csv("sphist.csv")
stock["Date"] = pd.to_datetime(stock["Date"])
boolean = stock["Date"] > datetime(year=2015, month=4, day=1)
stock = stock.sort_values(by="Date", ascending=True)
# print(stock.columns)
stock["day_5_price"] = stock.rolling(window=5, 
                                     on="Date").mean().loc[:,"Close"]
stock["day_30_price"] = stock.rolling(window=30, 
                                      on="Date").mean().loc[:,"Close"]
stock["day_365_price"] = stock.rolling(window=365, 
                                       on="Date").mean().loc[:,"Close"]
stock["mean_ratio"] = stock["day_5_price"] / stock["day_365_price"]
stock["day_5_std"] = stock.rolling(window=5, 
                                   on="Date").std().loc[:,"Close"]
stock["day_365_std"] = stock.rolling(window=5, 
                                     on="Date").std().loc[:,"Close"]
stock["std_ratio"] = stock["day_5_std"] / stock["day_365_std"]

# print(stock["day_5"].head(7))
stock = stock.shift(periods=1)
# print(stock.head(7))
stock = stock[stock["Date"] > datetime(year=1951, month=1, day=3)]
stock = stock.dropna(axis=0).copy()
# print(stock['Close'].isnull().sum())
train = stock[stock["Date"] < datetime(year=2013, month=1, day=1)]
test = stock[stock["Date"] >= datetime(year=2013, month=1, day=1)]
# print(len(train), len(test))
model = LinearRegression()
features = train.drop(['Close', 'High', 'Low', 'Open', 'Volume', 
                    'Adj Close', 'Date'], axis=1).copy()
target = train["Close"]
model.fit(features, target)
test_features = test.drop(['Close', 'High', 'Low', 'Open', 'Volume', 
                    'Adj Close', 'Date'], axis=1).copy()
predictions = model.predict(test_features)
mae = mean_absolute_error(test["Close"], predictions)
print(len(test['Close']))
print(len(predictions))
print(test.columns)