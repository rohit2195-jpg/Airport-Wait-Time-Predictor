
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime as dt
import holidays
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os


def gradient_boost(train, test, airport_to_location, air_code):

    model = linear_model.LinearRegression()
    pd.set_option('display.max_columns', None)

    X = train[["HourRange", "FlightDate"]].copy()
    X_test = test[["HourRange", "FlightDate"]].copy()

    X["FlightDate"] = pd.to_datetime(X["FlightDate"])
    X_test["FlightDate"] = pd.to_datetime(X_test["FlightDate"])

    X['CurrHour'] = X['HourRange'].str[:2].astype(int)
    X_test['CurrHour'] = X_test['HourRange'].str[:2].astype(int)

    X["Date"] = pd.to_datetime(X["FlightDate"].astype(str) + " " + X['CurrHour'].astype(str), format='%Y-%m-%d %H')
    X_test["Date"] = pd.to_datetime(X_test["FlightDate"].astype(str) + " " + X_test['CurrHour'].astype(str),
                                    format='%Y-%m-%d %H')

    Y = train["AverageWait"]
    Y_test = test["AverageWait"]

    # getting weather data for testing and training data

    start = X["FlightDate"].min()
    end = X["FlightDate"].max()
    data = Hourly('72219', start, end)
    data = data.fetch()

    start_test = X_test["FlightDate"].min()
    end_test = X_test["FlightDate"].max()
    data_test = Hourly('72219', start_test, end_test)
    data_test = data_test.fetch()

    data = data[["temp", "wspd"]]
    data_test = data_test[["temp", "wspd"]]

    X = pd.merge(X, data, how="left", left_on="Date", right_on="time")
    X_test = pd.merge(X_test, data_test, how="left", left_on="Date", right_on="time")

    # organizing the data for regression

    X.drop(["FlightDate"], axis=1, inplace=True)
    X.drop(["CurrHour"], axis=1, inplace=True)
    X_test.drop(["FlightDate"], axis=1, inplace=True)
    X_test.drop(["CurrHour"], axis=1, inplace=True)

    # setting year to a dummy value so it doesn't influence regression
    X["Date"] = X["Date"].dt.strftime("%Y-%m-%d")
    X_test["Date"] = X_test["Date"].dt.strftime("%Y-%m-%d")

    X['HourRange'] = X['HourRange'].str[:2].astype(int)
    X_test['HourRange'] = X_test['HourRange'].str[:2].astype(int)


    X["DayofWeek"] = pd.to_datetime(X["Date"]).dt.dayofweek
    X_test["DayofWeek"] = pd.to_datetime(X_test["Date"]).dt.dayofweek

    X["Month"] = pd.to_datetime(X["Date"]).dt.month
    X_test["Month"] = pd.to_datetime(X_test["Date"]).dt.month

    X['IsHoliday'] = X['Date'].apply(lambda date: 1 if date in holidays.US() else 0)
    X_test['IsHoliday'] = X_test['Date'].apply(lambda date: 1 if date in holidays.US() else 0)

    X.drop("Date", axis=1, inplace=True)
    X_test.drop("Date", axis=1, inplace=True)

    X.drop("temp", axis=1, inplace=True)
    X_test.drop("temp", axis=1, inplace=True)
    X.drop("wspd", axis=1, inplace=True)
    X_test.drop("wspd", axis=1, inplace=True)

    print(X.head())

    X.fillna(value=0, inplace=True)
    X_test.fillna(value=0, inplace=True)


    model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=5, loss="squared_error")
    model.fit(X, Y)

    predictions = model.predict(X_test)

    print("mean squared error: ", mean_squared_error(Y_test, model.predict(X_test)))
    print("mean absolute error: ", mean_absolute_error(Y_test, model.predict(X_test)))
    print("r2_score: ", r2_score(Y_test, model.predict(X_test)))

    filename = os.path.join("..", "saved_models", air_code+'.pkl')
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pickle.dump(model, open(filename, 'wb'))












