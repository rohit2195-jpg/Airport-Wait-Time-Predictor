
from sklearn import linear_model
import pandas as pd
import matplotlib as plt
from meteostat import *
from sklearn.metrics import mean_squared_error, r2_score
import datetime as dt

'''

def CustomLinearRegression(train, test, airport_to_location):

    model = linear_model.LinearRegression()
    pd.set_option('display.max_columns', None)

    X = train[["FlightCount", "HourRange", "FlightDate"]].copy()
    X_test = test[["FlightCount", "HourRange", "FlightDate"]].copy()

    X["FlightDate"] = pd.to_datetime(X["FlightDate"])
    X_test["FlightDate"] = pd.to_datetime(X_test["FlightDate"])

    X['CurrHour'] = X['HourRange'].str[:2].astype(int)
    X_test['CurrHour'] = X_test['HourRange'].str[:2].astype(int)

    X["Date"] = pd.to_datetime(X["FlightDate"].astype(str) + " " + X['CurrHour'].astype(str), format='%Y-%m-%d %H')
    X_test["Date"] =  pd.to_datetime(X_test["FlightDate"].astype(str) + " " + X_test['CurrHour'].astype(str), format='%Y-%m-%d %H')

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


    X = pd.merge(X, data, how="left", left_on="Date", right_on="time")
    X_test = pd.merge(X_test, data_test, how="left", left_on="Date", right_on="time")

    #organizing the data for regression

    X.drop(["FlightDate"], axis=1, inplace=True)
    X.drop(["CurrHour"], axis=1, inplace=True)
    X_test.drop(["FlightDate"], axis=1, inplace=True)
    X_test.drop(["CurrHour"], axis=1, inplace=True)

    # setting year to a dummy value so it doesn't influence regression
    X["Date"] = X["Date"].dt.strftime("%Y-%m-%d")
    X_test["Date"] = X_test["Date"].dt.strftime("%Y-%m-%d")

    X['HourRange'] = X['HourRange'].str[:2].astype(int)
    X_test['HourRange'] = X_test['HourRange'].str[:2].astype(int)

    X["Date"] = pd.to_datetime(X["Date"]).dt.dayofyear
    X_test["Date"] = pd.to_datetime(X_test["Date"]).dt.dayofyear

    X.fillna(value=0, inplace=True)
    X_test.fillna(value=0, inplace=True)

    print(X.head())
    print(Y.head())
    print(X_test.head())
    print(Y_test.head())


    model = model.fit(X, Y)

    predictions = model.predict(X_test)

    print("mean error: ", mean_squared_error(predictions, Y_test))
    print("r2 score: ", r2_score(predictions, Y_test))
    print("\n\n prediction: ", predictions)
    print("\n\n test data: ", Y_test)

'''





