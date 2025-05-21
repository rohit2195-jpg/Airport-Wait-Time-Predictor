
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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf


def nueral_net(train, test, airport_to_location):
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

    X["Day"] = pd.to_datetime(X["Date"]).dt.day
    X_test["Day"] = pd.to_datetime(X_test["Date"]).dt.day

    X["DayofWeek"] = pd.to_datetime(X["Date"]).dt.dayofweek
    X_test["DayofWeek"] = pd.to_datetime(X_test["Date"]).dt.dayofweek

    X["Month"] = pd.to_datetime(X["Date"]).dt.month
    X_test["Month"] = pd.to_datetime(X_test["Date"]).dt.month

    X['IsHoliday'] = X['Date'].apply(lambda date: 1 if date in holidays.US() else 0)
    X_test['IsHoliday'] = X_test['Date'].apply(lambda date: 1 if date in holidays.US() else 0)

    X.drop("Date", axis=1, inplace=True)
    X_test.drop("Date", axis=1, inplace=True)



    X.fillna(value=0, inplace=True)
    X_test.fillna(value=0, inplace=True)

    X["hour_sin"] = np.sin(2 * np.pi * X["HourRange"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["HourRange"] / 24)
    X_test["hour_sin"] = np.sin(2 * np.pi * X_test["HourRange"] / 24)
    X_test["hour_cos"] = np.cos(2 * np.pi * X_test["HourRange"] / 24)
    X.drop("HourRange", axis=1, inplace=True)
    X_test.drop("HourRange", axis=1, inplace=True)

    X["dow_sin"] = np.sin(2 * np.pi * X["DayofWeek"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["DayofWeek"] / 7)
    X_test["dow_sin"] = np.sin(2 * np.pi * X_test["DayofWeek"] / 7)
    X_test["dow_cos"] = np.cos(2 * np.pi * X_test["DayofWeek"] / 7)
    X.drop("DayofWeek", axis=1, inplace=True)
    X_test.drop("DayofWeek", axis=1, inplace=True)

    X["month_sin"] = np.sin(2 * np.pi * X["Month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["Month"] / 12)
    X_test["month_sin"] = np.sin(2 * np.pi * X_test["Month"] / 12)
    X_test["month_cos"] = np.cos(2 * np.pi * X_test["Month"] / 12)
    X.drop("Month", axis=1, inplace=True)
    X_test.drop("Month", axis=1, inplace=True)

    X["day_sin"] = np.sin(2 * np.pi * X["Day"] / 31)
    X["day_cos"] = np.cos(2 * np.pi * X["Day"] / 31)
    X_test["day_sin"] = np.sin(2 * np.pi * X_test["Day"] / 31)
    X_test["day_cos"] = np.cos(2 * np.pi * X_test["Day"] / 31)
    X.drop("Day", axis=1, inplace=True)
    X_test.drop("Day", axis=1, inplace=True)

    print(X.head())

    tf.random.set_seed(42)

    # Create a model using the Sequential API
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])

    model.fit(X, Y, epochs=200)


    print("mean squared error: ", mean_squared_error(Y_test, model.predict(X_test)))
    print("r2_score: ", r2_score(Y_test, model.predict(X_test)))











    















