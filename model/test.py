import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import holidays
import datetime as dt

def checkAccuracy(test, airport_to_location, air_code):
    pd.set_option('display.max_columns', None)

    X_test = test[["HourRange", "FlightDate"]].copy()

    X_test["FlightDate"] = pd.to_datetime(X_test["FlightDate"])

    X_test['CurrHour'] = X_test['HourRange'].str[:2].astype(int)

    X_test["Date"] = pd.to_datetime(X_test["FlightDate"].astype(str) + " " + X_test['CurrHour'].astype(str),
                                    format='%Y-%m-%d %H')

    Y_test = test["AverageWait"]

    # organizing the data for regression


    X_test.drop(["FlightDate"], axis=1, inplace=True)
    X_test.drop(["CurrHour"], axis=1, inplace=True)

    # setting year to a dummy value so it doesn't influence regression
    X_test["Date"] = X_test["Date"].dt.strftime("%Y-%m-%d")

    X_test['HourRange'] = X_test['HourRange'].str[:2].astype(int)

    X_test["Day"] = pd.to_datetime(X_test["Date"]).dt.day

    X_test["DayofWeek"] = pd.to_datetime(X_test["Date"]).dt.dayofweek

    X_test["Month"] = pd.to_datetime(X_test["Date"]).dt.month

    X_test['IsHoliday'] = X_test['Date'].apply(lambda date: 1 if date in holidays.US() else 0)

    X_test['Date'] = pd.to_datetime(X_test['Date'])

    X_test['IsDayBeforeHoliday'] = X_test['Date'].apply(
        lambda date: 1 if (date + dt.timedelta(days=1)) in holidays.US() else 0)

    X_test.drop("Date", axis=1, inplace=True)

    X_test.fillna(value=0, inplace=True)


    X_test["hour_sin"] = np.sin(2 * np.pi * X_test["HourRange"] / 24)
    X_test["hour_cos"] = np.cos(2 * np.pi * X_test["HourRange"] / 24)
    X_test.drop("HourRange", axis=1, inplace=True)


    X_test["dow_sin"] = np.sin(2 * np.pi * X_test["DayofWeek"] / 7)
    X_test["dow_cos"] = np.cos(2 * np.pi * X_test["DayofWeek"] / 7)
    X_test.drop("DayofWeek", axis=1, inplace=True)


    X_test["month_sin"] = np.sin(2 * np.pi * X_test["Month"] / 12)
    X_test["month_cos"] = np.cos(2 * np.pi * X_test["Month"] / 12)
    X_test.drop("Month", axis=1, inplace=True)


    X_test["day_sin"] = np.sin(2 * np.pi * X_test["Day"] / 31)
    X_test["day_cos"] = np.cos(2 * np.pi * X_test["Day"] / 31)
    X_test.drop("Day", axis=1, inplace=True)

    #print(X.head())


    #best model

    loaded_model = pickle.load(open(os.path.join("..", "saved_models", air_code + '.pkl'), 'rb'))

    loaded_prediction = loaded_model.predict(X_test)
    print("mean squared error", mean_squared_error(Y_test, loaded_model.predict(X_test)))
    print("mean absolute error", mean_absolute_error(Y_test, loaded_model.predict(X_test)))
    print("r2 score", r2_score(Y_test, loaded_model.predict(X_test)))

    test["DayAndHour"] = test["FlightDate"] + " " + test["HourRange"]

    plt.figure(figsize=(16, 6))

    plt.plot(test["DayAndHour"], loaded_prediction, color='red', label='Prediction')
    plt.plot(test["DayAndHour"], Y_test, color='blue', label='Actual')

    plt.xlabel('Date')
    plt.ylabel('Predictions')
    plt.title( "Predicted vs Actual Wait times for  " + air_code)
    plt.legend()
    N = 10
    xticks = np.arange(0, len(test["DayAndHour"]), N)
    plt.xticks(xticks, test["DayAndHour"].iloc[xticks], rotation=45, fontsize=8)

    plt.figtext(0.1, -0.05, f'MSE: {mean_squared_error(Y_test, loaded_model.predict(X_test)):.2f} ', ha='left', fontsize=10)

    plt.tight_layout()  # Make sure everything fits without overlap
    plt.show()




airport_to_location = {
    'ORD': (41.978611, -87.904724),
}
filepath = './test/Awt.cbp.gov_ORD_2025-26-05-2025-01-06.csv'

for file in os.listdir("./test"):
    if file.endswith(".csv"):
        filepath = file
        file = os.path.join(".", "test", file)
        data = pd.read_csv(file)
        aircode = filepath[filepath.find('_') + 1:filepath.find('_') + 4]
        test_set = data
        print(aircode)
        checkAccuracy(test_set, airport_to_location, aircode)





