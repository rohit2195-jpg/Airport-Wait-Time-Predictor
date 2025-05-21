import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sklearn
import numpy as np
import pickle
import pandas as pd
import holidays
import pytz
import datetime as dt

def datetime_to_unix(dt, timezone_str):

    timezone = pytz.timezone(timezone_str)
    dt_aware = timezone.localize(dt)
    timestamp = dt_aware.timestamp()
    return timestamp

app = Flask(__name__)
CORS(app, support_credentials=True)

iata_to_time = {
    'ORD': 'America/Chicago'
}

iata_to_icaoUS = {
}
iata_file = open("iata-icao.csv","r")
lines = iata_file.readlines()
for line in lines:
    line = line.strip().split(",")
    if (line[0].strip("\"") == "US"):
        iata_to_icaoUS[line[2].strip("\"")] = line[3].strip("\"")


@app.route('/live', methods=['POST'])
def live():
    print("hello")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    loaded_model = pickle.load(open(os.path.join("..", "saved_models", data["air_code"] + '.pkl'), 'rb'))

    X_test = pd.DataFrame([data])
    X_test["Date"] = pd.to_datetime(X_test["day"]+ " " + X_test["month"] + " " + X_test["year"] + " " + X_test["HourRange"], format="%d %m %Y %H")
    X_test["DayofWeek"] = X_test["Date"].dt.dayofweek
    X_test["Month"] = X_test["Date"].dt.month
    X_test['IsHoliday'] = X_test['Date'].apply(lambda date: 1 if date in holidays.US() else 0)

    X_test['Date'] = pd.to_datetime(X_test['Date'])

    X_test['IsDayBeforeHoliday'] = X_test['Date'].apply(
        lambda date: 1 if (date + dt.timedelta(days=1)) in holidays.US() else 0)

    X_test["hour_sin"] = np.sin(2 * np.pi * X_test["HourRange"].astype(int) / 24)
    X_test["hour_cos"] = np.cos(2 * np.pi * X_test["HourRange"].astype(int) / 24)
    X_test.drop("HourRange", axis=1, inplace=True)


    X_test["dow_sin"] = np.sin(2 * np.pi * X_test["DayofWeek"] / 7)
    X_test["dow_cos"] = np.cos(2 * np.pi * X_test["DayofWeek"] / 7)
    X_test.drop("DayofWeek", axis=1, inplace=True)


    X_test["month_sin"] = np.sin(2 * np.pi * X_test["Month"] / 12)
    X_test["month_cos"] = np.cos(2 * np.pi * X_test["Month"] / 12)
    X_test.drop("Month", axis=1, inplace=True)


    X_test["day_sin"] = np.sin(2 * np.pi * X_test["day"].astype(int) / 31)
    X_test["day_cos"] = np.cos(2 * np.pi * X_test["day"].astype(int) / 31)

    X_test.drop("Date", axis=1, inplace=True)
    X_test.drop("day", axis=1, inplace=True)
    X_test.drop("month", axis=1, inplace=True)
    X_test.drop("year", axis=1, inplace=True)
    X_test.drop("air_code", axis=1, inplace=True)
    #X_test.drop("temp", axis=1, inplace=True)
    #X_test.drop("wspd", axis=1, inplace=True)

    print(X_test)
    loaded_prediction = loaded_model.predict(X_test)
    print(loaded_prediction)
    return jsonify(loaded_prediction[0].astype(float))

@app.route('/graph', methods=['POST'])
def graph():
    data = request.get_json()
    print(data)
    loaded_model = pickle.load(open(os.path.join("..", "saved_models", data["air_code"] + '.pkl'), 'rb'))

    ans = []

    X_test = pd.DataFrame([data])
    X_test["Date"] = pd.to_datetime(X_test["day"]+ " " + X_test["month"] + " " + X_test["year"], format="%d %m %Y")
    X_test["DayofWeek"] = X_test["Date"].dt.dayofweek
    X_test["Month"] = X_test["Date"].dt.month
    X_test['IsHoliday'] = X_test['Date'].apply(lambda date: 1 if date in holidays.US() else 0)

    X_test['Date'] = pd.to_datetime(X_test['Date'])

    X_test['IsDayBeforeHoliday'] = X_test['Date'].apply(
        lambda date: 1 if (date + dt.timedelta(days=1)) in holidays.US() else 0)

    X_test["hour_sin"] = np.sin(2 * np.pi * X_test["HourRange"].astype(int) / 24)
    X_test["hour_cos"] = np.cos(2 * np.pi * X_test["HourRange"].astype(int) / 24)
    X_test.drop("HourRange", axis=1, inplace=True)

    X_test["dow_sin"] = np.sin(2 * np.pi * X_test["DayofWeek"] / 7)
    X_test["dow_cos"] = np.cos(2 * np.pi * X_test["DayofWeek"] / 7)
    X_test.drop("DayofWeek", axis=1, inplace=True)

    X_test["month_sin"] = np.sin(2 * np.pi * X_test["Month"] / 12)
    X_test["month_cos"] = np.cos(2 * np.pi * X_test["Month"] / 12)
    X_test.drop("Month", axis=1, inplace=True)

    X_test["day_sin"] = np.sin(2 * np.pi * X_test["day"].astype(int) / 31)
    X_test["day_cos"] = np.cos(2 * np.pi * X_test["day"].astype(int) / 31)

    X_test.drop("Date", axis=1, inplace=True)
    X_test.drop("day", axis=1, inplace=True)
    X_test.drop("month", axis=1, inplace=True)
    X_test.drop("year", axis=1, inplace=True)
    X_test.drop("air_code", axis=1, inplace=True)

    print(X_test)
    for i in range(0, 24):
        X_test["hour_sin"] = np.sin(2 * np.pi * i / 24)
        X_test["hour_cos"] = np.cos(2 * np.pi * i / 24)
        prediction = loaded_model.predict(X_test)
        ans.append((i, prediction[0].astype(float)))

    return jsonify(ans)

app.run(port=5001, debug=True)