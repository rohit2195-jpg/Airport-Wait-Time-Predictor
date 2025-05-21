
from sklearn import linear_model
import pandas as pd

import datetime as dt
import holidays

import numpy as np

def xgb_model(train, test, airport_to_location, air_code):

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

    X['Date'] = pd.to_datetime(X['Date'])
    X_test['Date'] = pd.to_datetime(X_test['Date'])

    X['IsDayBeforeHoliday'] = X['Date'].apply(lambda date: 1 if (date + dt.timedelta(days=1)) in holidays.US() else 0)
    X_test['IsDayBeforeHoliday'] = X_test['Date'].apply(lambda date: 1 if (date + dt.timedelta(days=1)) in holidays.US() else 0)


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

    '''
    #best model
    
    model = xg.XGBRegressor(objective='reg:squarederror',
                             n_estimators=150,
                             learning_rate=0.1,
                             max_depth=9,
                             random_state=42,
                             )
    model.fit(X, Y)
    y_pred = model.predict(X_test)

    print("mean squared error", mean_squared_error(Y_test, model.predict(X_test)))
    print("mean absolute error", mean_absolute_error(Y_test, model.predict(X_test)))
    print("r2 score", r2_score(Y_test, model.predict(X_test)))
    '''
    '''
    #hyper parameter tuning for maxdepth
    depth = list(range(100,2000,100))
    mse = []
    r2 = []
    for i in range(0, len(depth)):
        print("training with depth", depth[i])
        model = xg.XGBRegressor(objective='reg:squarederror',
                             n_estimators=depth[i],
                             learning_rate=0.1,
                             max_depth=9,
                             random_state=42,
                             )
        model.fit(X, Y)
        mse.append(mean_squared_error(Y_test, model.predict(X_test)))
        r2.append(r2_score(Y_test, model.predict(X_test)))
    plt.plot(depth, mse)
    plt.show()
    plt.plot(depth, r2)
    plt.show()
    '''

    '''
    param_grid = {
        'max_depth': [9],
        'learning_rate': [0.1],
        'subsample': [0.7],
        'n_estimators': [ 175, 185, 200, 215, 225],

    }

    model = xg.XGBRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', verbose=1)
    grid_search.fit(X, Y)


    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    '''

    '''
    filename = os.path.join("..", "saved_models", air_code+'.pkl')
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pickle.dump(model, open(filename, 'wb'))
    '''











