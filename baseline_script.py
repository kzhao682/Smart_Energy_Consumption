import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from dateutil.relativedelta import *
import statsmodels.tsa.api as smt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import pickle
from script import *


#function to plot AR forecast and MSE
def ar_scores(data, building_id, p, length=1, total_length=30):

    start_train, end_train, start_test, end_test = find_dates(building_id, length, total_length)
    df_baseline_train, df_baseline_test = create_train_test(data, start_train, end_train, start_test, end_test, 24*length)
    df_baseline_train.loc[start_test:,'Hourly_Usage'] = np.nan
    ar_baseline = sm.tsa.statespace.SARIMAX(endog=df_baseline_train.Hourly_Usage,
                                  trend=None,
                                  order=(p, 0, 0),
                                  seasonal_order=(0, 0, 0, 0),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
    results = ar_baseline.fit()
    mse, rmse = add_forecast(results, df_baseline_test, df_baseline_train, start_test, end_test)
    plot_forecast(df_baseline_train, 500)

    return mse, rmse


#function to plot MA forecast and MSE
def ma_scores(data, building_id, q, length=1, total_length=30):

    start_train, end_train, start_test, end_test = find_dates(building_id, length, total_length)
    df_baseline_train, df_baseline_test = create_train_test(data, start_train, end_train, start_test, end_test, 24*length)
    df_baseline_train.loc[start_test:,'Hourly_Usage'] = np.nan
    ar_baseline = sm.tsa.statespace.SARIMAX(endog=df_baseline_train.Hourly_Usage,
                                  trend=None,
                                  order=(0, 0, q),
                                  seasonal_order=(0, 0, 0, 0),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
    results = ar_baseline.fit()
    mse, rmse = add_forecast(results, df_baseline_test, df_baseline_train, start_test, end_test)
    plot_forecast(df_baseline_train, 500)

    return mse, rmse
