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


df_meta_refined = pickle.load(open('pickle_files/df_meta_refined.p', 'rb'))


#function to strip timezone from datetime
def strip_tz(row):
        return row['localhour'].strftime('%Y-%m-%d %H:%M:%S')


#function to indicate weekends
def is_weekend(row):
    if row['Day_of_Week'] > 4:
        return 1
    else:
        return 0


#load dataframe for building
def load_building(building_id, start, end):

    df = pickle.load(open('pickle_files/df_{}.p'.format(building_id), 'rb'))
    df = df.sort_values('localhour', ascending=True)
    df.index = pd.to_datetime(df.apply(strip_tz, axis=1))
    df = df.rename(columns={'use':'Hourly_Usage'})
    df = df.loc[start:end,['localhour','Hourly_Usage']]

    return df


#function to load weather of respective city
def load_weather(city):

    df = pickle.load(open('weather_{}.p'.format('austin'), 'rb'))
    df = df.sort_values('localhour', ascending=True)
    df.index = pd.to_datetime(df.apply(strip_tz, axis=1))

    return df



#function to find latest month of egauge data
def find_egauge_dates(dataid, length=30, final_date=None):

    df = pickle.load(open('pickle_files/df_{}.p'.format(dataid),'rb'))

    if not final_date:
        final_date = df_meta_refined.loc[dataid, 'egauge_max_time']

    final_date = datetime.strftime(final_date,'%Y-%m-%d %H:%M:%S')
    final_date = datetime.strptime(final_date, '%Y-%m-%d %H:%M:%S')

    max_date = datetime.strptime('2016-10-31 00:00:00', '%Y-%m-%d %H:%M:%S')

    if final_date > max_date:
        final_date = max_date

    end_date = datetime.strptime('{}-{}-{} 00:00:00'.format(final_date.year, final_date.month, final_date.day), '%Y-%m-%d %H:%M:%S')
    time_delta = timedelta(days=length)
    start_date = end_date - time_delta

    return start_date, end_date


