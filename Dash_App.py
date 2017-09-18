import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import statsmodels.api as sm
from start_script import *
from arima_script import *
from garch_script import *
from appliances_script import *
from baseline_script import *
import itertools


import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.dashboard_objs as dashboard
import IPython.display
from IPython.display import Image


df = load_building(545, '2016-7-01 00:00:00', '2016-10-31 00:00:00')
df_weather_austin = load_weather('austin')


start_trainx, end_trainx, start_testx, end_testx = find_dates(545, 10, 120)
df_trainx, df_testx = create_train_test(df, start_trainx, end_trainx, start_testx, end_testx, 240)

df_trainx = pickle.load(open('Dash/sarimax_30.p', 'rb'))


app = dash.Dash()


app.layout = html.Div(children=[
    html.H1(children='Past 30 Days'),

    html.Div(children='''
        Next Day Prediction
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df_trainx.loc[start_trainx:end_testx,:].index, 
                 'y': df_trainx.loc[start_trainx:end_testx,'Hourly_Usage'], 'type': 'scatter', 'name': 'Hourly_Usage'},
                {'x': df_trainx.loc[start_trainx:end_testx,:].index, 
                 'y': df_trainx.loc[start_trainx:end_testx,'forecast'], 'type': 'scatter', 'name': 'Forecast'}
            ],
            'layout': {
                'title': '30 Days SARIMAX'
            }
        }
    )
])



if __name__ == '__main__':
    app.run_server(debug=True)
