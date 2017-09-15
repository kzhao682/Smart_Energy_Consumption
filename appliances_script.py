import pandas as pd
import numpy as np
import pickle
from script import *
from sarimax_script import *
from garch_script import *

df_meta = pickle.load(open('pickle_files/df_meta_refined.p', 'rb'))


#function to find average usage of appliances in neighborhood
def average_appliances(appliances, start=None, end=None):
    appliance_data ={}
    for appliance in appliances:
        building_id = df_meta[df_meta[appliance]=='yes'].index
        total_use = 0
        count = 0
        for building in building_id:
            try:
                data = pickle.load(open('pickle_files/df_{}.p'.format(building),'rb'))
                data = data.sort_values('localhour', ascending=True)
                data.index = data.apply(strip_tz, axis=1)
                if start and end:
                    use = data.loc[start:end, appliance]
                else:
                    use = data.loc['2016-10-01 00:00:00':'2016-10-31 00:00:00', appliance]
                if sum(use) > 0:
                    total_use += sum(use)
                    count += 1
            except:
                continue
        appliance_data[appliance] = total_use/count
        print(appliance, appliance_data[appliance])

    return appliance_data


#function to find appliance usage of building
def monthly_usage(building_id, appliances, start=None, end=None):

    appliance_data = {}

    for appliance in appliances:
        data = pickle.load(open('pickle_files/df_{}.p'.format(building_id),'rb'))
        data = data.sort_values('localhour', ascending=True)
        data.index = data.apply(strip_tz, axis=1)
        if start and end:
            total_use = data.loc[start:end, appliance]
        else:
            total_use = data.loc['2016-10-01 00:00:00': '2016-10-31 00:00:00',appliance]
        appliance_data[appliance] = total_use

    return appliance_data
