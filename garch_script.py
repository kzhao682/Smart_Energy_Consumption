import arch
from script import *
from sarimax_script import *
from datetime import datetime, timedelta



def fit_garch(data):

    arch_model = arch.arch_model(data.Hourly_Usage, p=1, q=1)
    results = arch_model.fit(update_freq=5)

    return results


def get_garch(data, model, building_id, final_date=None):

    start_train, end_train, start_test, end_test = find_dates(4874)

    arch_forecast = model.forecast(start=start_test)

    x = data.loc[start_train:end_test,'localhour']
    y_actual= data.loc[start_test:end_test, 'Hourly_Usage']
    y_predict = arch_forecast.variance

    return x, y_actual, y_predict



#function to plot GARCH forecast and mean squared error
def plot_garch(data, model, building_id, final_date=None):

    start_train, end_test = find_egauge_dates(4874)
    time_delta_1 = timedelta(days=1)
    time_delta_2 = timedelta(hours=1)
    end_train = end_test - time_delta_1
    start_test = end_train + time_delta_2
    start_plot = end_test - timedelta(days=4)
    start_train = str(start_train)
    end_train = str(end_train)
    start_test = str(start_test)
    end_test = str(end_test)
    start_plot = str(start_plot)

    plt.subplots(1,1,figsize=(20,6))
    arch_forecast = model.forecast(start=start_test)
    plt.plot(data.localhour, arch_forecast.variance)
    plt.plot(data.loc[start_plot:end_test, 'localhour'],data.loc[start_plot:end_test, 'Hourly_Usage'])
    plt.legend()

    return mean_squared_error(data.loc[start_test:,'Hourly_Usage'], arch_forecast.variance[-24:])
