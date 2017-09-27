# Overview:

Global carbon emission must be reduced in order to combat the effects of climate change. The reduction of 
building energy consumption alone can have a very beneficial effect on the environment as it currently 
accounts for 40% of carbon emissions due to fossil fuels. Green building technologies is central to
developing a safer and more sustainable planet.

This project explores the potential of time series analysis to forecast building energy consumption gathered
by smart meter data. Harnessing the power of IoT technologies is key to developing more eco-conscious and
sustainable behavior. The best time series model was a seasonal ARIMA  model with exogenous variables including 
appliance usage as well as weather. This model was included in the interactive dashboard, which is a platform
to inform users of their energy consumption behaviors and encourage them to develop more sustainable practices.

# Data:

The data was granted by the Pecan Street Dataport, which operates on a PostgreSQL database. The Dataport contains
information on energy and water consumption of individual building units. This project solely focuses on electricity
consumption data measured in kWh at an hourly level. It also includes detailed information on energy consumption
due to specific appliances as well as hourly data on various weather features.

# Tools:

- `SQLAlchemy` - acquire PostgreSQL data
- `Statsmodels` - build ARIMA time series models
- `Arch` - build GARCH time series models
- `FBProphet` - build Facebook Prophet time series models
- `Scikit-learn` - regression models to predict monthly energy consumption based on physical features of building
- `Plot.ly` - data visualization and interactive dashboard
