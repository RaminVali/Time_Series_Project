
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
# Wrangling and Data Source Libraries
import os
import json
import requests
# Viz and EDA Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime
# Stats Libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
# Metrics
from sklearn.metrics import mean_squared_error
# Arch
from arch import arch_model
# Prophet
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

 
"""
# Consumer Price Index Forecast App
Consumer Price Index for All Urban Consumers (CPI-U) is a common indicator of inflation and rose 0.1 percent in March on a seasonally
adjusted basis, after increasing 0.4 percent in February, the U.S. Bureau of Labor Statistics [reported] (https://www.bls.gov/news.release/cpi.nr0.htm)
today. 

In this project we access, parse, visualize and analyze the data from the U.S. Bureau of Labor Statistics, and use it to predict the inflation rate uing 
[ARIMA](https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html) and [Prophet](https://facebook.github.io/prophet/).

## Getting or Refreshing the Data
"""

TODAY = datetime.today().strftime("%Y-%m-%d")
STARTYEAR = 1980
st.write(f'Today is {TODAY}')
# Finding the current month to see if we have uptodate data.
currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year

"""
Using json from the [BLC Public API](https://beta.bls.gov/dataQuery/find?fq=survey:[cu]&s=popularity:D)
Consumer Price Index for All Urban Consumers (CPI-U)
- CUUR0000SA0 unadjusted
- CUSR0000SA0 seasonaly adjusted

The monthly release dates for the CPI are indicated (here)[https://www.bls.gov/schedule/news_release/cpi.htm].

Since the data for each month is released before the 15th of the following month, we will use the 15th as he trigger date for updates. """


def get_json_data(currentYear):
    """
    Function ussing the API to get the jason data up to the current uear when called. Returns a json file
    """
    headers = {'Content-type': 'application/json'}
    
    all_json_data = []
    step = 20 #in years
    for year in range (STARTYEAR,currentYear+1,step):
        parameters = json.dumps({"registrationkey":st.secrets['registrationkey'], "seriesid":['CUSR0000SA0'], "startyear":str(year), "endyear":str(year+step), "calculations":"true"})
        response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data = parameters, headers = headers)
        json_data = json.loads(response.text)
        all_json_data.append(json_data)
    with open ('data/all_json_data.json','w') as outfile:
        json.dump(all_json_data, outfile)
    return all_json_data

def parse_json(file):
    """
        Function to loop through and parse the outpot json from the BLS website
    """    
    data_dict =dict()
    for section in file:
        for part in section['Results']['series']:
            for item in part['data']:
                year = item['year']
                month = item['periodName']
                value = item['value']
                data_dict[month+' '+ year] = value
    df = pd.DataFrame(data_dict.items(),columns = ['date','CPI'])
    df['CPI'] = df['CPI'].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df
# Checking and loading data
path = 'data/all_json_data.json'
if os.path.isfile(path):
    # if the file exists
    with open (path,'r') as openfile:
        all_json_data = json.load(openfile)
    df = parse_json(all_json_data)

    if (currentMonth - df.index[-1].month == 2) & (currentDay>=15): #refresh data if new daya is available based on schedule.
        all_json_data = get_json_data(currentYear)
        df = parse_json(all_json_data)
    else:
        pass
else:
    all_json_data = get_json_data(currentYear)
    df = parse_json(all_json_data)

df = df[df.index>'01/12/1983'].copy()

"""Here is the dataframe we will be working with:"""
st.write(df.tail(5))

"""## Visualising and Exploring the Data
A given time series is thought to consist of three systematic components including level, trend, seasonality, and one non-systematic component called noise.

These components are defined as [follows](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/):

- Level: The average value in the series.
- Trend: The increasing or decreasing value in the series.
- Seasonality: The repeating short-term cycle in the series.
- Noise: The random variation in the series."""
def plotly_plot (df, x , y, title, width, height): # Plotly plotting function for time series
    """Function to create plotly plots for me"""
    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=list(x), y=list(y)))
    # Add range slider
    fig.update_layout(        
        title_text=title,
        width=width,
        height=height,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
    margin=dict(l=20, r=20, t=60, b=20),
)
    return fig

fig =plotly_plot(df,df.index, df.CPI, 'Raw Consumer Price Index', 700, 300)
st.plotly_chart(fig)

# Plot percentage
df['pct-change'] =df.CPI.pct_change(periods=12)*100
fig = plotly_plot(df,df.index, df['pct-change'], 'Consumer Price Index Percentage Chnage', 700, 300)
st.plotly_chart(fig)



# Data trends and rolling statistics
numfig = 5
rmean=df['CPI'].rolling(window=12).mean()
rstd=df['CPI'].rolling(window=12).std()
# First order differencing
diff = df['CPI']-df['CPI'].shift(1)
diff = diff.dropna()
detrend = pd.DataFrame(diff, index = diff.index)
fig, ax = plt.subplots(numfig,1, sharex = True, figsize = (10,20))
ax[0].plot(df['CPI'] , color='black',label='Original')
ax[0].plot(rmean , color='red',label='Rolling Mean')
ax[1].plot(rstd,color='blue',label = 'Rolling Standard Deviation')
ax[2].plot(df['CPI']-rmean , color='green',label='Residual of CPI and Rolling Mean')
ax[3].plot((df['CPI']-rmean)/rstd , color='darkorchid',label='(CPI-Rolling Mean)/Rolling std')
ax[4].plot(diff , color='lime', label='Differencing order of 1')
for i in range(0,numfig):
 ax[i].legend(loc='best')
ax[0].set_title("Data Trends and Rolling Statistics", fontdict = {'fontsize':22})
ax[4].set_xlabel('Years',fontsize = 22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=10)
st.pyplot(fig)

"""
As can be seen form the data trend, the CPI data is not stationary. To ensure statiinarity, we can use the differencing or
normalisation by rolling mean. We used the [KPSS](https://en.wikipedia.org/wiki/KPSS_test) and 
[ADF](https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test) tests to determine stationarity after the deternding operations.
Detrending by subtracting the rolling mean is chosen to make the data stationary.
"""

# YoY % Change
plot_df = df[df.index>='01/01/2000']
plot_df['date'] = plot_df.index
plot_df['month'] = plot_df['date'].apply(lambda x: x.month)
plot_df['year'] = plot_df['date'].apply(lambda x: x.year)
fig = px.box(plot_df, x="month", y="pct-change",
            hover_data='year',
            points = 'all',
            labels = {'pct-change':'CPI Percentage Change', 'month':'Month'})
fig.update_layout(
    xaxis = dict(tickmode = 'linear'),
        title_text="YoY CPI Percentage Change from 2000")
st.plotly_chart(fig)
"""We can see that June and October have the largest outliers in terms of the percetange CPI change. 
         Also note how the highest CPI rate increases belongs to 2022."""

# Rain Cloud Chart
import ptitprince as pt
fig, ax = plt.subplots(figsize=(10, 15))
sns.set_style("darkgrid")
pt.RainCloud(x = plot_df.index.year, y = plot_df['pct-change'], data = plot_df, palette = "Set2", bw = .09,
                 width_viol = 2, ax = ax, orient = 'h')
ax.set_title("Percentage Change of CPI From 2000 to the Present Day",fontsize = 22)
ax.set_xlabel('Consumer Price Index Percentge Change', fontsize = 18)
ax.set_ylabel('Years', fontsize = 18)

st.pyplot(fig)
"""Visualising the distribution of percentage CPI change per year from the year 2000 up to the present, we can see the increase
         in the change of the consumer price index. Year 2008, 2009 and 2021 had the highest spread of CPI percentage point change."""

##### ARIMA or FB PROPHET #####
""" ## Forecasting
In this section we use ARIMA, GARCH and Fb Prophetto forecast the future CPI values."""
option = st.selectbox('Pick the forecasting method:', ['ARIMA', 'FB Prophet'])
if st.button('Submit Choice'):
    if option == 'ARIMA':
        """ The CPI has a clear trend and is non-stationary, so we make it stationary by detrending using rolling average
        and then add the average to the predicted values to get the final prediction."""

        detrend['CPI-rolMean'] = (df['CPI']-rmean)
        detrend.dropna(inplace=True)
        #ARIMA
        arima_model = ARIMA(detrend['CPI-rolMean'], order = (1,0,2),freq="MS")
        arima_fit = arima_model.fit()
        ARIMA_forecast = arima_fit.forecast(steps=12) # We forecast for 12 months. 
        st.write("### The ARIMA Model Summary is:")
        st.write(arima_fit.summary())

        def forecster (df, preds):    
            total_pred = df['CPI'] #creating structure for the future ARIMA forecast
            rmean=total_pred.rolling(window=12).mean()
            for p in preds: #each value for the forecast is the addition of the ARIMA predictions witht he rolling mean of the previous month. The new rolling mean is recalculated every time. 
                monthly_pred_row = {'CPI':p+rmean[len(rmean)-1]}
                total_pred = total_pred.append(pd.Series(monthly_pred_row), ignore_index=True)
                rmean=total_pred.rolling(window=12).mean()
            total_pred.index= df.index.append(ARIMA_forecast.index)
            total_pred = pd.Series(total_pred)
            return total_pred
        total_pred_ARIMA = forecster(df, ARIMA_forecast)

        residuals = arima_fit.resid[1:]
        fig, ax = plt.subplots(2,1)
        residuals.plot(title='Residuals', ax=ax[0])
        residuals.plot(title='Density', kind='kde', ax=ax[1])
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
        fig.tight_layout()
        st.pyplot(fig)





        #GARCH
        st.write('### The GARCH Model Summary is')
        garch = arch_model(detrend['CPI-rolMean'], vol='garch', p=1, o=0, q=1)
        garch_fitted = garch.fit()
        st.write(garch_fitted.summary())
        # one-step out-of sample forecast
        GARCH_forecast = garch_fitted.forecast(horizon=12).variance[-1:].T # Checked the comments on the original source, have to use variance here as opposed to the mean. 
        GARCH_forecast =  pd.Series(list(GARCH_forecast['2023-03-01']), index = ARIMA_forecast.index)
        total_pred_GARCH = forecster(df, GARCH_forecast)
        
        fig = garch_fitted.plot()
        fig.set_size_inches(5, 3) # Bug in program, have to assign to a to avoid duplicate plotting
        st.pyplot(fig)
        
        fig = plt.figure(figsize = (5,3))
        sns.kdeplot(garch_fitted.resid)
        st.pyplot(fig)

        """# Forecasting Results"""
        ## PLOTTING
        # Plotting the predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = df[df.index>'01-12-2019'].index, y = df[df.index>'01-12-2019']['CPI'], mode = 'lines', name = 'CPI'))
        fig.add_trace(go.Scatter(x = total_pred_ARIMA.tail(12).index, y = total_pred_ARIMA.tail(12), mode = 'markers', name = 'ARIMA forecast'))
        fig.add_trace(go.Scatter(x = total_pred_GARCH.tail(12).index, y = total_pred_GARCH.tail(12), mode = 'markers', name = 'GARCH forecast'))
        fig.update_layout(title=dict(text = 'Current CPI trend and the forecast for  the next 12 months',
                                    font = dict(size = 22)),autosize=False,
                                    
            width=900,
            height=500,
            )
        st.plotly_chart(fig)
    else:
        #PROPHET
        st.write('We use the Facebook Prophet Package to make some on the fly forcasts')
        df = parse_json(all_json_data)
        df = df[df.index>'01-01-2000']
        df.reset_index(inplace=True)
        df_p = df.copy()
        df_p.rename(columns={'date':'ds','CPI':'y'}, inplace=True)
        df_p['ds'] = df_p['ds'].astype(str)
        m = Prophet()
        m.fit(df_p)
        future = m.make_future_dataframe(periods = 365)
        forecast = m.predict(future)
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        """ ### Consumer Price Index Forecast"""
        fig = plot_plotly(m, forecast,xlabel ='Years', ylabel="Consumer Price Index", figsize = (700, 500))
        st.plotly_chart(fig)
        """ ### Inspecting the components"""
        fig = plot_components_plotly(m, forecast,figsize = (700, 200))
        st.plotly_chart(fig)




    


 