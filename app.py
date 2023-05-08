
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

st.title('Consumer Price Index Forecast App')
"""
Consumer Price Index for All Urban Consumers (CPI-U) is a common indicator of inflation and rose 0.1 percent in March on a seasonally
adjusted basis, after increasing 0.4 percent in February, the U.S. Bureau of Labor Statistics [reported] (https://www.bls.gov/news.release/cpi.nr0.htm)
today. 

In this project we access, parse, visualize and analyze the data from the U.S. Bureau of Labor Statistics, and use it to predict the inflation rate uing 
[ARIMA](https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html) and [Prophet](https://facebook.github.io/prophet/).

### Loading the data
"""

TODAY = datetime.today().strftime("%Y-%m-%d")
STARTYEAR = 1980
st.write(f'Today is {TODAY}')
# Finding the current month to see if we have uptodate data.
currentDay = datetime.now().day
currentMonth = datetime.now().month
currentYear = datetime.now().year

def get_json_data(currentYear):
    """
    Function ussing the API to get the jason data up to the current uear when called. Returns a json file
    """
    headers = {'Content-type': 'application/json'}
    
    all_json_data = []
    step = 20 #in years
    for year in range (STARTYEAR,currentYear+1,step):
        parameters = json.dumps({"registrationkey":"0c59b5a5919e423f961de3321be9f7ef", "seriesid":['CUSR0000SA0'], "startyear":str(year), "endyear":str(year+step), "calculations":"true"})
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

"""### Visualising and Exploring the Data"""
def plotly_plot (df, x , y, title, width, height): # Plotly plotting function for time series
    """Function to create plotly plots for me"""
    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=list(x), y=list(y)))

    # Set title
    fig.update_layout(
        title_text=title,
        width=width,
        height=height
    )
    # Add range slider
    fig.update_layout(
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
        )
    )
    return fig

fig =plotly_plot(df,df.index, df.CPI, 'Raw Consumer Price Index', 700, 300)
st.plotly_chart(fig)

# Plot percentage
df['pct-change'] =df.CPI.pct_change(periods=12)*100
fig = plotly_plot(df,df.index, df['pct-change'], 'Consumer Price Index Percentage Chnage', 700, 300)
st.plotly_chart(fig)

## YoY Percentage Change 
df['date'] = df.index
df['month'] = df['date'].apply(lambda x: x.month)
fig = px.box(df[12:], x="month", y="pct-change", points = "all" )
fig.update_layout(
    xaxis = dict(tickmode = 'linear'),
        title_text="Year on Year CPI Percentage Change: Monthly Box-Plots")
st.plotly_chart(fig)