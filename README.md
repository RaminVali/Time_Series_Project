# Time_Series_Project
## Examining and Predicting the Consumer Price Index (CPI)

In this project we scrape, explore and use the [CPI](https://www.investopedia.com/terms/c/consumerpriceindex.asp) data acquired from the United States [Beuraue of Labor Statistics](https://www.bls.gov/cpi/) to predict the future CPI. We acquire the data using the API, in json format, convert that to a pandas dataframe and perform analysis on it. After the EDA, we de-trend the data and use ARIMA, GARCH and Facebook Prophet.

## To run the webapp
Simply click [here](https://cpi-prediction.herokuapp.com/). The app automatically refreshes data when the new monthly CPI rate becomes available. You can choose whether to perform ARIMA-GARCH prediction or Facebook Prophet.

## To modify the code and make you own webapp
Clone the repository, you need the setup.sh, requirements.txt (note the version numbers), and the Procfile. 
The notebook contains extra information/plots regarding ARIMA and GARCH training/testing as well as stationarity checks.
We checked stationarity and used first order differencing and subtraction of the rolling mean to de-trend the data.
The downloaded jason file is store in the \data directory. 
Once your own app is ready you can push the repo to heroku directly, or to your own github and connect your repo to heroku. Then you can build the app and run it. The notebook is for the EDA and testing and you need app.py for the on-line deployment.

![Deployment](Heroku.png)

## Sources:
https://github.com/liannewriting/YouTube-videos-public/blob/main/arima-model-time-series-prediction-python/time-series-arima.ipynb

https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff

https://stats.stackexchange.com/questions/213551/how-is-the-augmented-dickey-fuller-test-adf-table-of-critical-values-calculate

https://stats.stackexchange.com/questions/581467/testing-by-using-kpss

https://github.com/pog87/PtitPrince



