import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import root_mean_squared_error
from scipy.stats import boxcox
from scipy.special import inv_boxcox


def arima_select(feature, trainFile, testFile):
    weatherDat = pd.read_csv(trainFile)
    featureData = weatherDat.loc[:, ['NAME', 'DATE', feature]].dropna()
    featureData['DATE'] = pd.to_datetime(featureData['DATE'])
    #plt.plot(featureData['DATE'], featureData[feature], c="blue")
    #plt.show()


    #plot_acf(featureData[feature], lags=500)
    #plot_pacf(featureData[feature], lags=75)
    #plt.show()
    #significant out to 28 lags according to pacf

    """
    #Uncomment to check differencing required to make data stationary
    
    #based on this plot we see that the data seems to have a trend which needs to de removed
    result = adfuller(featureData[feature])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
  
    result = adfuller(EauClaireData[feature])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    """

    testData = pd.read_csv(testFile)
    testData['DATE'] = pd.to_datetime(testData['DATE'])
    trainData = featureData.loc[1:, ['DATE', feature]]
    testData = testData.loc[1:, ['DATE', feature]]

    # remaining work is to reign in parameter tuning
    model = ARIMA(trainData[feature], order=(24,1,12)).fit()
    forecast_base = model.forecast(len(testData))

    modelSeasonal = ARIMA(trainData[feature], order=(15,1,12), seasonal_order = (1,1,2,24)).fit()
    forecast_seasonal = modelSeasonal.forecast(len(testData))

    #check residual sum of squares
    score_base =  root_mean_squared_error(testData[feature], forecast_base)
    score_seasonal = root_mean_squared_error(testData[feature], forecast_seasonal)

    return forecast_base, score_base, forecast_seasonal, score_seasonal
