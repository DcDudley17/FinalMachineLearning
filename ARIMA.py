import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
from scipy.special import inv_boxcox


def arima_select(feature, filePath):
    weatherDat = pd.read_csv(filePath)

    featureData = weatherDat.loc[:, ['NAME', 'DATE', feature]].dropna()
    featureData['DATE'] = pd.to_datetime(featureData['DATE'])
    #EauClaireData = tempData[tempData['NAME'] == "EAU CLAIRE RGNL AP, WI US"]
    plt.plot(featureData['DATE'], featureData[feature], c="blue")
    plt.show()


    #based on this plot we see that the data seems to have a trend which needs to de removed
    result = adfuller(featureData[feature])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])


    plot_acf(featureData[feature], lags=48)
    plot_pacf(featureData[feature], lags=48)
    plt.show()
    #significant out to 28 lags according to pacf

    """
    #Uncomment to check differencing required to make data stationary
    EauClaireDataD1 = EauClaireData
    EauClaireDataD1[feature] = EauClaireData[feature].diff().dropna()
    plt.plot(EauClaireDataD1['DATE'], EauClaireDataD1[feature])
    plt.show()
    
    result = adfuller(EauClaireData[feature])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    """


    #using boxcox to normalize varience
    #EauClaireData[feature], lam = boxcox(EauClaireData[feature])

    # need to split train and test data
    trainData = featureData.iloc[:int(len(featureData)*0.8)]
    testData = featureData.iloc[int(len(featureData)*0.8):]


    # remaining work is to rein in parameter tuning
    model = ARIMA(trainData[feature], order=(28,1,12)).fit()
    boxcox_forecast = model.forecast(len(testData))
    #undoing the boxcox transformation messes things up
    #forecasts = inv_boxcox(boxcox_forecast, lam)


    plt.plot(trainData['DATE'], trainData[feature], c="blue")
    plt.plot(testData['DATE'], testData[feature], c="red")
    plt.plot(testData['DATE'], boxcox_forecast, c="green")
    plt.show()

    #check residual sum of squares
    score = np.sum((boxcox_forecast - testData[feature])**2)
    print(score)


