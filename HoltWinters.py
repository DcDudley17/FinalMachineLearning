import pandas as pd
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def holt_wint_select(feature, trainFile, testFile):
    trainData = pd.read_csv(trainFile)
    testData = pd.read_csv(testFile)

    #note: first observation is missing
    trainData = trainData.loc[1:, ['DATE', feature]]
    testData = testData.loc[1:, ['DATE', feature]]

    #for some reason datetime vars get passed as string, so they need to be reassigned
    trainData['DATE'] = pd.to_datetime(trainData['DATE'], format='%Y-%m-%d %H:%M:%S')
    testData['DATE'] = pd.to_datetime(testData['DATE'], format='%Y-%m-%d %H:%M:%S')

    #based on data, most of our data has additive for seasonality and trend and has a 24 hour daily cycle
    holt_wint_model = ExponentialSmoothing(trainData[feature] ,seasonal_periods=24, seasonal="add", trend="add", dates=trainData['DATE']).fit()
    holt_wint_forecast = holt_wint_model.forecast(testData.DATE.nunique())

    rmse = root_mean_squared_error(testData[feature], holt_wint_forecast)

    return holt_wint_forecast, rmse