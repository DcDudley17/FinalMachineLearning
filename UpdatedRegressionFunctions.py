import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

"""
The bellow function takes in both our weather data from previous days and our actual data we are predicting.
It then filters so that we only have our numerical columns and then runs randomForrest, linear, and SVR with rbf.
it then takes the last 73 hours from the marchApril set and then compares to the next 73 hours of data. 
We then print a graph in the end for our wind prediction
"""
def standardRegressions(feature, trainFile, testFile):
    # Load dataset
    df = pd.read_csv(trainFile)
    actual = pd.read_csv(testFile)
    df = df.iloc[1:]
    actual = actual.iloc[1:]

    # Filter only numeric columns (exclude unneeded data that doesn't have effect on weather)
    df = df.select_dtypes(include=['number']).drop(columns=['ELEVATION', 'LATITUDE', 'LONGITUDE'])
    actual = actual.select_dtypes(include=['number']).drop(columns=['ELEVATION', 'LATITUDE', 'LONGITUDE'])
    actual = actual.drop(actual.columns[0], axis=1)
    numHours = actual[feature].shape[0]
    ts = np.linspace(0,168,numHours)
    LinearView = pd.DataFrame(ts, columns=['hours'])
    SVRView = pd.DataFrame(ts, columns=['hours'])
    randomForView = pd.DataFrame(ts, columns=['hours'])

    randForest = RandomForestRegressor(n_estimators=100)
    linear = LinearRegression()
    modelSVR = SVR(kernel='rbf',C=100, epsilon=0.01, gamma=0.01)


    # target = df.columns.tolist()

    targetlist = np.array(['HLY-WCHL-NORMAL','HLY-WIND-AVGSPD','HLY-TEMP-NORMAL'])

    dfsave = df.copy()
    for i in range(0,len(targetlist)):
        # Choose target column to predict
        df = dfsave.copy()

        target = targetlist[i]

        # Shift target to predict next time step
        df['target_future'] = df[target].shift(-1)

        # Drop last row with NaN target
        df.dropna(inplace=True)

        # Features and target
        feature_cols = df.columns.difference(['target_future'])  # Keep all columns except the shifted one
        X = df[feature_cols]
        y = df['target_future']
        X = (X - np.average(X, axis=0)) / np.std(X, axis=0)
        X.drop(X.columns[-1], axis=1, inplace = True)
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        #Train using our three different models
        randForest.fit(X_train, y_train)
        linear.fit(X_train, y_train)
        modelSVR.fit(X_train, y_train)

        #Get values we need
        future_input = X_test.tail(numHours) # The tail gathers the last x hours worth of data

        #Our prediction with each of our 3 different prediction functions
        randomPredict = randForest.predict(future_input)
        linearPredict = linear.predict(future_input)
        SVRPredict = modelSVR.predict(future_input)

        # Get the starting index of the test split relative to the full dataset
        """note: realData is not in fact the real data, this just shifts the 'realData' down to match predictions"""
        start_index = len(X) - len(X_test) - 5
        realData = df[target].iloc[start_index: start_index + numHours].reset_index(drop=True)


        linearData = pd.DataFrame(linearPredict, columns=[target])
        SVRData = pd.DataFrame(SVRPredict, columns=[target])
        randomForData = pd.DataFrame(randomPredict, columns=[target])
        LinearView = pd.concat([LinearView, linearData], axis=1)
        randomForView = pd.concat([randomForView,randomForData], axis=1)
        SVRView = pd.concat([SVRView, SVRData], axis=1)
    LinearView.to_csv("LinearReg.csv", index=False)
    randomForView.to_csv("RandomForrestReg.csv", index=False)
    SVRView.to_csv("SVRReg.csv", index=False)