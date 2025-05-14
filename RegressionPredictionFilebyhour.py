import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

"""
The bellow function takes in both our weather data from previous days and then does a train test split to do our predicting.
It then filters so that we only have our numerical columns and then runs randomForrest, linear, and SVR with rbf.
it then takes the last 169 hours from the marchApril set and then compares to the next 169 hours of data. 
We then print a graph in the end for our wind prediction
"""

# Load dataset
df = pd.read_csv("MarchAprilWeatherData.csv")
actual = pd.read_csv("actualWeather.csv")


actual.to_csv('actualWeather.csv', index=False) 

#gather just one station
df = df.query('NAME.str.contains("EAU", case=False)')
actual = actual.query('NAME.str.contains("EAU", case=False)')

# Filter only numeric columns and remove elevation and latitude and longitude as these features don't need to be predicted
df = df.select_dtypes(include=['number']).drop(columns=['ELEVATION', 'LATITUDE', 'LONGITUDE'])
actual = actual.select_dtypes(include=['number']).drop(columns=['ELEVATION', 'LATITUDE', 'LONGITUDE'])

#Gather the number of hours we want to predict
numHours = 169
ts = np.linspace(0,168,numHours)
newView = pd.DataFrame(ts, columns=['hours'])

randForest = RandomForestRegressor(n_estimators=100)
linear = LinearRegression()
modelSVR = SVR(kernel='rbf',C=100, epsilon=0.01, gamma=0.01)

#This allows for adding in each feature of which we want to predict
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

    # Split data (80% train, 20% test)
    #Had to set the shuffle to false so we keep the time series in place 
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

    # Get the starting index of the test split relative to the full dataset, so
    # we can compare the proper time range
    start_index = len(X) - len(X_test) - 5
    realData = df[target].iloc[start_index: start_index + numHours].reset_index(drop=True)

    #Need to figure out why it predicts the days from March set not the next days
    print(randomPredict)
    print(linearPredict)
    print(SVRPredict)
    print(realData)

    #This is how we plot each model against the actual data we have. 
    plt.plot(ts,randomPredict,'r',ts,linearPredict,'b',ts,SVRPredict,'m',ts,realData,'k')
    plt.legend(["randomForest","Linear","SVR","Actual"])
    plt.title(f"7-Day Prediction for {target}")
    plt.show()

    test = pd.DataFrame(linearPredict, columns=[target])
    newView = pd.concat([newView,test], axis=1)

    # Compute and print R2 scores
    print(f"R2 score (Random Forest) for {target}: {randForest.score(X_test, y_test):.4f}")
    print(f"R2 score (Linear Regression) for {target}: {linear.score(X_test, y_test):.4f}")
    print(f"R2 score (SVR) for {target}: {modelSVR.score(X_test, y_test):.4f}")
newView.to_csv("Test.csv", index=False)
