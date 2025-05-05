import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask.array import argmin
from param import Series
from shapely.ops import triangulate
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
from ARIMA import arima_select

#This code displays a map of the US with our weather stations
#Shown with a white line grid showing which area the station covers

"""
df = pd.read_csv("MarchAprilWeatherData(in).csv")

# check the number of errors labeled as -9999 in each column
for column in df.columns:
    countErrors = df[column].value_counts().get(-9999)
    print(f"{column} number of errors: {countErrors}")
df.mask(df == -9999, inplace=True)
df.dropna(inplace=True)
df = df.drop(df.columns[[0,1]], axis=1)
df.to_csv("Final.csv")
"""

#This is the function that stores the (X,Y) coordinates when clicking on the map
def click_on_map(event):
    if event.inaxes:
        global x
        x = event.xdata
        global y
        y = event.ydata
        fig.canvas.mpl_disconnect(com)
        plt.close()


#this function takes in longitude and latitude and finds the closest 3 weather stations
def triagnulate(lat, long, InputStations):

    dist = pd.DataFrame(columns=['Name', 'Distance', 'Weight'])
    threeClosest = pd.DataFrame(columns=['Name', 'Distance', 'Weight'])

    #calculate the distance from each station to the input locations and store it into an array
    for i in range(0, len(InputStations)):
        distance = np.sqrt(np.abs(np.abs(long - InputStations.LONGITUDE.iloc[i]) ** 2
                           + (np.abs(lat - InputStations.LATITUDE.iloc[i]) ** 2)))
        dist.loc[len(dist)] = [InputStations.NAME.iloc[i], distance, 0]

    #sort by distance and take the closest 3 stations data
    dist = dist.sort_values(by='Distance', ascending=True)
    threeClosest = dist[0:3]

    #calculate the weights for each distance
    for i in range(0,3):
        threeClosest.Weight.iloc[i] = threeClosest.Distance.iloc[i]/np.sum(threeClosest.Distance)

    prop1 = 1 / threeClosest['Weight'].iloc[0]
    prop2 = 1 / threeClosest['Weight'].iloc[1]
    prop3 = 1 / threeClosest['Weight'].iloc[2]

    w1 = prop1 / (prop1 + prop2 + prop3)
    w2 = prop2 / (prop1 + prop2 + prop3)
    w3 = prop3 / (prop1 + prop2 + prop3)

    threeClosest.at[0, 'Weight'] = w1
    threeClosest.at[1, 'Weight'] = w2
    threeClosest.at[2, 'Weight'] = w3

    return threeClosest

"""
This function takes in a data frame of every weather station, and a data frame of the 3 closest stations
along with their weights. It then filters down to only having data of the stations used in triangulation.
After that it applies the corresponding weights to every numeric feature and returns a DataFrame of the 
calculated weighted average for each feature for every unique hour in the data set.
"""
def getLocationData(df, threeClosest):

    # gets the 3 closest stations weather data, and resets indexing
    df = df.where((df['NAME'] == threeClosest.Name.iloc[0]) |
                            (df['NAME'] == threeClosest.Name.iloc[1]) |
                            (df['NAME'] == threeClosest.Name.iloc[2]))
    df.dropna(inplace=True)

    ##apply the weights to each sections data and storing it as a dataframe of only the data portion

    # want to slit this into label info and data info data frames in order to do mass calculations
    testStationLabels = df.iloc[:, 1:6]

    # first third (takes the name at index 1)
    loc1Name = df.NAME.iloc[1]
    weightedDataL1 = (df.iloc[:int(len(df) / 3), 6:] *
                      threeClosest.Weight.loc[threeClosest['Name'] == loc1Name].values[0])

    # second third (calcualtes the name at half way point since range (1/3,2/3) contains 1/2 for ease of calculation
    loc2Name = df.NAME.iloc[int(len(df) / 2)]
    weightedDataL2 = (df.iloc[int(len(df) / 3):int(len(df) * 2 / 3), 6:] *
                     threeClosest.Weight.loc[threeClosest['Name'] == loc2Name].values[0])

    # last third (takes the name at last index)
    loc3Name = df.NAME.iloc[-1]
    weightedDataL3 = (df.iloc[int(len(df) * 2 / 3 + 1):, 6:] *
                     threeClosest.Weight.loc[threeClosest['Name'] == loc3Name].values[0])

    #combine all tree weighted sections back into one DataFrame
    weightedData = pd.concat([weightedDataL1, weightedDataL2, weightedDataL3], axis=0)
    fullWeighted = pd.concat([testStationLabels, weightedData], axis=1)

    # sort the fullWeighted by date in order to aggregate info later
    fullWeighted['DATE'] = pd.to_datetime(fullWeighted['DATE'], format='%m/%d/%Y %H:%M')
    fullWeighted = fullWeighted.sort_values(by='DATE', ascending=True)

    LocationAvg = pd.DataFrame(columns=['NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE',
                                        'HLY-DEWP-NORMAL', 'HLY-HIDX-NORMAL', 'HLY-HTDH-NORMAL',
                                        'HLY-TEMP-10PCTL', 'HLY-TEMP-90PCTL', 'HLY-TEMP-NORMAL',
                                        'HLY-WCHL-NORMAL', 'HLY-WIND-1STDIR', 'HLY-WIND-1STPCT',
                                        'HLY-WIND-AVGSPD', 'HLY-WIND-VCTDIR', 'HLY-WIND-VCTSPD',
                                        'HLY-PRES-10PCTL','HLY-PRES-90PCTL','HLY-PRES-NORMAL'])

    for i in range(0,fullWeighted['DATE'].nunique()-1):
        newRow = pd.DataFrame({'NAME': "Input",
                               'LATITUDE': y,
                               'LONGITUDE': x,
                               'ELEVATION': 0,
                               'DATE': fullWeighted['DATE'].iloc[i * 3],
                               'HLY-DEWP-NORMAL': [np.sum([fullWeighted['HLY-DEWP-NORMAL'].iloc[i * 3],
                                                           fullWeighted['HLY-DEWP-NORMAL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-DEWP-NORMAL'].iloc[i * 3 + 2]])],
                               'HLY-HIDX-NORMAL': [np.sum([fullWeighted['HLY-HIDX-NORMAL'].iloc[i * 3],
                                                           fullWeighted['HLY-HIDX-NORMAL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-HIDX-NORMAL'].iloc[i * 3 + 2]])],
                               'HLY-HTDH-NORMAL': [np.sum([fullWeighted['HLY-HTDH-NORMAL'].iloc[i * 3],
                                                           fullWeighted['HLY-HTDH-NORMAL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-HTDH-NORMAL'].iloc[i * 3 + 2]])],
                               'HLY-TEMP-10PCTL': [np.sum([fullWeighted['HLY-TEMP-10PCTL'].iloc[i * 3],
                                                           fullWeighted['HLY-TEMP-10PCTL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-TEMP-10PCTL'].iloc[i * 3 + 2]])],
                               'HLY-TEMP-90PCTL': [np.sum([fullWeighted['HLY-TEMP-90PCTL'].iloc[i * 3],
                                                           fullWeighted['HLY-TEMP-90PCTL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-TEMP-90PCTL'].iloc[i * 3 + 2]])],
                               'HLY-TEMP-NORMAL': [np.sum([fullWeighted['HLY-TEMP-NORMAL'].iloc[i * 3],
                                                           fullWeighted['HLY-TEMP-NORMAL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-TEMP-NORMAL'].iloc[i * 3 + 2]])],
                               'HLY-WCHL-NORMAL': [np.sum([fullWeighted['HLY-WCHL-NORMAL'].iloc[i * 3],
                                                           fullWeighted['HLY-WCHL-NORMAL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-WCHL-NORMAL'].iloc[i * 3 + 2]])],
                               'HLY-WIND-1STDIR': [np.sum([fullWeighted['HLY-WIND-1STDIR'].iloc[i * 3],
                                                           fullWeighted['HLY-WIND-1STDIR'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-WIND-1STDIR'].iloc[i * 3 + 2]])],
                               'HLY-WIND-1STPCT': [np.sum([fullWeighted['HLY-WIND-1STPCT'].iloc[i * 3],
                                                           fullWeighted['HLY-WIND-1STPCT'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-WIND-1STPCT'].iloc[i * 3 + 2]])],
                               'HLY-WIND-AVGSPD': [np.sum([fullWeighted['HLY-WIND-AVGSPD'].iloc[i * 3],
                                                           fullWeighted['HLY-WIND-AVGSPD'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-WIND-AVGSPD'].iloc[i * 3 + 2]])],
                               'HLY-WIND-VCTDIR': [np.sum([fullWeighted['HLY-WIND-VCTDIR'].iloc[i * 3],
                                                           fullWeighted['HLY-WIND-VCTDIR'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-WIND-VCTDIR'].iloc[i * 3 + 2]])],
                               'HLY-WIND-VCTSPD': [np.sum([fullWeighted['HLY-WIND-VCTSPD'].iloc[i * 3],
                                                           fullWeighted['HLY-WIND-VCTSPD'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-WIND-VCTSPD'].iloc[i * 3 + 2]])],
                               'HLY-PRES-10PCTL': [np.sum([fullWeighted['HLY-PRES-10PCTL'].iloc[i * 3],
                                                           fullWeighted['HLY-PRES-10PCTL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-PRES-10PCTL'].iloc[i * 3 + 2]])],
                               'HLY-PRES-90PCTL': [np.sum([fullWeighted['HLY-PRES-90PCTL'].iloc[i * 3],
                                                           fullWeighted['HLY-PRES-90PCTL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-PRES-90PCTL'].iloc[i * 3 + 2]])],
                               'HLY-PRES-NORMAL': [np.sum([fullWeighted['HLY-PRES-NORMAL'].iloc[i * 3],
                                                           fullWeighted['HLY-PRES-NORMAL'].iloc[i * 3 + 1],
                                                           fullWeighted['HLY-PRES-NORMAL'].iloc[i * 3 + 2]])]
                               })
        LocationAvg = pd.concat([LocationAvg, newRow], axis=0)
    return LocationAvg

#This is how we get the grid of the US
us_states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
final = pd.read_csv("Final.csv")[["LONGITUDE", "LATITUDE"]]
final = final.to_numpy()
final = final[final[:, 0] > -105]
final = final[final[:, 1] > 39]

#Get the latitude, longitude and station location name
stationsLatLon = pd.read_csv("Final.csv")[["NAME","LATITUDE", "LONGITUDE"]]
stationsLatLon = stationsLatLon.drop_duplicates()
stationsFull = pd.read_csv("Final.csv")


#Using Kmeans to get our stations
kmeans = KMeans(41)
kmeans.fit(final)
centers = kmeans.cluster_centers_


plt.style.use('dark_background')

fig, ax = plt.subplots()
us_states.plot(ax=ax, color='none', edgecolor='cyan')

#This is our plotting function, which plots our grid based on latitude and longitude
vor = Voronoi(centers)
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors="LightGrey", line_width=2.0, ax=ax)
ax.set_xlim([-105, -80])
ax.set_ylim([38, 50])
plt.scatter(final[:, 0], final[:, 1], c=kmeans.labels_, s=4.0)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="Magenta", s=4.0)

#creates a canvas event when the map is clicked
com = fig.canvas.mpl_connect('button_press_event', click_on_map)
plt.show()


stationsWithWeights = triagnulate(y, x, stationsLatLon)
customLocData = getLocationData(stationsFull, stationsWithWeights)

#customLocData.to_csv("CustomLocation.csv")
#arima_select('HLY-TEMP-NORMAL', "CustomLocation.csv")
