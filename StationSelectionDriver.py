import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
from ARIMA import arima_select
from CustomLocCreator import getLocationData, triagnulate

#This code displays a map of the US with our weather stations
#Shown with a white line grid showing which area the station covers

"""
#cleaning method used on raw data
this was used to clean the data
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

#This is how we get the grid of the US
us_states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")
final = pd.read_csv("Final.csv")[["LONGITUDE", "LATITUDE"]]
final = final.to_numpy()
final = final[final[:, 0] > -105]
final = final[final[:, 1] > 39]

#Get the latitude, longitude and station location name
stationsLatLon = pd.read_csv("Final.csv")[["NAME","LATITUDE", "LONGITUDE"]]
stationsLatLon = stationsLatLon.drop_duplicates()
stationsFullTrain = pd.read_csv("Final.csv")
stationsFullTrain['DATE'] = pd.to_datetime(stationsFullTrain['DATE'], format='%m/%d/%Y %H:%M')
stationsFullTest = pd.read_csv("ActualAprilWeather(in).csv")
stationsFullTest = stationsFullTest.drop(stationsFullTest.columns[[0,1]], axis=1)
stationsFullTest['DATE'] = pd.to_datetime(stationsFullTest['DATE'], format='%m/%d/%Y %H:%M')

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
customLocTrainData = getLocationData(stationsFullTrain, stationsWithWeights, x,y)
customLocTestData = getLocationData(stationsFullTest, stationsWithWeights,x,y)
customLocTrainData.to_csv("CustomLocationTrain.csv")
customLocTestData.to_csv("CustomLocationTest.csv")
arima_select('HLY-TEMP-NORMAL', "CustomLocationTrain.csv", "CustomLocationTest.csv")
