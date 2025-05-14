import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
from ARIMA import arima_select
from HoltWinters import holt_wint_select
from CustomLocCreator import getLocationData, triagnulate
from UpdatedRegressionFunctions import standardRegressions
import tkinter as tk
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

#This code displays a map of the US with our weather stations
#Shown with a white line grid showing which area the station covers

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
stationsFullTest = pd.read_csv("actualWeather.csv")
stationsFullTest = stationsFullTest.drop(stationsFullTest.columns[[0,1]], axis=1)
stationsFullTest['DATE'] = pd.to_datetime(stationsFullTest['DATE'], format='%m/%d/%Y %H:%M')

#Using Kmeans to get our stations
kmeans = KMeans(41)
kmeans.fit(final)
centers = kmeans.cluster_centers_

# Set up the plot to show the map of wisconsin
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
plt.style.use('default')
#uses selected location to generate the custom train and test set then saves them to files
stationsWithWeights = triagnulate(y, x, stationsLatLon)
customLocTrainData = getLocationData(stationsFullTrain, stationsWithWeights, x,y)
customLocTestData = getLocationData(stationsFullTest, stationsWithWeights,x,y)
customLocTrainData.to_csv("CustomLocationTrain.csv")
customLocTestData.to_csv("CustomLocationTest.csv")

#this is a hard-coded feature selection
#targetlist = ['HLY-WCHL-NORMAL','HLY-WIND-AVGSPD','HLY-TEMP-NORMAL', 'HLY-HIDX-NORMAL']
Feature = 'HLY-TEMP-NORMAL'

#use the models and getting their info
customLocTestData = pd.read_csv("CustomLocationTest.csv")
customLocTestData = customLocTestData[1:]
standardRegressions(Feature,"CustomLocationTrain.csv", "CustomLocationTest.csv")
LinearData = pd.read_csv("LinearReg.csv")
Linear_Forecast = LinearData[Feature]
SVRData = pd.read_csv("SVRReg.csv")
SVR_Forecast = SVRData[Feature]
RandForData = pd.read_csv("RandomForrestReg.csv")
RF_Forecast = RandForData[Feature]
HWES_Forecast, rmse_HWES = holt_wint_select(Feature,"CustomLocationTrain.csv", "CustomLocationTest.csv")
arima_Forecast, rmse_arima, sarima_Forecast, rmse_sarema = arima_select(Feature, "CustomLocationTrain.csv", "CustomLocationTest.csv")
print(f"Holt Winters RMSE: {rmse_HWES} ARIMA RMSE:{rmse_arima}, SARIMA RMSE: {rmse_sarema}")

#Creats a GUI to access info
root = tk.Tk()
root.geometry("1000x600")
root.title("Graph Comparisons")
label = tk.Label(root, text="Select Models", font = ('Arial', 20))
label.pack()

#create a frame to position the elements in the application
frame = (Frame(root))
frame.pack(side="left")
fig = Figure(figsize=(7,5), dpi=100)
graph = fig.add_subplot()
graph.plot(customLocTestData['DATE'], customLocTestData[Feature], c="black")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()


#set up checkboxes for each of the models we used
check_state_Linear = tk.IntVar()
checkLinear = tk.Checkbutton(frame, text="Linear Regression", variable=check_state_Linear)
checkLinear.pack(padx=10, pady=10, anchor="w")

check_state_SVR = tk.IntVar()
checkLinear = tk.Checkbutton(frame, text="Support Vector Regression", variable=check_state_SVR)
checkLinear.pack(padx=10, pady=10, anchor="w")

check_state_RF = tk.IntVar()
checkLinear = tk.Checkbutton(frame, text="Random Forest", variable=check_state_RF)
checkLinear.pack(padx=10, pady=10, anchor="w")

check_state_ARIMA = tk.IntVar()
checkLinear = tk.Checkbutton(frame, text="ARIMA", variable=check_state_ARIMA)
checkLinear.pack(padx=10, pady=10, anchor="w")

check_state_SARIMA = tk.IntVar()
checkLinear = tk.Checkbutton(frame, text="Seasonal ARIMA", variable=check_state_SARIMA)
checkLinear.pack(padx=10, pady=10, anchor="w")

check_state_HWES = tk.IntVar()
checkLinear = tk.Checkbutton(frame, text="Holt Winters Exponential Smoothing", variable=check_state_HWES)
checkLinear.pack(padx=10, pady=10, anchor="w")

def plotGraph():
    graph.clear()
    graph.plot(customLocTestData['DATE'], customLocTestData[Feature], c="black")
    
    if (check_state_Linear.get() == 1):
        graph.plot(customLocTestData['DATE'], Linear_Forecast)
    if (check_state_SVR.get() == 1):
        graph.plot(customLocTestData['DATE'], SVR_Forecast)
    if (check_state_RF.get() == 1):
        graph.plot(customLocTestData['DATE'], RF_Forecast)
    if (check_state_ARIMA.get() == 1):
       graph.plot(customLocTestData['DATE'], arima_Forecast)
    if (check_state_SARIMA.get() == 1):
        graph.plot(customLocTestData['DATE'], sarima_Forecast)
    if (check_state_HWES.get() == 1):
        graph.plot(customLocTestData['DATE'], HWES_Forecast)
    canvas.draw()

#creates a button that will show the graphs of the selected models when pressed
ShowGraphButton = tk.Button(frame, text="Show Graphs", command=plotGraph)
ShowGraphButton.pack(pady=20, side="top")

root.mainloop()
