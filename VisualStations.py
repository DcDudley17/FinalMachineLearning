import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd

#This code displays a map of the US with our weather stations
#Shown with a white line grid showing which area the station covers

#This is how we get the grid of the US
us_states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")

final = pd.read_csv("MarchAprilWeatherData.csv")[["LONGITUDE", "LATITUDE"]]
#final = final.CleanFile()
final = final.to_numpy()
final = final[final[:, 0] > -105]
final = final[final[:, 1] > 39]

#Using Kmeans to get our stations
kmeans = KMeans(41)
kmeans.fit(final)
centers = kmeans.cluster_centers_


plt.style.use('dark_background')

fig, ax = plt.subplots()
us_states.plot(ax=ax, color='none', edgecolor='white')

#This is our plotting function, which plots our grid based on latitude and longitude
vor = Voronoi(centers)
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors="white", line_width=2.0, ax=ax)
ax.set_xlim([-105, -80])
ax.set_ylim([38, 50])

#uncomment this if you want to see our stations in color
plt.scatter(final[:, 0], final[:, 1], c=kmeans.labels_, s=4.0)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", s=4.0)


plt.show()