import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#This is trying to determine what parameters are best for classifying our weather station


df = pd.read_csv("MarchAprilWeatherData.csv")
# Convert integers to floats
df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')



#Dropping columns that we don't want
df = df.drop(columns=['STATION'])
df = df.drop(columns=['DATE'])
df = df.drop(columns=['Unnamed: 0'])

#These make it 100% accurate, so want to see without
df = df.drop(columns=['LATITUDE'])
df = df.drop(columns=['LONGITUDE'])

#Can be used as it is a good classifier
df = df.drop(columns=['ELEVATION'])

#This just shows the first two words of the station Name
df['NAME'] = df['NAME'].str.split().str[:2].str.join(' ')

# Select variables
y = df['NAME'].to_numpy() #Target
X = df.drop(columns=['NAME']).to_numpy()  # predictors

X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier()

# parameters = {"max_depth": range(2,13)}
# grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
# grid_search.fit(X_train, y_train)
# score_df = pd.DataFrame(grid_search.cv_results_)
#print(score_df[['param_max_depth','mean_test_score','rank_test_score']])

#max_depth = grid_search.best_params_["max_depth"]
#print(max_depth)

max_depth = 8

clf = RandomForestClassifier(max_depth=max_depth, oob_score=True, verbose=3)
clf.fit(X_train, y_train)


importances = pd.DataFrame(clf.feature_importances_, index=df.columns[1:])
importances.plot.bar()
plt.show()

print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
