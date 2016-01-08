import os
import numpy as np
import math
from itertools import combinations
from scipy.spatial import distance
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import cross_validation
import matplotlib.patches as mpatches


emotions = {}
for root, dirs, files in os.walk("Emotion/"):
    path = root.split('/')
    for _file in files:
        text = open(("/").join(path) + "/" + _file, "r").read()
        name = _file.split("_")[0] + "_" + _file.split("_")[1]
        emotions[name] = int(text.split(".")[0])


print "Load the comparison data"
res_dict = pd.read_csv("compare_data.csv")

np_points = res_dict.select_dtypes(exclude=["object", "int64"])

np_points = np_points.as_matrix()
np_points = np.nan_to_num(np_points)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(np_points)



pca = PCA(n_components=5)
pca.fit(x_scaled)
pca_features = pca.transform(x_scaled)


X = []
Y = []

for i in range(1, 8):
	for n in res_dict["name"]:
		if n in emotions.keys() and emotions[n] == i:
			ind = np.where(res_dict["name"]==n)[0][0]
			X.append(pca_features[ind])
			Y.append(i)



gamma = [x/100.0 for x in range(0, 21)]
Xp = []
yy = []
max_acc = 0.0
max_val = 0.0
for x in gamma:
	classifier = svm.SVC(kernel="rbf", gamma=x)
	classifier.fit(X, Y) 

	scores = cross_validation.cross_val_score(classifier, X, Y, cv=5)

	Xp.append(scores.mean())
	yy.append(x)
	if scores.mean() > max_acc:
		max_acc = scores.mean()
		max_val = x
plt.plot(yy, Xp)
plt.xlabel('gamma')
plt.ylabel('accuracy')
plt.show()

print "We get " + str(max_acc) + " accuracy when gamma=" + str(max_val)
