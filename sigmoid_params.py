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


coef0 = [0.0,0.1,0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
gamma = [x/100.0 for x in range(0, 21)]
colors = ["k", "b", "g", "r", "c", "m", "y", "0.5", "0.25", "0.75"]
max_acc = 0.0
max_gamma = 0.0
max_coeff = 0.0
ll = []
for x in coef0:
	Xp = []
	yy = []
	for c in gamma:
		classifier = svm.SVC(kernel="sigmoid", gamma=c, coef0=x)
		classifier.fit(X, Y) 

		scores = cross_validation.cross_val_score(classifier, X, Y, cv=5)

		Xp.append(scores.mean())
		yy.append(c)
		if max_acc < scores.mean():
			max_acc = scores.mean()
			max_coeff = x
			max_gamma = c
	plt.plot(yy, Xp, colors[coef0.index(x)])
	ll.append(mpatches.Patch(color=colors[coef0.index(x)], label=str(x)+" coeff"))
plt.xlabel('gamma')
plt.ylabel('accuracy')
plt.legend(handles=ll,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


print "We get " + str(max_acc) + " accuracy when gamma=" + str(max_gamma) + " and coeff=" + str(max_coeff)
