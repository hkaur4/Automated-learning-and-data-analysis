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
from sklearn.metrics import confusion_matrix
from sklearn import base
import pickle

'''
print "Reading the file"
data = pd.read_csv("featuredata.csv")

print "Converting to numpy array"
np_points = data.select_dtypes(exclude=["object"]).as_matrix()

all_points = []

for row in np_points:
	points = []
	point = []
	for item in row:
		point.append(item)
		if len(point) == 2:
			points.append(point)
			point = []
	all_points.append(points)

all_dists = []

print "Calculating distances"
for row in all_points:
	name = data["name"][all_points.index(row)]
	dists = {"name": name}
	for ai,bi in combinations(range(0,len(row)), 2):
		x = row[ai]
		y = row[bi]
		if ai > bi:
			ai,bi = bi,ai
		header = str(ai) + ":" + str(bi)
		dists[header] = distance.euclidean(np.array(x), np.array(y))
	all_dists.append(dists)


print "Saving as csv"
all_dists = pd.DataFrame(all_dists)
all_dists.to_csv("distance_data.csv")
print all_dists

print "Load the distance data"
all_dists = pd.read_csv("distance_data.csv")

comp_dict = {}

print "Comparing distances"
for i, item in all_dists.iterrows():
	split_name = item["name"].split("_")
	name = split_name[0] + "_" + split_name[1]
	_id = int(split_name[-1].split(".")[0])
	if name in comp_dict.keys():
		comp_dict[name][_id] = item
	else:
		comp_dict[name] = {_id: item}

res_dict = []
for x in comp_dict.keys():
	res_dict_item = {"name": x}
	vals = comp_dict[x].keys()
	vals.remove(1)
	item_a = comp_dict[x][1]
	item_b = comp_dict[x][vals[0]]
	res_dict_item["list"] = []
	s_keys = item_a.keys()
	s_keys = sorted(s_keys)
	for key in s_keys:
		if isinstance(item_a[key], float):
			denom = item_a[key]
			if denom == 0:
				denom = 1.0
			res_dict_item["list"].append(100*(item_b[key]-item_a[key])/denom)
	res_dict.append(res_dict_item)

print "Saving as csv"
res_dict = pd.DataFrame(res_dict)
res_dict.to_csv("compare_data.csv")
'''


emotions = {}
for root, dirs, files in os.walk("Emotion/"):
    path = root.split('/')
    for _file in files:
        text = open(("/").join(path) + "/" + _file, "r").read()
        name = _file.split("_")[0] + "_" + _file.split("_")[1]
        emotions[name] = int(text.split(".")[0])


print "Load the comparison data"
res_dict = pd.read_csv("compare_data.csv")

#np_points = res_dict.select_dtypes(exclude=["object", "int64"])
np_points = res_dict["list"]
np_points = [eval(x) for x in np_points]

ll = [len(x) for x in np_points]
print list(set(ll))
np_points = np.array(np_points)

#np_points = np_points.as_matrix()
print np_points.shape
np_points = np.nan_to_num(np_points)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(np_points)


#FROM THE SCREE PLOT, BEST n_components is 5
pca = PCA(n_components=5)
pca.fit(x_scaled)
print x_scaled.shape
pickle.dump(pca, open("pca.bin", "wb"))
pca_features = pca.transform(x_scaled)
print pca_features.shape


X = []
Y = []

#colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', '0.5']
labels = {1: "anger", 2: "neutral", 3: "disgust", 4: "fear", 5: "happiness", 6: "sadness", 7:"surprise"}
for i in range(1, 8):
#	x_points = []
#	y_points = []
	for n in res_dict["name"]:
		if n in emotions.keys() and emotions[n] == i:
			ind = np.where(res_dict["name"]==n)[0][0]
#			x_points.append(pca_features[ind,0])
#			y_points.append(pca_features[ind,1])
			X.append(pca_features[ind])
			Y.append(i)
#	plt.scatter(x_points, y_points, c=colors[i], label=labels[i], s=50)


#plt.legend(fontsize=8)

#plt.show()


print "KERNEL: linear"
linear_classifier = svm.SVC(kernel="linear", probability=True)
linear_classifier.fit(X, Y) 

scores = cross_validation.cross_val_score(linear_classifier, X, Y, cv=5)

print "SCORES:"
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

linear_preds = linear_classifier.predict(X)
print confusion_matrix(Y, linear_preds)
print
print


#FROM POLYNOMIAL PARAMS, BEST degree=2 and coef0=0.5
print "KERNEL: polynomial"
poly_classifier = svm.SVC(kernel="poly", degree=2, coef0=0.5, probability=True)
poly_classifier.fit(X, Y) 

scores = cross_validation.cross_val_score(poly_classifier, X, Y, cv=5)

print "SCORES:"
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

poly_preds = poly_classifier.predict(X)
print confusion_matrix(Y, poly_preds)
print
print


#FROM RBF PARAMS, BEST gamma = 0.08
print "KERNEL: rbf"
rbf_classifier = svm.SVC(kernel="rbf", gamma=0.08, probability=True)
rbf_classifier.fit(X, Y) 

scores = cross_validation.cross_val_score(rbf_classifier, X, Y, cv=5)

print "SCORES:"
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

rbf_preds = rbf_classifier.predict(X)
cm = confusion_matrix(Y, rbf_preds)
print cm
print
print


#FROM SIGMOID PARAMS, BEST gamma=0.01 and coeff=0.0
print "KERNEL: sigmoid"
sigmoid_classifier = svm.SVC(kernel="sigmoid", gamma=0.01, coef0=0.0, probability=True)
sigmoid_classifier.fit(X, Y) 

scores = cross_validation.cross_val_score(sigmoid_classifier, X, Y, cv=5)

print "SCORES:"
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

sigmoid_preds = sigmoid_classifier.predict(X)
print confusion_matrix(Y, sigmoid_preds)
print
print

'''linear_preds = linear_classifier.predict_proba(X)
poly_preds = poly_classifier.predict_proba(X)
rbf_preds = rbf_classifier.predict_proba(X)
sigmoid_preds = sigmoid_classifier.predict_proba(X)

all_preds = zip(linear_preds, poly_preds, rbf_preds, sigmoid_preds)
mix_preds = []
for a,b,c,d in all_preds:
	prob_list = [sum(x) for x in zip(a,b,c,d)]
	res = prob_list.index(max(prob_list)) + 1
	mix_preds.append(res)
print "FINAL"
print confusion_matrix(Y, mix_preds)
'''

'''
class EnsembleClassifier(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict_proba(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X))
        return np.mean(self.predictions_, axis=0)

    def predict(self, x):
    	preds = self.predict_proba(x)
    	return [ x.tolist().index(max(x))+1 for x in preds]

ens = EnsembleClassifier([linear_classifier, poly_classifier, rbf_classifier, sigmoid_classifier])
scores = cross_validation.cross_val_score(ens, X, Y, cv=5)
print "ENSEMBLE"
print "SCORES:"
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

ens_preds = ens.predict(X)
print confusion_matrix(Y, ens_preds)
print
print
'''



#SINCE RBF IS THE BEST WE PLOT THE CONFUSION MATRIX OF THE MODEL


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels.keys()))
    plt.xticks(tick_marks, labels.values(), rotation=45)
    plt.yticks(tick_marks, labels.values())
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


#STORE THE RBF MODEL
pickle.dump(rbf_classifier, open("rbf_model.bin", "wb"))
