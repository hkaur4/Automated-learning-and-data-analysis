import numpy as np
import cv2
from time import sleep
import pickle
import os
import pandas as pd
import stasm
from itertools import combinations
from scipy.spatial import distance
import numpy as np
from sklearn import preprocessing
import math
from sklearn.decomposition import PCA
from sklearn import svm


cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	cv2.imshow('frame',frame)
	pressed_key = cv2.waitKey(1) & 0xFF
	if pressed_key == ord('n'):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray, (640, 490)) 
		cv2.imwrite("test/neutral.png", gray)
		sleep(1)
	if pressed_key == ord('e'):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray, (640, 490)) 
		cv2.imwrite("test/emotion.png", gray)
		sleep(1)
	if pressed_key == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break


print "Loading the model"

emotions = {}
for root, dirs, files in os.walk("Emotion/"):
    path = root.split('/')
    for _file in files:
        text = open(("/").join(path) + "/" + _file, "r").read()
        name = _file.split("_")[0] + "_" + _file.split("_")[1]
        emotions[name] = int(text.split(".")[0])


res_dict = pd.read_csv("compare_data.csv")

np_points = res_dict["list"]
np_points = [eval(x) for x in np_points]

np_points = np.array(np_points)
np_points = np.nan_to_num(np_points)
min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(np_points)
x_scaled = np_points


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


classifier = svm.SVC(kernel="rbf", gamma=0.08)
classifier.fit(X, Y) 


img_neutral = cv2.imread("test/neutral.png", cv2.IMREAD_GRAYSCALE)
img_emotion = cv2.imread("test/emotion.png", cv2.IMREAD_GRAYSCALE)


landmarks = stasm.search_single(img_neutral)
landmarks = stasm.force_points_into_image(landmarks, img_neutral)
neutral_list = []
for point in landmarks:
	neutral_list.append([point[0], point[1]])
neutral_dists = {}
for ai,bi in combinations(range(0,len(neutral_list)), 2):
	x = neutral_list[ai]
	y = neutral_list[bi]
	if ai > bi:
		ai,bi = bi,ai
	header = str(ai) + ":" + str(bi)
	neutral_dists[header] = distance.euclidean(np.array(x), np.array(y))

landmarks = stasm.search_single(img_emotion)
landmarks = stasm.force_points_into_image(landmarks, img_emotion)

emotion_list = []
for point in landmarks:
	emotion_list.append([point[0], point[1]])
emotion_dists = {}
for ai,bi in combinations(range(0,len(emotion_list)), 2):
	x = emotion_list[ai]
	y = emotion_list[bi]
	if ai > bi:
		ai,bi = bi,ai
	header = str(ai) + ":" + str(bi)
	emotion_dists[header] = distance.euclidean(np.array(x), np.array(y))

final_dists = []
s_keys = sorted(neutral_dists.keys())
for key in s_keys:
	final_dists.append(100*(emotion_dists[key]-neutral_dists[key])/neutral_dists[key])

np_points = np.array([final_dists])
np_points = np.nan_to_num(np_points)
min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(np_points)
x_scaled = np_points
#pca = pickle.load(open("pca.bin", "rb"))
pca_features = pca.transform(x_scaled)
labels = {1: "anger", 2: "neutral", 3: "disgust", 4: "fear", 5: "happiness", 6: "sadness", 7:"surprise"}
print "DETECTED EMOTION:"
print labels[classifier.predict(pca_features)[0]]