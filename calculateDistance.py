import os
import numpy as np
import math
from itertools import combinations
from scipy.spatial import distance
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
	for x,y in combinations(row, 2):
		header = str(row.index(x)) + ":" + str(row.index(y))
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
	for key in item_a.keys():
		if isinstance(item_a[key], float):
			if item_a[key] != 0:
				res_dict_item[key] = 100*(item_b[key]-item_a[key])/item_a[key]
	res_dict.append(res_dict_item)

print "Saving as csv"
res_dict = pd.DataFrame(res_dict)
res_dict.to_csv("compare_data.csv")
