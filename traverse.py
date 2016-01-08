import os
import cv2
import stasm
import csv
	
rootDir = '/home/shivani/Pictures/cohn-kanade-images-extracted/'
csvFile = open('featuredata.csv', 'wb')
writer = csv.writer(csvFile)

for subdir, dirs, files in os.walk(rootDir):
	for imageFile in files:
		path =  os.path.join(subdir, imageFile)
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			print("Cannot load", path)
			raise SystemExit
		
		landmarks = stasm.search_single(img)
		
		landmarks = stasm.force_points_into_image(landmarks, img)
		list1 = [imageFile]
		for point in landmarks:
			list1.append(point[0])
			list1.append(point[1])
		writer.writerows([list1])
		#print len(list1)
csvFile.close()