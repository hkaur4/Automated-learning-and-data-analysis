import os.path
import cv2
import stasm

path = os.path.join(stasm.DATADIR, '/home/shivani/Pictures/cohn-kanade-images-extracted/S132_006_00000023.png')

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
	print("Cannot load", path)
	raise SystemExit

stasm.init()
stasm.open_image(img)

landmarks = stasm.search_auto()

if len(landmarks) == 0:
	print("No face found in", path)
else:
	landmarks = stasm.force_points_into_image(landmarks, img)
	for point in landmarks:
		x=round(point[1])
		y=round(point[0])
		img[x-1][y-1] = 0
		img[x-1][y] = 0
		img[x-1][y+1] = 0
		img[x][y-1] = 0
		img[x][y] = 0
		img[x][y+1] = 0
		img[x+1][y-1] = 0
		img[x+1][y] = 0
		img[x+1][y+1] = 0

cv2.imshow("stasm minimal", img)
cv2.waitKey(0)