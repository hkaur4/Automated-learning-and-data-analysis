import os.path
import cv2
import stasm
import math
import numpy as np 

path = os.path.join(stasm.DATADIR, '/home/shivani/Pictures/cohn-kanade-images-extracted/S052_001_00000001.png')
path2 = os.path.join(stasm.DATADIR, '/home/shivani/Pictures/cohn-kanade-images-extracted/S052_001_00000015.png')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
if img is None:
	print("Cannot load", path)
	raise SystemExit

stasm.init()
stasm.open_image(img)
landmarks = stasm.search_auto()

stasm.open_image(img2)
landmarks2 = stasm.search_auto()

canvas = np.zeros((490,640,3), np.uint8)

#ht, wd = cv2.GetSize(img)
print img.shape
#ht, wd = cv2.GetSize(img2)
print img2.shape

print canvas.shape

def drawArrowLine(image, p, q, color, thickness, line_type, shift):
	arrow_magnitude = 0.3* math.sqrt(((p[0]-q[0])*(p[0]-q[0]))+((p[1]-q[1])*(p[1]-q[1])))
	cv2.line(image, p, q, color, thickness, line_type, shift)
	angle = np.arctan2(p[1]-q[1], p[0]-q[0])
	p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
	cv2.line(image, p, q, color, thickness, line_type, shift)
	p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
	cv2.line(image, p, q, color, thickness, line_type, shift)

if len(landmarks) == 0:
	print("No face found in", path)
else:
	landmarks = stasm.force_points_into_image(landmarks, img)
	landmarks2 = stasm.force_points_into_image(landmarks2, img2)

	count = 1
	for origin, dest in zip(landmarks,landmarks2):
		drawArrowLine(canvas,(int(origin[0]),int(origin[1])),(int(dest[0]),int(dest[1])),(57,255,20),1,8,0)

		#cv2.putText(canvas,str(count),(int(origin[0]),int(origin[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255,0,0))
		#cv2.putText(canvas,str(count),(int(dest[0]),int(dest[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(57,255,20))
		count = count+1
cv2.imwrite('displacement.png',canvas)
cv2.imshow("stasm minimal", canvas)
cv2.waitKey(0)