import os
import shutil 
rootDir =  '/home/shivani/Pictures/cohn-kanade-images/'
copyDir = '/home/shivani/Pictures/cohn-kanade-images-extracted/'
for subdir, dirs, files in os.walk(rootDir):
	files = [x for x in files if x and x[0]!= "."]
	fileLen = len(files)
	if (fileLen != 0):
		files.sort()
		srcpath =  os.path.join(subdir, files[0])
		destpath = os.path.join(copyDir, files[0])
		#print srcpath
		shutil.copyfile(srcpath,destpath)
		srcpath =  os.path.join(subdir, files[-1])
		destpath = os.path.join(copyDir, files[-1])
		shutil.copyfile(srcpath,destpath)
