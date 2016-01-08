import os


vals = []
check = {}

for root, dirs, files in os.walk("Emotion/"):
    path = root.split('/')
    for _file in files:
        text = open(("/").join(path) + "/" + _file, "r").read()
        name = _file.split("_")[0] + "_" + _file.split("_")[1]
        vals.append({"name": name, "val": int(text.split(".")[0])})
        check[name] = int(text.split(".")[0])
        if check[name] == 2:
        	print name

print list(set(check.values()))
print len(vals)
print len(check.keys())