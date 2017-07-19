import glob
import os
import fnmatch
import re

rootDir="/data/oir/"
labeledFile="/home/saxiao/oir/data/labeled.txt"
rawFile="/home/saxiao/oir/data/raw.txt"

keys=["quantified", "quant", "QUANT", "QUANTIFIED", "_Q", "yellow", "Yellow"]

rawMatches = []
labeledMatches = []
for key in keys:
	for root, dirnames, filenames in os.walk(rootDir):
		for filename in fnmatch.filter(filenames, '*' + key + '*.tif'):
			regexpSearch = re.sub("(\s*"+key+"\s*)", "\\s*", filename)
			for f in os.listdir(root):
				if re.search(regexpSearch, f):
					rawMatches.append(os.path.join(root, f))
					labeledMatches.append(os.path.join(root, filename))

f = open(labeledFile, "w")
f.write("\n".join(labeledMatches))
f.close()

f = open(rawFile, "w")
f.write("\n".join(rawMatches))
f.close()
