# Merge roi between input and output frame

import cv2

import os

from os import listdir
from os.path import isfile, join
from os import walk

mypath = 'data/vids/covid2'
outputpath = 'data/output/covid2'
x = 291
y = 234
h = 355 - y
w = 544 - x
roi = [x, y, h, w] # x, y, w, h
mergepath = 'data/merge/covid2'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#filelist =[]
#for (dirpath, dirnames, filenames) in walk(mypath):
#    filelist.extend(filenames)
#    break

for f in onlyfiles:
    print(f)
    m1 = cv2.imread(join(mypath, f))
    m2 = cv2.imread(join(outputpath, f))
    #m1[x:x+w, y:y+h, :] = m2[x:x+w, y:y+h, :]
    m1[y:y+h, x:x+w, :] = m2[y:y+h, x:x+w,  :]
    cv2.rectangle(m1, (x,y), (x+w,y+h), (0,0,255), 1)
    cv2.imwrite(join(mergepath, f), m1)
