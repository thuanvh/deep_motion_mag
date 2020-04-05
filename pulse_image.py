import os
import cv2
from os.path import join, isfile
from os import listdir
import numpy as np


def generatepulse(mypath, outputpath, xrange, yrange, itersave):
    if not os.path.exists(outputpath) :
        os.makedirs(outputpath)

    onlyfiles = [f for f in listdir(mypath) if f.endswith('.png') and isfile(join(mypath, f))]
    onlyfiles.sort()

    #onlyfiles =[]
    #for (dirpath, dirnames, filenames) in walk(mypath):
    #    filelist.extend(filenames)
    #    break
    img0 = cv2.imread(join(mypath, onlyfiles[0]))
    shape0 = img0.shape
    x = xrange
    if x[0] < 0 : x[0] = 0
    if x[1] >= shape0[1] : x[1] = shape0[1]
    
    y = yrange
    if y[0] < 0 : y[0] = 0
    if y[1] >= shape0[0] : y[1] = shape0[0]

    t = len(onlyfiles)
    i = 0
    hor = y[1] - y[0] == 1
    h = x[1] - x[0] if hor else y[1] - y[0]
    m1 = np.zeros((h, t, 3), np.uint8)

    for f in onlyfiles:
        print(f)
        m2 = cv2.imread(join(mypath, f))
        roi = m2[y[0]:y[1], x[0]:x[1],  :]
        if hor : roi = np.rot90(roi)
        m1[0:h, i:i+1, :] = roi
        if itersave:
            cv2.imwrite(join(outputpath, 'phase_iter_' + f), m1)
        i = i+1
    cv2.imwrite(join(outputpath, 'phase_total_{}_{}_{}_{}.png'.format(x[0],x[1],y[0],y[1])), m1)


if __name__ == "__main__":
    mypath = 'data/output/test1'
    outputpath = 'data/phase/test1'
    xrange = [0, 700]
    yrange = [220, 221]
    #xrange = [320, 321]
    #yrange = [0, 800]
    generatepulse(mypath, outputpath, xrange, yrange, False)
    