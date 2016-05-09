#Katie Swanson
#CMSC 471 Proj 3
#This file allows you to load one image against my pickled trained program.
#Format is 'filename.jpg' and you can only do one at a time per run (on command line)

import sys
import pickle
from PIL import Image

def readInput():
    fileName = sys.argv[1]
    im = Image.open(fileName)
    gd = im.getdata()
    pixels = []
    for pixel in gd:
        pixels.append(pixel[0])
        pixels.append(pixel[1])
        pixels.append(pixel[2])

    return [pixels]

def main():
    pixels = readInput()
    dataSet = pickle.load(open('proj3.p', 'rb'))
    dataPredict = dataSet.predict(pixels)
    print dataPredict[0]

main()
