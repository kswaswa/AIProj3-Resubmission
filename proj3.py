#Katie Swanson
#CMSC 471 Proj 3

import cv2
import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
import pickle
from PIL import Image
import os


usrImg = 0

labels = []
descriptors = []

validationSet = []

smileys = []
hats = []
hashtags = []
hearts = []
dollars = []


def append(name, strName):
    global descriptors
    global labels
    for i in range(len(name)):
        labels.append(strName)
        #python image library
        im = Image.open(name[i])
        gd = im.getdata()
        temp = []
        for pixel in gd:
            temp.append(pixel[0])
            temp.append(pixel[1])
            temp.append(pixel[2])
        #RGB Array
        descriptors.append(temp)

def readInFile(filePath):
    temp = []
    d = filePath
    for f in os.listdir(d):
        temp.append(os.path.join(d, f))
    return temp

def getPics():
    global smileys
    global hats
    global hashtags
    global hearts
    global dollars
    smileys = readInFile('/Users/katie/Documents/Data/01')
    hats = readInFile('/Users/katie/Documents/Data/02')
    hashtags = readInFile('/Users/katie/Documents/Data/03')
    hearts = readInFile('/Users/katie/Documents/Data/04')
    dollars = readInFile('/Users/katie/Documents/Data/05')

    names = [smileys, hats, hashtags, hearts, dollars]
    strNames = ['Smiley', 'Hat', 'Hashtag', 'Heart', 'Dollar']

    for i in range(len(names)):
        append(names[i], strNames[i])


def testData():
    global validationSet
    global smileys
    global hats
    global hashtags
    global hearts
    global dollars
    clf = pickle.load(open('proj3.p', 'rb'))

    validationImgs = [smileys.pop(), hats.pop(), hashtags.pop(), hearts.pop(), dollars.pop(), smileys.pop(),hats.pop(), hashtags.pop(), hearts.pop(), dollars.pop()]

    for i in range(len(validationImgs)):
        #python image library                                                          
        im = Image.open(validationImgs[i])
        gd = im.getdata()
        temp = []
        for pixel in gd:
            temp.append(pixel[0])
            temp.append(pixel[1])
            temp.append(pixel[2])

        validationSet.append([temp])

    for i in validationSet:
        print 'Validation prediction is: ', clf.predict(i)[0]
    print 'should be: smiley, hat, hashtag, heart, dollar, then repeat'


def myPickle():
    global labels
    global descriptors
    getPics()

    clf = sklearn.svm.SVC()
    clf.fit(descriptors, labels)
    pickle.dump(clf, open('proj3.p', 'wb'))


def printScore(labels, descriptors):
    model = sklearn.svm.SVC(kernel='linear',C=1)
    scores = sklearn.cross_validation.cross_val_score(model, descriptors, labels, cv = 10)
    sum = 0
    for i in range(len(scores)):
        sum += scores[i]

    score = sum / len(scores)
    print "Average accuracy is: ", score*100, "%."


def normalize(descriptors):
    min = 100000000000
    for i in descriptors:
        for j in i:
            if j < min:
                min = j
    max = 0
    for i in descriptors:
        for j in i:
            if j > max:
                max = j

    normalized = []
    for i in descriptors:
        tempArr = []
        for j in i:
            temp = j - min
            temp2 = temp / float(max-min)
            tempArr.append(temp2)
        normalized.append(tempArr)

    return normalized

def main():
    global descriptors
    global labels
    getPics()
    descriptors = normalize(descriptors)
    printScore(labels, descriptors)
    testData()
    myPickle()

main()




