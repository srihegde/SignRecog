import cv2
import numpy as np
import segment
import itertools
from sklearn.naive_bayes import GaussianNB

SIZE = 40
enableShow = False

words = []      #list containing number of words ina video
sentences = []      #list of sentences
wordSize = []       #list containing avg word size(in terms of frame number)
frameCount = []
datapoints = []
labels = []

with open('DiggedData.dat', 'r') as f:
    line = f.readlines()
    for i in xrange(len(line)):
        line[i] = line[i].split(';')
        sentences.append(line[i][0])
        words.append(int(line[i][1].strip()))
        wordSize.append(int(line[i][3].strip()))
        frameCount.append(int(line[i][2].strip()))
        break        


for i in xrange(len(sentences)):
    line = sentences[i].strip().split()
    if(line[0][0] == '['):
        line = line[1:]
    if(line[len(line)-1][0] == '['):
        line = line[:len(line)-1]
    for j in xrange(len(line)):
        labels.append(line[j])

        
videos = segment.getDenseOptFlow(enableShow)


for i in xrange(len(videos)):
    video = videos[i]
    if(sentences[i][0] == '['):
        video = video[5:]
    if(sentences[i][len(sentences)-1] == ']'):
        video = video[:len(video)-5]

    start = 0
    for j in xrange(words[i]):
        
        datapoint = video[start:start+wordSize[i]]
        #print np.array(datapoint).shape()
        datapoint = list(itertools.chain(*datapoint))
        datapoint = list(itertools.chain(*datapoint))
        datapoint = list(itertools.chain(*datapoint))
        l = len(datapoint)
        reqLen = 250*336*2*40
        reqPadLen = reqLen - l
        padList = [0]*reqPadLen
        datapoint.extend(padList)
        print len(datapoint)
        datapoints.append(datapoint)
        start = start+wordSize[i]



datapoints = np.asarray(datapoints)
labels = np.asarray(labels)

classifier = GaussianNB()
classifier.fit(datapoints,labels)


    
