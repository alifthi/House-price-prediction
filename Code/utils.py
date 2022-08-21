import numpy as np
from glob import glob
import cv2 as cv
import sklearn as sk
import pandas as pd
class utils():
    def __init__(self,txtPath,imagePath):
        self.image,self.txtFeature , self.label = self.loadData(txtPath,imagePath)
        self.const = None

    def split(self):
        split = sk.modelselection.train_test_split(self.Data,test_size = 0.2)
        return split
    def loadData(self,txtPath,imgPath):
        txtFeatur = pd.DataFrame([[float(t) for t in x.split('\n')[0].split(' ')] for x in open(txtPath).readlines()])
        txtFeatur.T.loc[[0,1,2,3],:]
        label = txtFeatur.loc[:,4] 
        txtFeatur = txtFeatur.T.loc[[0,1,2,3],:].T 
        max = txtFeatur.max()
        txtFeatur = txtFeatur/max
        self.const = np.max(label)
        label = label/self.const


        for i,p in enumerate(glob(imgPath+'/*')):
            bedroom = []
            bathroom = []
            frontal = []
            kitchen = []
            place = p.split('/')[-1].split('.')[0].split('_')[1]
            if place == 'bedroom':
                bedroom.append(self.preProcess(cv.imread(p)))
            elif place == 'bathroom':
                bathroom.append(self.preProcess(cv.imread(p)))
            elif place == 'frontal':
                frontal.append(self.preProcess(cv.imread(p)))
            else:
                kitchen.append(self.preProcess(cv.imread(p)))
            if i%500 == 0:
                print('[INFO] {}th image loaded'.format(i))
        # bedroom = np.array(bedroom)
        # bathroom = np.array(bathroom)
        # frontal = np.array(frontal)
        # kitchen = np.array(kitchen)
        return ([bedroom,bathroom,frontal,kitchen],txtFeatur, label)
    @staticmethod
    def preProcess(image):
        image = cv.resize(image,(128,128))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        return image