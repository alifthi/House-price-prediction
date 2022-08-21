import numpy as np
from glob import glob
import cv2 as cv
import sklearn as sk
class utils():
    def __init__(self,txtPath,imagePath) -> None:
        self.Data = self.loadData(txtPath,imagePath)


    def split(self):
        split = sk.modelselection.train_test_split(self.Data,test_size = 0.2)
        return split
    def loadData(self,txtPath,imgPath):
        self.txtFeatur = pd.DataFrame([[float(t) for t in x.split('\n')[0].split(' ')] for x in open(txtPath).readlines()])
        self.txtFeatur.T.loc[[0,1,2,3],:]
        self.label = self.txtFeatur.loc[:,4] 
        self.txtFeatur = self.txtFeatur.T.loc[[0,1,2,3],:].T 
        max = self.txtFeatur.max()
        self.txtFeatur = self.txtFeatur/max
        self.const = np.max(self.label)
        self.label = self.label/self.const


        for i,p in enumerate(glob(imgPath+'/*')):
            (bedroom,bathroom,frontal,kitchen) = [[],[],[],[]]
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
        bedroom = np.array(self.bedroom)
        bathroom = np.array(self.bathroom)
        frontal = np.array(self.frontal)
        kitchen = np.array(self.kitchen)
        return np.concatenate((bedroom,bathroom,frontal,kitchen))
    
    @staticmethod
    def preProcess(image):
        image = cv.resize(image,(128,128))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        