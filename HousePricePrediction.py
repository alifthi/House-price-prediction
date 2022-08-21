from email.mime import image
import os
from turtle import shape
from unicodedata import numeric 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers as ksl
import numpy as np
import sklearn as sk
import pandas as pd
import cv2 as cv

class Model():
    def __init__(self,loadWeights = False,loadModel = False,weightAddr = None,modelAddr=None):
        if loadModel:
            self.net = self.loadModel(modelAddr)
        elif not(loadModel and loadWeights):
            self.net = self.buildModel()
        elif loadWeights:
            self.net = self.loadWeights()
            self.loadWeights(weightAddr)
        else :
            print('[Error] incompatible inputs !')
        self.bedroom = []
        self.bathroom = []
        self.frontal = []
        self.kitchen =  [] 
        self.txtFeatur = []
        self.const = None
        self.label = []
        



    def myLoss(y_true,y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred),axis = -1)
    def compile(self,loss,metrics,optim):
        self.net.compile(loss = loss,optimizer = optim ,metrics = metrics)

    def train(self,e,b,Data):
        hist = self.net.fit(Data,self.label,epochs = e,batch_size = b)
        return hist

    def buildModel(self):
        inpIm1 = ksl.Input((128,128,3),name = 'image1Input')
        inpIm2 = ksl.Input((128,128,3),name = 'image2Input')
        inpIm3 = ksl.Input((128,128,3),name = 'image3Input')
        inpIm4 = ksl.Input((128,128,3),name = 'image4Input')
        
        x = ksl.Conv2D(32,kernel_size=3,strides=1,padding='same')(inpIm1)
        x = ksl.Lambda(self.relu)(x)
        x = ksl.MaxPool2D([2,2])(x)
        x = ksl.Conv2D(64,kernel_size=3,padding='same')(x)
        x = ksl.Lambda(self.relu)(x)
        x = ksl.MaxPool2D([2,2])(x)
        x = ksl.Conv2D(128,kernel_size=3,padding='same')(x)
        x = ksl.Lambda(self.relu)(x)
        x = ksl.MaxPool2D([2,2])(x)

        w = ksl.Conv2D(32,kernel_size=3,strides=1,padding='same')(inpIm2)
        w = ksl.Lambda(self.relu)(w)
        w = ksl.MaxPool2D([2,2])(w)
        w = ksl.Conv2D(64,kernel_size=3,padding='same')(w)
        w = ksl.Lambda(self.relu)(w)
        w = ksl.MaxPool2D([2,2])(w)
        w = ksl.Conv2D(128,kernel_size=3,padding='same')(w)
        w = ksl.Lambda(self.relu)(w)
        w = ksl.MaxPool2D([2,2])(w)

        y = ksl.Conv2D(32,kernel_size=3,strides=1,padding='same')(inpIm3)
        y=ksl.Lambda(self.relu)(y)
        y = ksl.MaxPool2D([2,2])(y)
        y = ksl.Conv2D(64,kernel_size=3,padding='same')(y)
        y = ksl.Lambda(self.relu)(y)
        y = ksl.MaxPool2D([2,2])(y)
        y = ksl.Conv2D(128,kernel_size=3,padding='same')(y)
        y = ksl.Lambda(self.relu)(y)
        y = ksl.MaxPool2D([2,2])(y)

        z = ksl.Conv2D(32,kernel_size=3,strides=1,padding='same')(inpIm4)
        z = ksl.MaxPool2D([2,2])(z)
        z = ksl.Conv2D(64,kernel_size=3,padding='same')(z)
        z = ksl.MaxPool2D([2,2])(z)
        z = ksl.Conv2D(64,kernel_size=3,padding='same')(z)
        z = ksl.MaxPool2D([2,2])(z)
        

        inpTxtFeatures = ksl.Input((4,),name = 'numFeatureInput')
        t = ksl.Dense(1024,'relu')(inpTxtFeatures)
        t = ksl.Dense(512,'relu')(t)
        t = ksl.Dense(256,'relu')(t)
        t = ksl.Reshape((16,16,1))(t)



        concatInput = ksl.concatenate([w,x,y,z,t],axis=3)
        out = ksl.Conv2D(128,kernel_size=3,padding='same')(concatInput)
        out = ksl.MaxPool2D([2,2])(out)
        out = ksl.Conv2D(256,kernel_size=3,padding='same')(out)
        out = ksl.MaxPool2D([2,2])(out)
        out = ksl.Flatten()(out)
        out = ksl.Dense(128,'relu')(out)
        out = ksl.Dense(64,'relu')(out)
        out = ksl.Dense(1,'linear')(out)
        net = tf.keras.Model(inputs = [inpIm1,inpIm2,inpIm3,inpIm4,inpTxtFeatures],outputs = out)
        return net
    def predict(self,inputList):
        numericFeatur = inputList[4:]
        numericFeatur = pd.DataFrame(numericFeatur)
        max = numericFeatur.max()
        numericFeatur = numericFeatur/max
        images = inputList[:4]
        images = [self.preProcess(im) for im in images]
        pred = self.net.predict(np.concatenate((images,numericFeatur.to_numpy())))
        return pred*self.const

    def saveModel(self):
        self.net.save('./Models/model.h5')
        self.net.save_weights('./Models/modelWeights.h5')
    def loadWeights(self,weightAddr):
        net = self.buildModel()
        net.load_weghts(weightAddr)
        return net
    @staticmethod
    def loadModel(modelAddr):
        net = tf.kerasl.models.load_model(modelAddr)
        return net

 
        return image
    @staticmethod
    def relu(x):
        return tf.maximum( 0.0,x)
   
