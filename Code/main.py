from HousePricePrediction import  Model
from utils import utils
from tensorflow.keras.optimizers import Adam


model = Model()
txtPath = r'/home/alifathi/Documents/AI/Git/House price prediction/Data/HousesInfo.txt'
imgPath = r'/home/alifathi/Documents/AI/Git/House price prediction/Data/house_dataset'
util = utils(txtPath = txtPath,imagePath=imgPath)
const = util.const
model.compile(loss = 'mse',metrics = ['mse'],optim = Adam(learning_rate = 0.01))
Hist = model.train(1,512,[util.image[3],util.txtFeatur],util.label)
model.plotHistory(Hist)