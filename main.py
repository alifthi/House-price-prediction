from HousePricePrediction import  Model

model = Model()
txtPath = r'/home/alifathi/Documents/AI/Git/House price prediction/Data/HousesInfo.txt'
imgPath = r'/home/alifathi/Documents/AI/Git/House price prediction/Data/house_dataset'
model.loadData(txtPath=txtPath,imgPath=imgPath)
model.compile(loss = 'mse',metrics = ['mse'],optim = tf.keras.optimizers.Adam(learning_rate = 0.01))
model.train(1,512)