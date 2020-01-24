import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Tratamento de dados
df = pd.read_json("Data/data.json")

data = df.filter(['X_TRAIN'])

x_train = data.values

data = df.filter(['Y_TRAIN'])

y_train = data.values

x_train_data = []
y_train_data = []

scaler = MinMaxScaler(feature_range=(0,1))

for i in range(x_train.shape[0]):
    x_train_data.append(scaler.fit_transform(x_train[i][0]))
    y_train_data.append(y_train[i][0])


x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

y_train_data = scaler.fit_transform(y_train_data)

#Modelo de predição
model = Sequential()
model.add(LSTM(60, return_sequences= True, input_shape = (x_train_data.shape[1], x_train_data.shape[2])))
model.add(LSTM(60, return_sequences= False))
model.add(Dense(40))
model.add(Dense(4))

model.compile(optimizer='adamax', loss="mse")

model.fit(x_train_data, y_train_data, batch_size=1, epochs=110, use_multiprocessing=True)

#Salvamento dos pesos do modelo treinado
model.save("stock_prices_predictions_v2.0.h5")
