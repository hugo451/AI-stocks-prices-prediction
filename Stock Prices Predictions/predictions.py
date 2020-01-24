from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web

#Fazendo a filtragem dos dados para predição:
df = web.DataReader('ITSA4.SA', data_source='yahoo', start='2019/10/23', end='2020/01/23')

df = df.filter(["Open", "Close", "High", "Low", "Volume"])

data = df.values

x_test = data[0:60]
y_test = data[60:]

#Escalando os valores e ajustando as medidas dos dados:
scaler = MinMaxScaler(feature_range=(0, 1))

x_test = scaler.fit_transform(x_test)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))


#Carregando o modelo previamente feito:
my_model = load_model('stock_prices_predictions.h5')

#Fazendo a predição da ação
prediction = my_model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

print(prediction)
print(y_test)