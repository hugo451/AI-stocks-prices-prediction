import pandas as pd
import numpy as np
import pandas_datareader as web

stocks = ['ITSA4.SA', 'BBAS3.SA', 'BBDC4.SA', 'ITUB4.SA', 'PETR4.SA', 'GGBR4.SA', 'ABEV3.SA', 'CIEL3.SA', 'WEGE3.SA', 'BRFS3.SA', 'VALE3.SA', 'USIM5.SA', 'CSNA3.SA']

datafra = pd.DataFrame()

for stock in stocks:

    df = web.DataReader(stock, data_source="yahoo", start="2015-01-01", end="2020-01-19")

    #Dataframe with the columns:
    data = df.filter(["Open", "Close", "High", "Low", "Volume"])
    #Converting the data for a numpy array
    dataset = data.values
    #Training data len:
    training_data_len = len(dataset)

    train_data = dataset[0:training_data_len, 0:]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(np.array(train_data[i-60:i, :]))
        y_train.append(np.array(train_data[i, 0:4]))

    d = {'X_TRAIN' : x_train, 'Y_TRAIN' : y_train}

    dafr = pd.DataFrame(d)

    datafra = datafra.append(dafr).reset_index(drop=True)
print(y_train[0])

f = open("data.json", 'w')

f.write(datafra.to_json())