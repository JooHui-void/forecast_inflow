import numpy as np
import pandas as pd
import os
import openpyxl
import tensorflow as tf
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from tensorflow.keras.layers import LSTM,Dense
from sklearn.metrics import mean_squared_error


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



train_num_24 = 2789
train_num_25 = 2892
train_num_19 = 1934

Location = "C:/Users/이시형/Desktop/data.xlsx"
df = pd.read_excel(Location, engine="openpyxl")

# df_metrix 만드는 과정
df_matrix = df.to_numpy()
df_matrix = df_matrix[1:train_num_25, 5:-2]
# print(df_matrix)

df_matrix = df_matrix.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
df_matrix = scaler.fit_transform(df_matrix)

# print(df_matrix)
lag = series_to_supervised(df_matrix, 2)

# print(lag)
# print(lag.shape[1])
# for i in range(215,258):
#     lag.drop(lag.columns[[i]], axis = 1, inplace = True)


lag = lag.values

tmp = lag.shape[1]-42

lag = lag[:,0:tmp]
print(lag.shape)



n_train_hours = 1500
train = lag[:n_train_hours, :]
test = lag[n_train_hours:, :]

train_X, train_y = train[:, :-1], train[:, -1]
print(train_X.shape)
test_X, test_y = test[:, :-1], test[:, -1]
# print(test_y)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
print(train_X.shape[0],train_X.shape[1],train_X.shape[2])
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



#testing
model = tf.keras.Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
# fit network
history = model.fit(train_X, train_y, epochs=97, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.show()

pred = model.predict(test_X)
plt.plot(pred,label='prediction')
plt.plot(test_y,label='real')
plt.show()


RMSE = mean_squared_error(test_y,pred)**0.5
print(RMSE)

#
# yhat = model.predict(test_X)
# plt.plot(yhat)

# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
# pred = model.predict(test_X)
#
#
#
# plt.figure(figsize=(12, 9))
# plt.plot(test_y, label='actual')
# plt.plot(pred, label='prediction')
# plt.legend()



# 정보 뽑아냄
waternum = np.array(df["홍수사상번호"])
waternum = waternum[1:train_num_25]  # 홍수번호 1~24까지만

waterin = np.array(df["유입량"])
waterin = waterin[1:train_num_25]
index = np.array(df["index"])
index = index[1:train_num_25]
# print(waterin)



# print(scaled)


# plt.plot(index, waterin)
# plt.xticks(index, waternum)
# plt.xlabel("index")
# plt.ylabel("waterin")
# ax = plt.axes()
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

# plt.plot(waterin[train_num_19:train_num_25])
# plt.show()