import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import tensorflow.keras as keras
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

from sklearn.model_selection import train_test_split
x_train_all,x_test,y_train_all,y_test =train_test_split(housing.data,housing.target,random_state=7)
x_train,x_valid,y_train,y_valid = train_test_split(x_train_all,y_train_all,random_state=11)

# 归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)
# 多输入
input_wide = keras.layers.Input(shape=[5])                  #多输入改动
input_deep = keras.layers.Input(shape=[6])                  #多输入改动
hidden1 = keras.layers.Dense(30,activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)
concat = keras.layers.concatenate([hidden2,input_deep])     #多输入改动
output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)                    #多输出改动
model = keras.models.Model(inputs=[input_wide,input_deep],outputs=[output,output2])       #多输入多输出




# 网络结构
# model.summary()

model.compile(loss='mean_squared_error',optimizer='adam')
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]
x_train_scaled_wide = x_train_scaled[:,:5]                  #多输入改动
x_train_scaled_deep = x_train_scaled[:,2:]                  #多输入改动
x_valid_scaled_wide = x_valid_scaled[:,:5]                  #多输入改动
x_valid_scaled_deep = x_valid_scaled[:,2:]                  #多输入改动
x_test_scaled_wide = x_test_scaled[:,:5]                    #多输入改动
x_test_scaled_deep = x_test_scaled[:,2:]                    #多输入改动



history = model.fit([x_train_scaled_wide,x_train_scaled_deep],[y_train,y_train],          #多输入多输出改动
                    validation_data=([x_valid_scaled_wide,x_valid_scaled_deep],[y_valid,y_valid]),      #多输入多输出改动            #多输入改动
                    epochs=100,callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)

 # 评估
print(model.evaluate([x_test_scaled_wide,x_test_scaled_deep],[y_test,y_test]))          #多输入多输出