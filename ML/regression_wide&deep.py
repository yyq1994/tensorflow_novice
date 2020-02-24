import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import os
import tensorflow.keras as keras

# 下载数据----------加利福尼亚房价
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# 分割数据
from sklearn.model_selection import train_test_split
x_train_all,x_test,y_train_all,y_test =train_test_split(housing.data,housing.target,random_state=7)
x_train,x_valid,y_train,y_valid = train_test_split(x_train_all,y_train_all,random_state=11)

# 归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

#网络构建    函数式API
input = keras.layers.Input(shape=x_train.shape[1:])
# deep
hidden1 = keras.layers.Dense(30,activation='relu')(input)
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)
# wide
concat = keras.layers.concatenate([input,hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs= [input],outputs =[output])

# 网络结构----不一样的
# model.summary()

# 模型编译
model.compile(loss='mean_squared_error',optimizer='adam')      #optimizer我用了 'sgd'几个epochloss就变成了nan了
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]

#模型训练
history = model.fit(x_train_scaled,y_train,validation_data=(x_valid_scaled,y_valid),epochs=100,callbacks=callbacks)

# 效果绘制
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)

 # 评估
print(model.evaluate(x_test_scaled,y_test))
