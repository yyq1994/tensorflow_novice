import numpy as np
import tensorflow.keras as keras
from sklearn.datasets import fetch_california_housing
# 数据下载
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

# 建立模型
def build_model(hidden_layers = 1,layer_size = 30,learning_rate = 3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size,
                                 activation='relu',
                                 input_shape=x_train_scaled.shape[1:]))
    for _ in range(hidden_layers-1):
        model.add(keras.layers.Dense(layer_size,'relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(loss='mse',optimizer=optimizer)
    return model

# 转化为sklearn  model
sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# 定义搜索参数
from scipy.stats import reciprocal          #获得连续数据
param_distribution = {
    'hidden_layers':[1,2,3,4],
    'layer_size':np.arange(1,100),
    'learning_rate':reciprocal(1e-4,1e-2)
}

# 搜索参数
from sklearn.model_selection import RandomizedSearchCV
random_search_cv = RandomizedSearchCV(sklearn_model,
                                      param_distribution,
                                      10)
# 回调函数
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]

# 训练模型
random_search_cv.fit(x_train_scaled,y_train,
                     epochs =100,
                     validation_data = (x_valid_scaled,y_valid),
                     callbacks = callbacks)

print('最好的参数：',random_search_cv.best_params_)
print('最好得分：',random_search_cv.best_score_)
print('最好模型：',random_search_cv.best_estimator_)
best_model = random_search_cv.best_estimator_

# 评估模型
best_model.evaluate(x_test,y_test)


# def plot_learning_curves(history):
#     pd.DataFrame(history.history).plot(figsize=(8,5))
#     plt.grid(True)
#     plt.gca().set_ylim(0,1)
#     plt.show()
#
# plot_learning_curves(history)
