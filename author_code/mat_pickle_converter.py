import pickle

import numpy as np
import scipy.io as sio

def rescale(y):
    y = y*2-1
    return np.reshape(y, [-1, 1])

# mat_contents = sio.loadmat("iris_train.mat")
# print(mat_contents.keys())
# pickle.dump((mat_contents["xtrain"], mat_contents["ytrain"]),
#     open("train_data.p", "wb"))
# print(mat_contents["x"].shape)
# print(np.unique(mat_contents["y"]))
X_train, y_train = pickle.load(open("train_data_0.p", "rb"))
X_test, y_test = pickle.load(open("test_data_0.p", "rb"))
# rescale y_train and y_test
y_train = rescale(y_train)
y_test = rescale(y_test)
print(y_train.shape, np.unique(y_train))
sio.savemat("iris_train.mat", {"xtrain": X_train, "ytrain": y_train})
sio.savemat("iris_test.mat", {"xtest": X_test, "ytest": y_test})
