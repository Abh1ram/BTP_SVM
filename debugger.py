import pickle
import sys

import numpy as np

from btp_python import SVM_Online


C_SVM = 5.
RESERVE_THRESHOLD = 5000.

model_params = {
  "C" : C_SVM,
  "eps" : RESERVE_THRESHOLD,
  }

X_train, y_train = pickle.load(open("tf_svm/problem_cancer1.p", "rb"))
# X_test, y_test = pickle.load(open("tf_svm/test_data_1.p", "rb"))
# scale y_train inputs
y_train = y_train*2-1
# y_test = y_test*2-1
print(X_train.shape)
y_train = np.reshape(y_train, [-1,])
# y_test = np.reshape(y_test, [-1,])
print(np.unique(y_train))
print(y_train.shape)
# input()
# input()
svm = SVM_Online(X=X_train, y=y_train, C_svm=C_SVM)
svm.train_all()
print(len(svm.Margin_v), len(svm.Error_v))
# pred_y = []
# for x in X_test:
#     pred_y.append(svm.predict(x))

# print(pred_y)
# from sklearn.metrics import accuracy_score
# print("ACCURACY: ", accuracy_score(pred_y, y_test))
# # print(len(svm.Margin_v), len(svm.Error_v))
