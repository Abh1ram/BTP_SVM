import pickle

import numpy as np
import tensorflow as tf

from svm_graph64 import simple_kernel, svm_train, svm_eval, create_svm_variables



C_SVM = 1.
RESERVE_THRESHOLD = 5000.

model_params = {
  "C" : C_SVM,
  "threshold" : RESERVE_THRESHOLD,
  "kernel" : simple_kernel,
  }

X_train, y_train = pickle.load(open("problem_cancer1.p", "rb"))
X_test, y_test = pickle.load(open("test_data_0.p", "rb"))
y_train = y_train*2-1
y_test = y_test*2-1
with tf.device("/cpu:0"):
  _ = create_svm_variables(X_train.shape[1])  
  _, all_vars = svm_train(X_train, y_train, model_params)
  pred_y = svm_eval(X_test, model_params)
  print(pred_y)
  # print(acc)
  from sklearn.metrics import accuracy_score
  print("ACCURACY: ", accuracy_score(pred_y, y_test))

