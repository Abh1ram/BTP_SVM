import pickle

import numpy as np
import tensorflow as tf

from var_free_graph import simple_kernel, svm_train, svm_eval



C_SVM = 5.
RESERVE_THRESHOLD = 5.

model_params = {
  "C" : C_SVM,
  "eps" : RESERVE_THRESHOLD,
  "kernel" : simple_kernel,
  }

X_train, y_train = pickle.load(open("random_cancer.p", "rb"))
with tf.device("/cpu:0"):
    _, all_vars = svm_train(X_train, y_train, model_params)

print(all_vars.marg_vec_x.shape, all_vars.err_vec_x.shape, all_vars.rem_vec_x.shape)
