import pickle

import numpy as np
import tensorflow as tf

from var_free_graph import simple_kernel, svm_train, svm_eval, create_all_vars



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
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #   if tf.train.get_checkpoint_state('./svm_model/'):
    #     # Restore previous model and continue training
    #     ckpt = tf.train.latest_checkpoint('./svm_model/')
    #     saver.restore(sess, ckpt)
    #     with tf.variable_scope("svm_model", reuse=True):
    #       # n = tf.get_variable("n", dtype=tf.int32)
    #       # b = tf.get_variable("b")
    #       # marg_vec_x = tf.get_variable("marg_vec_x")
    #       all_vars = create_all_vars(model_params)
    #   req = sess.run(all_vars)

# print(all_vars.marg_vec_x.shape, all_vars.err_vec_x.shape, all_vars.rem_vec_x.shape)
