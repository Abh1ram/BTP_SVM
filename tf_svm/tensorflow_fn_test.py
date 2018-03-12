# import numpy as np
# Assuming None control dep is bad and not used
import tensorflow as tf

def f1(i, j):
    # with tf.control_dependencies(None):
        # j = tf.get_variable("j")
        k = tf.map_fn(lambda i: j[i], j)
        return (i+1, k)

def f2():
    j = tf.get_variable("j", dtype=tf.int32).read_value()
    return tf.while_loop(lambda i,j: tf.less(i, 10), lambda i,j: f1(i,j), [i,j])

with tf.Session() as sess:
    i = tf.constant([1,4.])
    # j = tf.constant(3.)
    # with tf.variable_scope("sc") as sc:
    #     j = tf.get_variable("j", initializer=tf.constant([0 ,1]))
    # sess.run(tf.global_variables_initializer())
    # with tf.variable_scope(sc, reuse=True):
    r = tf.reshape(tf.reshape(i, [-1, 1]), [-1,])
    print(sess.run(r))
# # NOTES:
# # The cache seems to be storing the name of the value instead of the tensor value
# # Think of a way to store the tensor value rather than
