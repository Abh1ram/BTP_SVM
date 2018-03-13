# import numpy as np
# Assuming None control dep is bad and not used
import tensorflow as tf
from collections import namedtuple

# Test shape invariants
def f1(i, p):
    tens = tf.concat([p.i, p.j], 0)
    return (i+1, p._replace(i=tens))

def kR(R, k):
    c1 = tf.shape(R)
    c1 = tf.concat([tf.reshape(k, [1,]), tf.reshape(c1[0]-k-1, [1,])], 0)
    c1 = tf.reshape(c1, [1, 2])
    c2 = tf.constant([[0,0]])
    row_k = tf.reshape(R[k, :], [1, tf.shape(R)[0]])
    col_k = tf.reshape(R[:, k], [-1, 1])
    pad_row = tf.concat([c1, c2], 0)
    pad_col = tf.concat([c2, c1], 0)
    return (tf.matmul(tf.pad(col_k, pad_col), tf.pad(row_k, pad_row)), 
        tf.matmul(col_k, row_k))

def f2():
    j = tf.get_variable("j", dtype=tf.int32).read_value()
    return tf.while_loop(lambda i,j: tf.less(i, 10), lambda i,j: f1(i,j), [i,j])

with tf.Session() as sess:
    # Pair = namedtuple("Pair", ["i", "j"])
    # print(Pair._fields)
    # p = Pair(i=tf.constant([1, 4.]), j=tf.constant([3, 4.]))
    # shp = Pair(tf.TensorShape([None,]), p.j.get_shape())
    # iter_ct = tf.constant(0.)
    # r = tf.while_loop(lambda i,p: tf.less(i, 3), f1, [iter_ct, p],
    #     shape_invariants=[iter_ct.get_shape(), shp])
    # R = tf.constant([[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12],
    #     [13, 14, 15, 16]])
    # # col2 = tf.reshape(R[:, 2], [3, 1])
    # paddings = tf.constant([[0,1], [0, 1]])
    # R1 = tf.pad(R, paddings, "CONSTANT")
    # R1[2, :] = R[2, :]
    # R1[:, 2] = R[:, 2]
    # r = tf.reshape(j, [-1, -1])
    # with tf.variable_scope("sc") as sc:
    #     j = tf.get_variable("j", initializer=tf.constant([0 ,1]))
    # sess.run(tf.global_variables_initializer())
    # with tf.variable_scope(sc, reuse=True):
    i = tf.constant([1., 2, 3])
    j = tf.constant([2,3,4.])
    print(sess.run(i*j))
    # print(sess.run(r2))
# # NOTES:
# # The cache seems to be storing the name of the value instead of the tensor value
# # Think of a way to store the tensor value rather than
