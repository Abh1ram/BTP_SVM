# import numpy as np
import tensorflow as tf
from collections import namedtuple

def f2(obj):
    return obj

def f1(i, p):
    return (i+1, p)

with tf.Session() as sess:
    Pair = namedtuple('Pair', 'j, k')
    ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
    c = lambda i, p: i < 10
    b = lambda i, p: (i + 1, p)
    ijk_final = tf.while_loop(c, lambda i, p: f1(i, p), ijk_0)
    print(sess.run(ijk_final))

# # NOTES:
# # The cache seems to be storing the name of the value instead of the tensor value
# # Think of a way to store the tensor value rather than
