# import numpy as np
import tensorflow as tf

def get_index(tens_, val):
  i = tf.constant(0)
  sz = tf.shape(tens_)[0]
  c = lambda i: tf.logical_and(tf.less(i ,sz), 
    tf.not_equal(tens_[i], val))
  b = lambda i: tf.add(i, 1)
  r = tf.while_loop(c, b, [i])
  # assert r is  tensor of rank 0
  return r

tens = tf.constant([])
val = tf.constant(4)
r = (tf.map_fn(lambda i: i, tens))
# (tens, val)

with tf.Session() as sess:
    print(sess.run(r))
# # NOTES:
# # The cache seems to be storing the name of the value instead of the tensor value
# # Think of a way to store the tensor value rather than
