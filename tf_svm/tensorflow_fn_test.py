# import numpy as np
# Assuming None control dep is bad and not used
import tensorflow as tf
from collections import namedtuple

# config=tf.ConfigProto(log_device_placement=True)
with tf.device("/cpu:0"):
    with tf.Session() as sess:
        i = tf.constant(1., dtype=tf.float64)
        j = tf.constant(2., dtype=tf.float64)
        z = i*j
        print(sess.run([z]))

