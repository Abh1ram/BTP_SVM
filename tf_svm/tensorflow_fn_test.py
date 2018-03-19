# import numpy as np
# Assuming None control dep is bad and not used
import tensorflow as tf
from collections import namedtuple

# config=tf.ConfigProto(log_device_placement=True)
with tf.device("/cpu:0"):
    with tf.Session() as sess:
        i = tf.constant([1,2,3])
        j = tf.constant([1,4, 3])
        z = tf.cast(tf.equal(i,j), tf.int32)
        print(sess.run(z))

