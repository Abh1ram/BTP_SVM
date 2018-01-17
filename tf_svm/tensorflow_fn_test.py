import numpy as np
import tensorflow as tf

feat = np.array([1,2,3])
labels = np.array([1,2,3])

with tf.Session() as session:
    fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": feat},
        y = labels,
        shuffle=False
        )

    for i in range(3):
        tup = fn()
        print(tup[0]["x"], tup[1])