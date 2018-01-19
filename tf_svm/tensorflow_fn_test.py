import numpy as np
import tensorflow as tf

age = np.arange(4) * 1.0
height = np.arange(32, 36)
x = {'age': age, 'height': height}
y = np.arange(-32, -28)

with tf.Session() as sess:

  input_fn = tf.estimator.inputs.numpy_input_fn(
      x, y, batch_size=2, shuffle=False, num_epochs=1)

  for i in range(3):
      tup = input_fn()
      sess.run(tup)
  # print(sess.run(tup))


# featrs = [tf.feature_column.numeric_column()]