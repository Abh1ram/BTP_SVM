import tensorflow as tf

from dist_config import cluster

def f1():
    x = tf.constant(5)
    with tf.device("/job:worker/task:1/device:GPU:0"):
        y2 = x - 10
    return y2

with tf.device("/job:worker/task:0/cpu:0"):
    z = f1()
    y = z + 100

print("Starting")
with tf.Session("grpc://localhost:2222", 
    config=tf.ConfigProto(log_device_placement=True)) as sess:
    print("Contacting...")
    result = sess.run(y)
    print(result)
    