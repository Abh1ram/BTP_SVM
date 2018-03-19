import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

def f1():
    x = tf.constant(5)
    with tf.device("/job:local/task:1"):
        y2 = x - 10
    return y2

with tf.device("/job:local/task:0"):
    z = f1()
    y = z + 100

print("Starting")
with tf.Session("grpc://localhost:2222", 
    config=tf.ConfigProto(log_device_placement=True)) as sess:
    print("Contacting...")
    result = sess.run(y)
    print(result)
    