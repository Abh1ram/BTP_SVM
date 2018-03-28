# Config of the cluster
# I can create a simple rpc which asks for task_id and ClusterSpec
import tensorflow as tf

cluster = tf.train.ClusterSpec(
    {"worker": ["localhost:2222", "172.16.114.80:2223"]})

master_worker_rpc = "grpc://localhost:2222"