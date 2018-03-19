# Get task number from command line
import sys
task_number = int(sys.argv[1])

import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["127.0.0.1:2222", "127.0.0.1:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=task_number)

print("Starting server #{}".format(task_number))

server.start()
server.join()