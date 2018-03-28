# Get task number from command line
import sys

import tensorflow as tf

from dist_config import cluster

# get the task id
task_number = int(sys.argv[1])

server = tf.train.Server(cluster, job_name="worker", task_index=task_number)

print("\n-------Starting server #{}\n".format(task_number))

server.start()
server.join()
