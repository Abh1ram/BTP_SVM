# Reimplementation of Cauwenbergh's Algo
# 1. Mostly variable free to remove most of the errors in while loops
#################### Ofcourse, I can't do that. No tensors other than variables are remembered
#################### through session calls. Status Quo reached again
#################### I could try to remove all operations interacting diff frame tensors
# 2. No more indices - store the points directly and move them
import os

import tensorflow as tf

from collections import namedtuple

from data_loader import extract_data

path_ = os.path.join("log_graph")

C_SVM = 5.
RESERVE_THRESHOLD = 5.

INF = 10**12 + 0.

# Simple dot product - returns a rank 0 tensor/scalar
def simple_kernel(x1, x2):
  return tf.tensordot(x1, x2, [0,0])

model_params = {
  "C" : C_SVM,
  "eps" : RESERVE_THRESHOLD,
  "kernel" : simple_kernel,
  }

# Calc f(x) = SUM (alpha_j * y_j * kernel(x_j, x)) for j in support vectors - marg, err
def calc_f(all_vars, params):
  alpha_err_ = tf.map_fn(lambda x: params["C"], all_vars.err_vec_x,
    dtype=tf.float32)
  # More parallelism
  supp_vec_x = tf.concat([all_vars.marg_vec_x, all_vars.err_vec_x], 0)
  supp_vec_y = tf.concat([all_vars.marg_vec_y, all_vars.err_vec_y], 0)
  alpha_supp_ = tf.concat([all_vars.alpha_marg, alpha_err_], 0)
  # Calc f(x) = SUM (alpha_j * y_j * kernel)
  kernel_vals = tf.map_fn(lambda x: params["kernel"](x, all_vars.x_c),
    supp_vec_x)
  return tf.tensordot(alpha_supp_, supp_vec_y*kernel_vals, [0,0]) + all_vars.b

# g(x,y) = f(x)*y - 1
def calc_g(all_vars, params):
  return calc_f(all_vars, params) * all_vars.y_c - 1.

def add_to_rem(eps, g_c, x_c, y_c, all_vars):
  rem_vec_x_ = tf.concat([all_vars.rem_vec_x, tf.reshape(x_c, [1, -1])], 0)
  rem_vec_y_ = tf.concat([all_vars.rem_vec_y, tf.reshape(y_c, [1,])], 0)
  g_rem_ = tf.concat([all_vars.g_rem, tf.reshape(g_c, [1,])], 0)

  return tf.cond(tf.less(g_c, eps),
    lambda: all_vars._replace(rem_vec_x=rem_vec_x_, rem_vec_y=rem_vec_y_,
      g_rem=g_rem_),
    lambda: all_vars)


# Terminate if candidate g_c equals 0 or candidate alpha_c equals C
def termn_condn(C_svm, all_vars):
  print_op = tf.Print(tf.constant(0), [all_vars.alpha_c, all_vars.g_c],
    "Checking while looop termn_condn")
  with tf.control_dependencies([print_op]):
    return tf.logical_or(tf.equal(all_vars.g_c, 0.),
      tf.equal(all_vars.alpha_c, C_svm))


def get_beta(kernel_fn, x_c, y_c, all_vars):
  # Calculate Q = y1*y2*K(x1*x2)
  q_tens = tf.map_fn(lambda x: kernel_fn(x_c, x), all_vars.marg_vec_x)
  q_tens = q_tens * y_c
  q_tens = q_tens * all_vars.marg_vec_y
  # Concat y_c in the starting
  q_tens = tf.concat([tf.reshape(y_c, [1,]), q_tens], 0)
  # Reshape q_tens for matrix mult
  q_tens_reshaped = tf.reshape(q_tens, [-1, 1])
  beta = -1 * tf.matmul(all_vars.R, q_tens_reshaped)
  # return reshaped beta
  return tf.reshape(beta, [-1,])

def mini_iter(params, i, all_vars):
  printer = tf.Print(tf.constant(0), [], "Inside body")
  # Handle case of matrix containing inf
  R_ = tf.cond(
    tf.equal(tf.shape(all_vars.marg_vec_x)[0], 0),
    lambda: tf.constant([[-INF]]),
    lambda: all_vars.R)
  all_vars = all_vars._replace(R=R_)
  # Calculate beta
  beta = get_beta(params["kernel"], all_vars.x_c, all_vars.y_c, all_vars)
  with tf.control_dependencies([beta]):
    return (i+1, all_vars)


def fit_point(params, all_vars):
  iter_ct = tf.constant(0.)
  fin_i, fin_vars = tf.while_loop(
    cond=lambda i, p: tf.logical_and(tf.logical_not(termn_condn(params["C"], p)),
      tf.less(i, 1)),
    body=lambda i, p: mini_iter(params, i, p), 
    loop_vars=(iter_ct, all_vars),
    parallel_iterations=1,
    back_prop=False,
    name="FIT_WHILE"
    )
  return fin_vars



def svm_train(x_, y_, params, all_vars):
  n_ = all_vars.n + 1
  # add new point
  all_vars = all_vars._replace(x_c=x_, y_c=y_, n=n_, alpha_c=tf.constant(0.))
  # calculate g_c
  g_c_ = calc_g(all_vars, params)
  # g_c_ = 1.
  all_vars = all_vars._replace(g_c=g_c_)
  # condition on the value of g
  return tf.cond(tf.greater(g_c_, 0),
    lambda: add_to_rem(params["eps"], g_c_, x_, y_, all_vars),
    lambda: fit_point(params, all_vars)
    )

# Create a namedtuple with all the required tensors
def create_all_vars(scope):
  AllVars = namedtuple("AllVars", ["n", "b", "marg_vec_x", "marg_vec_y",
    "alpha_marg", "err_vec_x", "err_vec_y", "g_err", "rem_vec_x", "rem_vec_y",
    "g_rem", "x_c", "y_c", "alpha_c", "g_c", "R", "Q_s"])

  all_vars = AllVars(
    n=tf.get_variable("n", dtype=tf.int32).read_value(),
    b=tf.get_variable("b").read_value(),
    marg_vec_x=tf.get_variable("marg_vec_x").read_value(),
    marg_vec_y=tf.get_variable("marg_vec_y").read_value(),
    alpha_marg=tf.get_variable("alpha_marg").read_value(),
    err_vec_x=tf.get_variable("err_vec_x").read_value(),
    err_vec_y=tf.get_variable("err_vec_y").read_value(),
    g_err=tf.get_variable("g_err").read_value(),
    rem_vec_x=tf.get_variable("rem_vec_x").read_value(),
    rem_vec_y=tf.get_variable("rem_vec_y").read_value(),
    g_rem=tf.get_variable("g_rem").read_value(),
    x_c=tf.constant([]), y_c=tf.constant(0.),
    alpha_c=tf.constant(0.), g_c=tf.constant(0.),
    R=tf.get_variable("R").read_value(),
    Q_s=tf.get_variable("Q_s").read_value())

  return all_vars

def create_svm_variables(x_shape):
  with tf.variable_scope("svm_model") as scope:
    # Variable creation
    # count of the number of data points seen
    n_ = tf.get_variable("n", initializer=tf.constant(0), trainable=False,
      dtype=tf.int32)
    # offset
    b_ = tf.get_variable("b", initializer=tf.constant(0.), trainable=False,)

    # Variable tensor - representing the Margin Support vector indices
    marg_vec_x_ = tf.get_variable("marg_vec_x", shape=[0, x_shape],
      validate_shape=False)
    marg_vec_y_ = tf.get_variable("marg_vec_y", initializer=tf.constant([]))

    # Variable tensor - representing the Error Support vector indices
    err_vec_x_ = tf.get_variable("err_vec_x", shape=[0, x_shape],
      validate_shape=False)
    err_vec_y_ = tf.get_variable("err_vec_y", initializer=tf.constant([]))

    # Variable tensor - representing the Remaining vector indices
    rem_vec_x_ = tf.get_variable("rem_vec_x", shape=[0, x_shape],
      validate_shape=False)
    rem_vec_y_ = tf.get_variable("rem_vec_y", initializer=tf.constant([]))

    # Variable for alpha of margin vectors
    alpha_marg_ = tf.get_variable("alpha_marg", shape=[0],
      validate_shape=False)
    # Variable denoting g of error support vectors
    g_err_ = tf.get_variable("g_err", shape=[0],
      validate_shape=False)
    # Variable denoting g of remaining vectors
    g_rem_ = tf.get_variable("g_rem", shape=[0],
      validate_shape=False)

    # Variable for the inverse Jacobian matrix R - initially Inf
    R_ = tf.get_variable("R", initializer=tf.constant([[INF]]), dtype=tf.float32,
      validate_shape=False)
    # Jacobian
    Q_s = tf.get_variable("Q_s", initializer=tf.constant([[0.]]))
    return scope

def update_vars(scope, all_vars):
  with tf.variable_scope(scope, reuse=True):
    op_list = []
    for field in set(all_vars._fields) - set(["x_c", "y_c", "alpha_c", "g_c", "n"]):
      op_list.append(tf.assign(tf.get_variable(field), getattr(all_vars, field),
        validate_shape=False))
    # separate operation for n due to int32
    op_list.append(tf.assign(tf.get_variable("n", dtype=tf.int32), all_vars.n))
    with tf.control_dependencies(op_list):
      return tf.constant(1.)


# svm_model_fn
def svm_model_fn():
  # make these command line args
  train_file = "data_1.csv"
  x_train, y_train = extract_data(train_file, "csv")
  # model_params["shape"] = x_train.shape[1]

  # Placeholders for inputs
  x_ = tf.placeholder(tf.float32, shape=(x_train.shape[1]))
  y_ = tf.placeholder(tf.float32, shape=())

  # Create namedtuple
  scope = create_svm_variables(x_shape=x_train.shape[1])
  with tf.variable_scope(scope, reuse=True):
    all_vars = create_all_vars(scope)
  
  # TODO : Create variables to store model
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    all_vars_upd = svm_train(x_, y_, model_params, all_vars)
    tr = update_vars(scope, all_vars_upd)
    for i in range(2):
      print("\n \n")
      print(sess.run([tr, all_vars_upd], feed_dict={x_ : x_train[i], y_ : y_train[i]}))
        # y_all_ = tf.get_variable("y_all")
        # print(sess.run(y_all_))
      # writer = tf.summary.FileWriter(path_, sess.graph)
      # writer.close()

if __name__ == "__main__":
  svm_model_fn()
