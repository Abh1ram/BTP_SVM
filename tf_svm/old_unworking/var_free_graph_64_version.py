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

C_SVM = tf.constant(5., dtype=tf.float64)
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
    dtype=tf.float64)
  # More parallelism
  supp_vec_x = tf.concat([all_vars.marg_vec_x, all_vars.err_vec_x], 0)
  supp_vec_y = tf.concat([all_vars.marg_vec_y, all_vars.err_vec_y], 0)
  alpha_supp_ = tf.concat([all_vars.alpha_marg, alpha_err_], 0)
  # Calc f(x) = SUM (alpha_j * y_j * kernel)
  kernel_vals = tf.map_fn(lambda x: params["kernel"](x, all_vars.x_c),
    supp_vec_x, dtype=tf.float64,)
  temp = supp_vec_y*kernel_vals
  return tf.tensordot(alpha_supp_, temp, [0,0]) + all_vars.b

# g(x,y) = f(x)*y - 1
def calc_g(all_vars, params):
  return calc_f(all_vars, params) * all_vars.y_c - 1.

def free_add_to_marg(kernel_fn, alpha_c, x_c, y_c, all_vars):
  # print_op = tf.Print(tf.constant(1.), [], "Inside free add")
  # with tf.control_dependencies([print_op]):
  Q_s_ = all_vars.Q_s
  marg_vec_x_ = all_vars.marg_vec_x
  marg_vec_y_ = all_vars.marg_vec_y
  alpha_marg_ = all_vars.alpha_marg
  # Update Q_s by adding new row and column in the end
  q_cc = y_c*y_c*kernel_fn(x_c, x_c)
  Q_marg_c = get_Q_vec(kernel_fn, x_c, y_c, marg_vec_x_, marg_vec_y_)
  Q_marg_c = tf.concat([tf.reshape(y_c, [1,]), Q_marg_c], 0)
  # Concat new row
  Q_s_ = tf.concat([Q_s_, tf.reshape(Q_marg_c, [1, -1])], 0)
  # Concat new column after adding q_cc
  Q_marg_c = tf.concat([Q_marg_c, tf.reshape(q_cc, [1,])], 0)
  Q_s_ = tf.concat([Q_s_, tf.reshape(Q_marg_c, [-1, 1])], 1)

  # print_op = tf.Print(tf.constant(1), [x_c, y_c, all_vars.Q_s, Q_s_],
  #  "BEFORE and AFTER: ")
  # with tf.control_dependencies([print_op]):
  # add x_c, y_c and alpha to marg_vec_x marg_vc_y and alpha_marg
  marg_vec_x_ = tf.concat([marg_vec_x_, tf.reshape(x_c, [1, -1])], 0)
  marg_vec_y_ = tf.concat([marg_vec_y_, tf.reshape(y_c, [1,])], 0)
  alpha_marg_ = tf.concat([alpha_marg_, 
    tf.reshape(alpha_c, [1,])], 0)

  return all_vars._replace(Q_s=Q_s_, marg_vec_x=marg_vec_x_,
    marg_vec_y=marg_vec_y_, alpha_marg=alpha_marg_)

def add_to_marg(kernel_fn, alpha_c, x_c, y_c, beta_c, gamma_c, all_vars):
  R_ = all_vars.R
  marg_vec_x_ = all_vars.marg_vec_x
  marg_vec_y_ = all_vars.marg_vec_y
  alpha_marg_ = all_vars.alpha_marg

  beta_c = tf.cond(
    # Check of beta_c is scalar, then recalc
    tf.equal(tf.shape(tf.shape(beta_c))[0], 0),
    lambda: get_beta(kernel_fn, x_c, y_c, all_vars),
    lambda: beta_c)

  beta_c = tf.concat([beta_c, tf.constant([1.], dtype=tf.float64)], 0)
  # calculate beta_c' * beta_c
  beta_mat = tf.matmul(
    tf.reshape(beta_c, [-1, 1]), tf.reshape(beta_c, [1, -1]))
  beta_mat = tf.constant(1., dtype=tf.float64)/gamma_c * beta_mat
  # Pad R and add beta_mat to it
  # One after each dim
  print_op = tf.Print(tf.constant(1.), [beta_c, gamma_c, x_c, R_,],
    "\nAdding to marg: ")
  with tf.control_dependencies([print_op]):
    pad_R = tf.constant([[0, 1], [0, 1]])
    R_ = tf.pad(R_, pad_R, "CONSTANT")
    R_ = R_ + beta_mat
    # add x_c, y_c and alpha to marg_vec_x marg_vc_y and alpha_marg
    marg_vec_x_ = tf.concat([marg_vec_x_, tf.reshape(x_c, [1, -1])], 0)
    marg_vec_y_ = tf.concat([marg_vec_y_, tf.reshape(y_c, [1,])], 0)
    alpha_marg_ = tf.concat([alpha_marg_, tf.reshape(alpha_c, [1,])], 0)

    return all_vars._replace(R=R_, marg_vec_x=marg_vec_x_,
    marg_vec_y=marg_vec_y_, alpha_marg=alpha_marg_)

def add_to_err(g_c, x_c, y_c, all_vars):
  err_vec_x_ = tf.concat([all_vars.err_vec_x, tf.reshape(x_c, [1, -1])], 0)
  err_vec_y_ = tf.concat([all_vars.err_vec_y, tf.reshape(y_c, [1,])], 0)
  g_err_ = tf.concat([all_vars.g_err, tf.reshape(g_c, [1,])], 0)

  return all_vars._replace(err_vec_x=err_vec_x_, err_vec_y=err_vec_y_,
    g_err=g_err_)

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
  # print_op = tf.Print(tf.constant(0), [all_vars.alpha_c, all_vars.g_c],
    # "Checking while loop termn_condn")
  # with tf.control_dependencies([print_op]):
  return tf.logical_or(
    tf.equal(all_vars.g_c, tf.constant(0., dtype=tf.float64)),
    tf.equal(all_vars.alpha_c, C_svm))

# This can be cached - not for Tensorflow though
def get_Q_vec(kernel_fn, x_c, y_c, vec_x, vec_y):
  # Calculate Q = y1*y2*K(x1*x2)
  q_tens = tf.map_fn(lambda x: kernel_fn(x_c, x), vec_x)
  q_tens = q_tens * y_c
  q_tens = q_tens * vec_y
  return q_tens

def get_beta(kernel_fn, x_c, y_c, all_vars):
  q_tens = get_Q_vec(kernel_fn, x_c, y_c, all_vars.marg_vec_x,
    all_vars.marg_vec_y)
  # Concat y_c in the starting
  q_tens = tf.concat([tf.reshape(y_c, [1,]), q_tens], 0)
  # Reshape q_tens for matrix mult
  q_tens_reshaped = tf.reshape(q_tens, [-1, 1])
  beta = -1 * tf.matmul(all_vars.R, q_tens_reshaped)
  # return reshaped beta
  return tf.reshape(beta, [-1,])

def get_gamma(kernel_fn, vec_x, vec_y, beta, all_vars):
  Q_i_c = get_Q_vec(kernel_fn, all_vars.x_c, all_vars.y_c, vec_x,
    vec_y)
  # Gamma_i = Q_i_c + SUM(get_Q_vec(x_i, y_i, marg) Beta) + Beta[0]*y_i
  elems = (vec_x, vec_y)
  gamma_tens = tf.map_fn(
    lambda x: tf.tensordot(
      get_Q_vec(kernel_fn, x[0], x[1], all_vars.marg_vec_x, all_vars.marg_vec_y),
      beta[1:], [0,0]),
    elems, dtype=tf.float64)
  gamma_tens = gamma_tens + (beta[0] * vec_y)
  return gamma_tens + Q_i_c

def get_trans_alpha_marg(C_svm, alpha_marg, beta_marg):
  elems = (alpha_marg, beta_marg)

  return tf.map_fn(
    lambda x: 
      tf.cond(
        tf.less(x[1], 0),
      # Alpha goes to zero
        lambda: -x[0]/x[1],
        lambda:
          tf.cond(
            tf.greater(x[1], 0),
            lambda: (C_svm - x[0])/x[1],
            lambda: tf.constant(float("Inf"), dtype=tf.float64)
          )
        ), 
      elems, dtype=tf.float64
    )

# Tensor should have non-zero shape 
def check_min(tens, val):
  min_indx = tf.argmin(tens)
  return tf.cond(tf.equal(tens[min_indx], val),
    lambda: tf.cast(min_indx, tf.int32),
    lambda: tf.shape(tens)[0] + 1)

def update_val(min_alpha, beta, gamma_err, gamma_rem, gamma_c, all_vars):
  # CALCULATING NEW VALUES
  # Calculate new b
  b_ = all_vars.b + beta[0]*min_alpha
  all_vars = all_vars._replace(b=b_)
  # Calculate new values of alpha for marg vec
  elems = (all_vars.alpha_marg, beta[1:])
  alpha_marg_ = tf.map_fn(lambda x: x[0] + min_alpha*x[1],
    elems, dtype=tf.float64)
  # Calculate new value of alpha_c
  alpha_c_ = all_vars.alpha_c + min_alpha
  all_vars = all_vars._replace(alpha_marg=alpha_marg_, alpha_c=alpha_c_)

  # Calculate new values of g_all
  elems = (all_vars.g_err, gamma_err)
  g_err_ = tf.map_fn(lambda x: x[0] + x[1]*min_alpha, elems,
    dtype=tf.float64)
  elems = (all_vars.g_rem, gamma_rem)
  g_rem_ = tf.map_fn(lambda x: x[0] + x[1]*min_alpha, elems,
    dtype=tf.float64)
  g_c_ = all_vars.g_c + gamma_c[0]*min_alpha
  return all_vars._replace(g_rem=g_rem_, g_err=g_err_, g_c=g_c_)

def rem_from_marg(kernel_fn, min_marg_indx, all_vars):
  R_ = all_vars.R
  marg_vec_x_ = all_vars.marg_vec_x
  marg_vec_y_ = all_vars.marg_vec_y
  alpha_marg_ = all_vars.alpha_marg

  k = min_marg_indx + 1
  # R_ij = R_ij - 1/R_kk*R_ik*R_kj
  row_k = tf.reshape(R_[k, :], [1, -1])
  col_k = tf.reshape(R_[:, k], [-1, 1])
  kR = 1/R_[k,k] * tf.matmul(col_k, row_k)
  # subtract kR from R
  R_ = R_ - kR
  # drop kth row and column
  R_ = tf.concat([R_[:k, :], R_[k+1:, :]], 0)
  R_ = tf.concat([R_[:, :k], R_[:, k+1:]], 1)
  # drop min_mar_indx from marg_vec_x and y and alpha_marg
  marg_vec_x_ = tf.concat([marg_vec_x_[:k-1], marg_vec_x_[k:]], 0)
  marg_vec_y_ = tf.concat([marg_vec_y_[:k-1], marg_vec_y_[k:]], 0)
  alpha_marg_ = tf.concat([alpha_marg_[:k-1], alpha_marg_[k:]], 0)

  return all_vars._replace(R=R_, marg_vec_x=marg_vec_x_,
    marg_vec_y=marg_vec_y_, alpha_marg=alpha_marg_)

def rem_from_rem(indx, all_vars):
  rem_vec_x_ = all_vars.rem_vec_x
  rem_vec_y_ = all_vars.rem_vec_y
  g_rem_ = all_vars.g_rem

  rem_vec_x_ = tf.concat([rem_vec_x_[:indx], rem_vec_x_[indx+1:]], 0)
  rem_vec_y_ = tf.concat([rem_vec_y_[:indx], rem_vec_y_[indx+1:]], 0)
  g_rem_ = tf.concat([g_rem_[:indx], g_rem_[indx+1:]], 0)
  return all_vars._replace(rem_vec_x=rem_vec_x_, rem_vec_y=rem_vec_y_,
    g_rem=g_rem_)

def rem_from_err(indx, all_vars):
  err_vec_x_ = all_vars.err_vec_x
  err_vec_y_ = all_vars.err_vec_y
  g_err_ = all_vars.g_err

  err_vec_x_ = tf.concat([err_vec_x_[:indx], err_vec_x_[indx+1:]], 0)
  err_vec_y_ = tf.concat([err_vec_y_[:indx], err_vec_y_[indx+1:]], 0)
  g_err_ = tf.concat([g_err_[:indx], g_err_[indx+1:]], 0)
  return all_vars._replace(err_vec_x=err_vec_x_, err_vec_y=err_vec_y_,
    g_err=g_err_)

def handle_rem(params, min_rem_indx, gamma_rem, all_vars):
  # Add the vector to marg support
  alpha_ = tf.constant(0.,dtype=tf.float64)
  x_ = all_vars.rem_vec_x[min_rem_indx]
  y_ = all_vars.rem_vec_y[min_rem_indx]
  
  all_vars = add_to_marg(params["kernel"], alpha_, x_, y_, tf.constant(-1.,
    dtype=tf.float64), gamma_rem[min_rem_indx], all_vars)
  return rem_from_rem(min_rem_indx, all_vars)

def handle_err(params, min_err_indx, gamma_err, all_vars):
  # Add the vector to marg support
  alpha_ = params["C"] 
  x_ = all_vars.err_vec_x[min_err_indx]
  y_ = all_vars.err_vec_y[min_err_indx]
  
  all_vars = add_to_marg(params["kernel"], alpha_, x_, y_, tf.constant(-1.,
    dtype=tf.float64),
    gamma_err[min_err_indx], all_vars)
  return rem_from_err(min_err_indx, all_vars)

def handle_marg(params, min_marg_indx, beta, all_vars):
  # add the vector to error or remain based on beta value
  g_ = tf.constant(0., dtype=tf.float64)
  x_ = all_vars.marg_vec_x[min_marg_indx]
  y_ = all_vars.marg_vec_y[min_marg_indx]
  all_vars = tf.cond(tf.less(beta[min_marg_indx+1], 0),
    lambda: add_to_rem(params["eps"], g_, x_, y_, all_vars),
    lambda: add_to_err(g_, x_, y_, all_vars))

  return rem_from_marg(params["kernel"], min_marg_indx, all_vars)

def handle_empty_rem(params, min_rem_indx, all_vars):
  # Add the vector to marg support
  alpha_ = tf.constant(0., dtype=tf.float64)
  x_ = all_vars.rem_vec_x[min_rem_indx]
  y_ = all_vars.rem_vec_y[min_rem_indx]
  
  all_vars = free_add_to_marg(params["kernel"], alpha_, x_, y_, all_vars)
  return rem_from_rem(min_rem_indx, all_vars)

def handle_empty_err(params, min_err_indx, all_vars):
  # Add the vector to marg support
  alpha_ = params["C"] 
  x_ = all_vars.err_vec_x[min_err_indx]
  y_ = all_vars.err_vec_y[min_err_indx]
  
  all_vars = free_add_to_marg(params["kernel"], alpha_, x_, y_, all_vars)
  return rem_from_err(min_err_indx, all_vars)

def handle_empty_marg(params, all_vars):
  print_op = tf.Print(tf.constant(1), [], "Inside handling empty marg")
  # Set Q_s as [[0]]
  with tf.control_dependencies([print_op]):
    all_vars = all_vars._replace(Q_s=tf.constant([[0.]], dtype=tf.float64))
  # Calculate min b for transition
  g_c_ = all_vars.g_c
  x_c_ = all_vars.x_c
  y_c_ = all_vars.y_c

  trans_c = -1*g_c_
  # calc trans b for err and rem
  # For err_vec, err_vec_y should have same sign as y_c_
  elems = (all_vars.err_vec_y, all_vars.g_err)
  trans_err = tf.map_fn(
    lambda x: tf.cond(
      tf.greater(x[0] * y_c_, 0),
      lambda: -1*x[1],
      lambda: tf.constant(float("Inf"), dtype=tf.float64)),
    elems, dtype=tf.float64)
  # For rem_vec, rem_vec_y should have opposite sign as y_c_
  elems = (all_vars.rem_vec_y, all_vars.g_rem)
  trans_rem = tf.map_fn(
    lambda x: tf.cond(
      tf.less(x[0] * y_c_, 0),
      lambda: x[1],
      lambda: tf.constant(float("Inf"), dtype=tf.float64)),
    elems, dtype=tf.float64)

  # Find the min transition_val for b
  all_trans = tf.concat(
    [trans_err, trans_rem, 
    tf.reshape(trans_c, [1,])], 0)
  abs_min_del_b = tf.reduce_min(all_trans)
  # change the sign of min_del_b to the sign of y_c_ as
  # this is the dirn of update
  min_del_b = tf.cast(y_c_ * abs_min_del_b, dtype=tf.float64)

  # update values
  b_ = all_vars.b + min_del_b
  g_c_ = g_c_ + (min_del_b * y_c_)
  # update g_err
  g_err_ = all_vars.g_err + (all_vars.err_vec_y * min_del_b)
  # update g_rem
  g_rem_ = all_vars.g_rem + (all_vars.rem_vec_y * min_del_b)

  # print_op = tf.Print(tf.constant(1.), [all_vars.g_c, all_vars.Q_s, min_del_b],
  #   "Line 351")
  # with tf.control_dependencies([print_op]):
  all_vars = all_vars._replace(b=b_, g_c=g_c_, g_err=g_err_, g_rem=g_rem_)

  # Transition vectors if g is zero and dirn of update is same as candidate
  # Check if min is in err sup vectors
  min_err_indx = tf.cond(tf.equal(tf.shape(trans_err)[0], 0),
    lambda: tf.constant(1),
    lambda: check_min(trans_err, abs_min_del_b),
    )
  all_vars = tf.cond(
    tf.less(min_err_indx, tf.shape(trans_err)[0]),
    lambda: handle_empty_err(params, min_err_indx, all_vars),
    lambda: all_vars)
  # Check if min is in rem vectors
  min_rem_indx = tf.cond(tf.equal(tf.shape(trans_rem)[0], 0),
    lambda: tf.constant(1),
    lambda: check_min(trans_rem, abs_min_del_b),
    )
  all_vars = tf.cond(
    tf.less(min_rem_indx, tf.shape(trans_rem)[0]),
    lambda: handle_empty_rem(params, min_rem_indx, all_vars),
    lambda: all_vars)
  # Check if min is in trans_c
  all_vars = tf.cond(
    tf.equal(trans_c, abs_min_del_b),
    lambda: free_add_to_marg(params["kernel"], all_vars.alpha_c,all_vars.x_c,
      all_vars.y_c, all_vars),
    lambda: all_vars)
  # update R as inverse of Q
  R_ = tf.matrix_inverse(all_vars.Q_s)
  return all_vars._replace(R=R_)

def mini_iter(params, all_vars):
  print_op = tf.Print(tf.constant(1.), [], "Inside mini iter")
  # Calculate beta
  with tf.control_dependencies([print_op]):
    beta = get_beta(params["kernel"], all_vars.x_c, all_vars.y_c, all_vars)
  # Calculate gamma
  gamma_err = get_gamma(params["kernel"], all_vars.err_vec_x, all_vars.err_vec_y,
    beta, all_vars)
  gamma_rem = get_gamma(params["kernel"], all_vars.rem_vec_x, all_vars.rem_vec_y,
    beta, all_vars)
  gamma_c = get_gamma(params["kernel"], tf.reshape(all_vars.x_c, [1, -1]),
    tf.reshape(all_vars.y_c, [1,]),beta, all_vars)

  # BOOK_KEEPING
  # For margin support vectors, if Beta_s > 0, alpha_c can go to C or 
  # if Beta_s < 0, alpha_s can go to 0, causing a transition in state
  trans_marg = get_trans_alpha_marg(params["C"],
    all_vars.alpha_marg, beta[1:])
  # For error support vectors, if Gamma_i > 0, then g_i increases to 0
  # and causes state change
  elems = (all_vars.g_err, gamma_err)
  trans_err = tf.map_fn(lambda x: tf.cond(
    tf.greater(x[1], 0),
    lambda: -1*x[0]/x[1],
    lambda: tf.constant(float("Inf"), dtype=tf.float64)),
  elems, dtype=tf.float64)
  # For the remaining vectors, if Gamma_i < 0, then g_i decreases to 0
  elems = (all_vars.g_rem, gamma_rem)
  trans_rem = tf.map_fn(lambda x: tf.cond(
    tf.less(x[1], 0),
    lambda: -1*x[0]/x[1],
    lambda: tf.constant(float("Inf"), dtype=tf.float64)),
  elems, dtype=tf.float64)
  # For candidate vector
  candidate_trans = tf.minimum(
      tf.cond(gamma_c[0] > 0,
        lambda: -1*all_vars.g_c/gamma_c[0],
        lambda: tf.constant(float("Inf"), dtype=tf.float64)),
      params["C"] - tf.cast(all_vars.alpha_c, tf.float64))

  # Find the minimum transition alpha
  all_trans = tf.concat(
    [trans_marg, trans_err, trans_rem, 
    tf.reshape(candidate_trans, [1,])], 0)
  min_alpha = tf.reduce_min(all_trans)

  print_op = tf.Print(tf.constant(1), [trans_marg, trans_err, trans_rem,
    candidate_trans, min_alpha], "TRANS: ", summarize=100)
  print_op2 = tf.Print(tf.constant(2), [all_vars.marg_vec_x, all_vars.marg_vec_y,
    all_vars.err_vec_x, all_vars.err_vec_y], "VECS: ", summarize=100)
  print_op3 = tf.Print(tf.constant(2), [beta, gamma_err], "BETA: ", summarize=100)
  with tf.control_dependencies([print_op, print_op2, print_op3]):
    # Update values:
    all_vars = update_val(min_alpha, beta, gamma_err, gamma_rem, gamma_c,
    all_vars)

  # TRANSITION AND MOVE VECTORS
  # Check if min is in marg sup vectors
  # Run get index only if tnsor sz is non-zero
  min_marg_indx = tf.cond(tf.equal(tf.shape(trans_marg)[0], 0),
    lambda: tf.constant(1),
    lambda: check_min(trans_marg, min_alpha),
    )
  all_vars = tf.cond(
    tf.less(min_marg_indx, tf.shape(trans_marg)[0]),
    lambda: handle_marg(params, min_marg_indx, beta, all_vars),
    lambda: all_vars,
    name="Transition_marg")

  # Check if min is in err sup vectors
  min_err_indx = tf.cond(tf.equal(tf.shape(trans_err)[0], 0),
    lambda: tf.constant(1),
    lambda: check_min(trans_err, min_alpha),
    )
  all_vars = tf.cond(
    tf.less(min_err_indx, tf.shape(trans_err)[0]),
    lambda: handle_err(params, min_err_indx, gamma_err, all_vars),
    lambda: all_vars)

  # Check if min is in rem sup vectors
  min_rem_indx = tf.cond(tf.equal(tf.shape(trans_rem)[0], 0),
    lambda: tf.constant(1),
    lambda: check_min(trans_rem, min_alpha),
    )
  all_vars = tf.cond(
    tf.less(min_rem_indx, tf.shape(trans_rem)[0]),
    lambda: handle_rem(params, min_rem_indx, gamma_rem, all_vars),
    lambda: all_vars)

  # Need to add the candidate vector here as I need g and gamma values
  return tf.cond(
    termn_condn(params["C"], all_vars),
    lambda: tf.cond(
      tf.equal(all_vars.g_c, 0.),
      lambda: add_to_marg(params["kernel"], all_vars.alpha_c,
        all_vars.x_c, all_vars.y_c, beta, gamma_c[0], all_vars),
      lambda: add_to_err(all_vars.g_c, all_vars.x_c, all_vars.y_c, all_vars)),
    lambda: all_vars)

def loop_body(params, i, all_vars):
  printer = tf.Print(tf.constant(0), [i], "\nInside body: ")
  # Handle case of matrix containing inf
  with tf.control_dependencies([printer]):
    all_vars = tf.cond(
      tf.equal(tf.shape(all_vars.marg_vec_x)[0], 0),
      lambda: handle_empty_marg(params, all_vars),
      lambda: all_vars)
  # If x_c was already added to marg, there is nothing to do
  all_vars = tf.cond(
    tf.equal(all_vars.g_c, 0),
    lambda: all_vars,
    lambda: mini_iter(params, all_vars))

  # printer = tf.Print(tf.constant(1.), [tf.shape(all_vars.marg_vec_x)], "SHAPE: ")
  # with tf.control_dependencies([printer]):
  return (i+1, all_vars)

def fit_point(params, all_vars):
  iter_ct = tf.constant(0., dtype=tf.float64)
  # Create shape invariants as while loop enforces strict checks on shape
  shape_invar = params["namedtuple"](
    n=all_vars.n.get_shape(), b=all_vars.b.get_shape(),
    marg_vec_x=tf.TensorShape([None, params["shape"]]),
    marg_vec_y=tf.TensorShape([None,]),
    alpha_marg=tf.TensorShape([None,]),
    err_vec_x=tf.TensorShape([None, params["shape"]]),
    err_vec_y=tf.TensorShape([None,]), g_err=tf.TensorShape([None,]),
    rem_vec_x=tf.TensorShape([None, params["shape"]]),
    rem_vec_y=tf.TensorShape([None,]), g_rem=tf.TensorShape([None,]),
    x_c=all_vars.x_c.get_shape(), y_c=all_vars.y_c.get_shape(),
    alpha_c=all_vars.alpha_c.get_shape(), g_c=all_vars.g_c.get_shape(),
    R=tf.TensorShape(None), Q_s=tf.TensorShape(None))
    # R=all_vars.R.get_shape(), Q_s=all_vars.Q_s.get_shape())
  
  fin_i, fin_vars = tf.while_loop(
    cond=lambda i, p: tf.logical_not(termn_condn(params["C"], p)), 
    # tf.logical_or(
    #   tf.logical_not(termn_condn(params["C"], p)),
    #   tf.less(i, 1)),
    body=lambda i, p: loop_body(params, i, p), 
    loop_vars=(iter_ct, all_vars),
    parallel_iterations=1,
    back_prop=False,
    name="FIT_WHILE",
    shape_invariants=(iter_ct.get_shape(), shape_invar)
    )
  return fin_vars

def svm_train(x_, y_, params, all_vars):
  n_ = all_vars.n + 1
  # add new point
  all_vars = all_vars._replace(x_c=x_, y_c=y_, n=n_,
    alpha_c=tf.constant(0., dtype=tf.float64))
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
def create_all_vars(params, scope):
  AllVars = namedtuple("AllVars", ["n", "b", "marg_vec_x", "marg_vec_y",
    "alpha_marg", "err_vec_x", "err_vec_y", "g_err", "rem_vec_x", "rem_vec_y",
    "g_rem", "x_c", "y_c", "alpha_c", "g_c", "R", "Q_s"])

  model_params["namedtuple"] = AllVars
  x_shape = model_params["shape"]
  # TODO: check if writing simple shape is enough instead of unnecessary reshape
  all_vars = AllVars(
    n=tf.reshape(tf.get_variable("n", dtype=tf.int32).read_value(),
      []),
    b=tf.reshape(tf.get_variable("b", dtype=tf.float64).read_value(), []),
    marg_vec_x=tf.reshape(tf.get_variable("marg_vec_x",
      dtype=tf.float64).read_value(), [-1, x_shape]),
    marg_vec_y=tf.reshape(tf.get_variable("marg_vec_y",
      dtype=tf.float64).read_value(), [-1,]),
    alpha_marg=tf.reshape(tf.get_variable("alpha_marg",
      dtype=tf.float64).read_value(), [-1,]),
    err_vec_x=tf.reshape(tf.get_variable("err_vec_x",
      dtype=tf.float64).read_value(), [-1, x_shape]),
    err_vec_y=tf.reshape(tf.get_variable("err_vec_y",
      dtype=tf.float64).read_value(), [-1,]),
    g_err=tf.reshape(tf.get_variable("g_err",
      dtype=tf.float64).read_value(), [-1,]),
    rem_vec_x=tf.reshape(tf.get_variable("rem_vec_x",
      dtype=tf.float64).read_value(), [-1, x_shape]),
    rem_vec_y=tf.reshape(tf.get_variable("rem_vec_y", 
      dtype=tf.float64).read_value(),[-1,]),
    g_rem=tf.reshape(tf.get_variable("g_rem",
      dtype=tf.float64).read_value(), [-1,]),
    x_c=tf.constant([], dtype=tf.float64), y_c=tf.constant(0.),
    alpha_c=tf.constant(0., dtype=tf.float64),
    g_c=tf.constant(0., dtype=tf.float64),
    R=tf.get_variable("R", dtype=tf.float64).read_value(),
    Q_s=tf.get_variable("Q_s", dtype=tf.float64).read_value())

  return all_vars

def create_svm_variables(x_shape):
  with tf.variable_scope("svm_model") as scope:
    # Variable creation
    # count of the number of data points seen
    n_ = tf.get_variable("n", initializer=tf.constant(0), trainable=False,)
    # offset
    b_ = tf.get_variable("b", initializer=tf.constant(0.,dtype=tf.float64),
     trainable=False,)

    # Variable tensor - representing the Margin Support vector indices
    marg_vec_x_ = tf.get_variable("marg_vec_x", shape=[0, x_shape],
      validate_shape=False, dtype=tf.float64)
    marg_vec_y_ = tf.get_variable("marg_vec_y", initializer=tf.constant([],
      dtype=tf.float64))

    # Variable tensor - representing the Error Support vector indices
    err_vec_x_ = tf.get_variable("err_vec_x", shape=[0, x_shape],
      validate_shape=False, dtype=tf.float64)
    err_vec_y_ = tf.get_variable("err_vec_y", initializer=tf.constant([],
      dtype=tf.float64))

    # Variable tensor - representing the Remaining vector indices
    rem_vec_x_ = tf.get_variable("rem_vec_x", shape=[0, x_shape],
      validate_shape=False, dtype=tf.float64)
    rem_vec_y_ = tf.get_variable("rem_vec_y", initializer=tf.constant([],
      dtype=tf.float64))

    # Variable for alpha of margin vectors
    alpha_marg_ = tf.get_variable("alpha_marg", initializer=tf.constant([],
      dtype=tf.float64), validate_shape=False,)
    # Variable denoting g of error support vectors
    g_err_ = tf.get_variable("g_err", initializer=tf.constant([],
      dtype=tf.float64), validate_shape=False, dtype=tf.float64)
    # Variable denoting g of remaining vectors
    g_rem_ = tf.get_variable("g_rem", initializer=tf.constant([],
      dtype=tf.float64), validate_shape=False, dtype=tf.float64)

    # Variable for the inverse Jacobian matrix R - initially Inf
    R_ = tf.get_variable("R", initializer=tf.constant([[INF]],
      dtype=tf.float64), validate_shape=False,)
    # Jacobian
    Q_s = tf.get_variable("Q_s",
      initializer=tf.constant([[0.]], dtype=tf.float64),
      dtype=tf.float64)
    return scope

def update_vars(scope, all_vars):
  with tf.variable_scope(scope, reuse=True):
    op_list = []
    for field in set(all_vars._fields) - set(["x_c", "y_c", "alpha_c", "g_c", "n"]):
      op_list.append(tf.assign(tf.get_variable(field,dtype=tf.float64),
        getattr(all_vars, field), validate_shape=False))
    # separate operation for n due to int32
    op_list.append(tf.assign(tf.get_variable("n", dtype=tf.int32), all_vars.n))
    with tf.control_dependencies(op_list):
      return tf.constant(1.)

# svm_model_fn
def svm_model_fn():
  # make these command line args
  train_file = "data_1.csv"
  x_train, y_train = extract_data(train_file, "csv")
  n = len(x_train)
  # x_train = x_train[1:]
  # y_train = y_train[1:]
  model_params["shape"] = x_train.shape[1]

  # Placeholders for inputs
  x_ = tf.placeholder(tf.float64, shape=(x_train.shape[1]))
  y_ = tf.placeholder(tf.float64, shape=())

  # Create namedtuple
  scope = create_svm_variables(x_shape=x_train.shape[1])
  with tf.variable_scope(scope, reuse=True):
    all_vars = create_all_vars(model_params, scope)
  
  # TODO : Create variables to store model
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    all_vars_upd = svm_train(x_, y_, model_params, all_vars)
    tr = update_vars(scope, all_vars_upd)
    for i in range(14):
      print(i)
      input()
      _, req = sess.run([tr, all_vars_upd],
        feed_dict={x_ : x_train[i], y_ : y_train[i]})
      print(req)
        # y_all_ = tf.get_variable("y_all")
        # print(sess.run(y_all_))
    writer = tf.summary.FileWriter(path_, sess.graph)
    writer.close()
    print("MARG: ", len(req.marg_vec_x), "ERR: ", len(req.err_vec_x),
      "REM: ", len(req.rem_vec_y))

if __name__ == "__main__":
  svm_model_fn()
