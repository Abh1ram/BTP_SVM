# Errors:
# 1. if map_fn elements size is zero, then indexing using it's elements
# fails
# 2. Variables shape doesn't show change even after reassigning
# and make shape at creation time as validate_shape=False
# Change marg_vec_ to marg_vec_.read_value() in add marg, rem marg, add err, rem err ...
# 3. While loop frame error, if variables inside and outside while loop are used in 
# same operation
# 4. Asychronous pains
# TF logical_and runs second terms even if first term fails
# Asynchronous prints
# 5. In while loop variables are not read after first run, i.e. other
# than loop vars no other ones run - https://github.com/tensorflow/tensorflow/issues/13616
# 6. tf cond expects same nested structure?? - How come this problem didn;t come up with
# other tf cond
# 7. https://github.com/tensorflow/tensor2tensor/issues/159

import tensorflow as tf

from collections import namedtuple

from data_loader import extract_data


C_SVM = 5.
RESERVE_THRESHOLD = -1

INF = 10**12 + 0.

import os
path_ = os.path.join("log_graph")

# Simple dot product - returns a rank 0 tensor/scalar
def simple_kernel(x1, x2):
  return tf.tensordot(x1, x2, [0,0])
  # val2 = tf.Print(val, [tf.constant("-----------Line 20"), val,
  #   tf.shape(val)])
  # return val2

model_params = {
  "C" : C_SVM,
  "eps" : RESERVE_THRESHOLD,
  "kernel" : simple_kernel,
  }


# Pipeline with fixed batch size of 1
def custom_input_fn(
  features, # Array like 
  labels,
  ): 
  assert features.shape[0] == labels.shape[0]
  feature_dict = {"x" : features}
  dataset = tf.data.Dataset.from_tensor_slices(
    (feature_dict, labels))
  return dataset.make_one_shot_iterator().get_next()

def calc_f(params, x_):
  x_all_ = tf.get_variable("x_all").read_value()
  y_all_ = tf.get_variable("y_all").read_value()
  b_ = tf.get_variable("b").read_value()
  marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32).read_value()
  err_vec_ = tf.get_variable("err_vec", dtype=tf.int32).read_value()
  alpha_marg_ = tf.get_variable("alpha_marg").read_value()

  alpha_err_ = tf.map_fn(lambda x: params["C"], err_vec_, dtype=tf.float32)
  # f(x_) = SUM(alpha[j] * y[j] * K(x_all[j] * x_)) + b
  
  # CHECK: Should I concat both the vectors to get all supp vectors
  # and then do the mulitplication (more parallelism) vs do them individually
  # and then join them (more distributed)??

  # More parallelism
  supp_vec_ = tf.concat([marg_vec_, err_vec_], 0)
  alpha_supp_ = tf.concat([alpha_marg_, alpha_err_], 0)

  # supp_vec_ = tf.Print(supp_vec_1, ["--------SUP VEC FOR f: ",supp_vec_1])
  temp = tf.cond(tf.equal(0, tf.shape(supp_vec_)[0]),
    lambda: tf.constant(0.),
    lambda: tf.map_fn(
      lambda i: y_all_[i] * params["kernel"](x_all_[i], x_),
      supp_vec_, dtype=tf.float32, name="58"))

  ans = tf.cond(tf.equal(0, tf.shape(supp_vec_)[0]),
    lambda: temp + b_,
    lambda: tf.tensordot(alpha_supp_, temp, [0,0], name="62") + b_)

  # ans = tf.Print(ans, [ans, temp, b_])
  return ans

# x_ - tensor
# y_ - tensor of rank 0/scalar
def calc_g(params, x_, y_):
  # g(x, y) = f(x)*y - 1 
  return (calc_f(params, x_) * y_) - 1.

def rem_from_err(loc_index, loop_var):
  err_vec_ = loop_var.err_vec
  g_err_ = loop_var.g_err

  err_vec_ = tf.concat([err_vec_[:loc_index], err_vec_[loc_index+1:]], 0)
  # REmove g val of the vector
  g_err_ = tf.concat([g_err_[:loc_index], g_err_[loc_index+1:]], 0)
  # Force the above two operations
  return loop_var._replace(err_vec=err_vec_, g_err=g_err_)

def rem_from_rem(loc_index, loop_var):
  rem_vec_ = loop_var.rem_vec
  g_rem_ = loop_var.g_rem

  rem_vec_ = tf.concat([rem_vec_[:loc_index], rem_vec_[loc_index+1:]], 0)
  # REmove g val of the vector
  g_rem_ = tf.concat([g_rem_[:loc_index], g_rem_[loc_index+1:]], 0)
  # Force the above two operations
  return loop_var._replace(rem_vec=rem_vec_, g_rem=g_rem_)

def add_to_err(g_c, glob_index, loop_var):
  err_vec_ = loop_var.err_vec
  g_err_ = loop_var.g_err
  # Add new index to rem_vec
  err_vec_ = tf.concat([err_vec_, tf.reshape(glob_index, [1])], 0)

  # Add g_c to g_rem
  g_err_ = tf.concat([g_err_, tf.reshape(g_c, [1])], 0)

  loop_var = loop_var._replace(err_vec=err_vec_, g_err=g_err_)
  return loop_var

# Returns updated loop_var
# TODO: Add the reserve threshold check
def add_to_rem(eps, g_c, glob_index, loop_var=None):
  if loop_var is not None:
    rem_vec_ = loop_var.rem_vec
    g_rem_ = loop_var.g_rem
    # Add new index to rem_vec
    rem_vec_ = tf.concat([rem_vec_, tf.reshape(glob_index, [1])], 0)
    # Add g_c to g_rem
    g_rem_ = tf.concat([g_rem_, tf.reshape(g_c, [1])], 0)
    loop_var = loop_var._replace(rem_vec=rem_vec_, g_rem=g_rem_)
    return loop_var

  else:
    rem_vec_ = tf.get_variable("rem_vec", dtype=tf.int32)
    g_rem_ = tf.get_variable("g_rem")
    op1 = tf.assign(rem_vec_, tf.concat(
      [rem_vec_.read_value(), tf.reshape(glob_index, [1])], 0))
    # Add g_c to g_rem
    op2 = tf.assign(g_rem_, tf.concat(
      [g_rem_.read_value(), tf.reshape(g_c, [1])], 0))
    with tf.control_dependencies([op1, op2]):
      return tf.constant(1)
  

# Terminate if candidate g_c equals 0 or candidate alpha_c equals C
def termn_condn(C_svm, loop_var):
  print_op = tf.Print(tf.constant(0), [tf.constant("TREM CONDN"),
   loop_var.alpha_c, loop_var.g_c])
  with tf.control_dependencies([print_op]):
    return tf.logical_or(tf.equal(loop_var.g_c, 0.),
      tf.equal(loop_var.alpha_c, C_svm))

# TODO: do matrix inverse when size is one/inf
# Calculates the co-efficient sensitivities
def get_beta(kernel_fn, x_all_, y_all_, glob_index, marg_vec_, R_):
  # marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32).read_value()
  # R_ = tf.get_variable("R").read_value()
  # with tf.control_dependencies(None):
    x_c = x_all_[glob_index]
    y_c = y_all_[glob_index]

    # Calculating Q_(s_i, c) = y_all[s_i]*y_c*
    temp_q = tf.cond(tf.equal(tf.shape(marg_vec_)[0], 0),
      lambda: tf.constant([]),
      lambda: tf.map_fn(
              lambda i: y_c * y_all_[i] * kernel_fn(x_c, x_all_[i]), 
              marg_vec_, dtype=tf.float32, name="BETA_MAP"),
      name="BETA_COND")
    # Add y_c as the first element
    temp_q = tf.concat([tf.reshape(y_c, [1,]), temp_q], 0)
    # reshape temp_q from (n,) to (n,1)/ rank 1 to rank 2 for matmul
    temp_q = tf.reshape(temp_q, [tf.shape(temp_q)[0], 1])
    # temp_q_p = tf.Print(temp_q, [tf.constant("LINE 150, BETA_CALC: "),
    #   tf.shape(x_all_), temp_q, R_])
    # print_op = tf.map_fn(lambda i: tf.Print(i, [i]), R_)
    # with tf.control_dependencies([print_op]):
    beta = -1 * tf.matmul(R_, temp_q)
    return tf.reshape(beta, [tf.shape(beta)[0]])

# Computes SUM(Q_ij*B_j)
def gamma_helper_fn(kernel_fn, beta_, indx, x_all_, y_all_, n, marg_vec_):
  # with tf.variable_scope(scope, reuse=True):

    iter_tensor = tf.range(tf.shape(marg_vec_)[0])
    # iter_tensor = tf.Print(iter_tensor_1, [tf.constant("GAMMA HELPER FN"), 
      # tf.shape(x_all_), y_all_, n, indx, iter_tensor_1, ])
    return tf.cond(
      tf.equal(tf.shape(iter_tensor)[0], 0),
      lambda: tf.constant(0.),
      lambda: tf.reduce_sum(
        tf.map_fn(
          lambda i: ( y_all_[indx] * y_all_[marg_vec_[i]] * beta_[i + 1] *
            kernel_fn(x_all_[indx], x_all_[marg_vec_[i]]) ),
          iter_tensor,
          dtype=tf.float32
          )
        )
      )

# Making this more distributed
# vec is a tensor of indices for which gamma needs to be computed
def get_gamma(kernel_fn, vec, beta_, x_all_, y_all_, n, marg_vec_):
  x_c = x_all_[n-1]
  y_c = y_all_[n-1]
  # vec = tf.Print(vec_1, [tf.constant("---------------Line 174"), beta_,
  #   y_all_, tf.shape(y_all_), tf.shape(x_all_), tf.shape(beta_)])
  return tf.cond(tf.equal(tf.shape(vec)[0], 0),
    lambda: tf.constant([]),
    lambda: tf.map_fn(
      lambda i: (y_all_[i]*y_c*kernel_fn(x_all_[i], x_c) + 
        beta_[0]*y_all_[i] +
        gamma_helper_fn(kernel_fn, beta_, i, x_all_, y_all_, n, marg_vec_)),
      vec,
      dtype=tf.float32)
    )

# TODO: Take beta as arg to prevent recalculation in case of new candidate vector
def add_to_marg_supp(kernel_fn, glob_index, gamma, alpha, x_all_,
  y_all_, loop_var):
  # with tf.variable_scope(scope, reuse=True):
  R_ = loop_var.R
  marg_vec_ = loop_var.marg_vec
  alpha_marg_ = loop_var.alpha_marg

  beta_indx = get_beta(kernel_fn, x_all_, y_all_, glob_index, marg_vec_, R_)
  beta_indx = tf.concat([beta_indx, [1]], 0)
  beta_shp = tf.shape(beta_indx)[0]

  beta_mat = tf.matmul(
    tf.reshape(beta_indx, [beta_shp, 1]),
    tf.reshape(beta_indx, [1, beta_shp]))

  beta_mat = 1/gamma * beta_mat

  # reshape R - will this be a problem while distributin???
  R_tens = R_
  R_tens = tf.concat([R_tens, tf.zeros([1, beta_shp-1])], 0)
  R_tens = tf.concat([R_tens, tf.zeros([beta_shp, 1])], 1)

  R_ = R_tens + beta_mat
  # add index to marg_vec_
  marg_vec_ = tf.concat([marg_vec_, tf.reshape(glob_index, [1])], 0)
  # add alpha to alpha_marg_
  alpha_marg_ = tf.concat([alpha_marg_, tf.reshape(alpha, [1])], 0),
  # Force the above two operations
  return loop_var._replace(R=R_, alpha_marg=alpha_marg_, marg_vec=marg_vec_)

# I need to test this function
def rem_helper(elem, R_, sz, k):
  i = elem//sz
  j = elem%sz
  return R_[i][j] - (1/R_[k][k])*R_[i][k]*R_[k][j]

# Need to assert that this value is same inverting Q_
def rem_from_marg_sup(kernel_fn, loc_index, loop_var):
  # with tf.variable_scope(scope, reuse=True):
  R_ = loop_var.R
  marg_vec_ = loop_var.marg_vec
  alpha_marg_ = loop_var.alpha_marg

  k = loc_index + 1

  sz = tf.shape(R_)[0]
  iter_tensor = tf.range(sz*sz)
  # iter_tensor = tf.reshape(iter_tensor, [sz, sz])

  # R_new = tf.Print(R_tens, [tf.constant("----------Line 252---"),
  #  tf.shape(R_tens), R_tens[0][0], sz])
  R_new = tf.map_fn(lambda i: rem_helper(i, R_, sz, k), 
    iter_tensor, dtype=tf.float32)
  R_new = tf.reshape(R_new, [sz, sz])
  # # drop k th row
  R_new = tf.concat([R_new[:k, :], R_new[k+1:, :]], 0)
  # # drop kth col
  R_new = tf.concat([R_new[:, :k], R_new[:, k+1:]], 1)

  R_ = R_new

  # remove index from marg_vec_
  marg_vec_ = tf.concat([marg_vec_[:loc_index], marg_vec_[loc_index+1:]], 0)
  # remove alpha value from alpha_marg
  alpha_marg_= tf.concat(
    [alpha_marg_[:loc_index], alpha_marg_[loc_index+1:]], 0)
  # Force the update operations
  return loop_var._replace(R=R_, marg_vec=marg_vec_, alpha_marg=alpha_marg_)

# Returns a loop_var of the updated loop_var
def handle_marg(params, min_indx, beta_, loop_var):
  # with tf.variable_scope(scope, reuse=True):
  marg_vec_ = loop_var.marg_vec
  # If beta[min_indx+1] < 0, then alpha becomes 0 and joins remaining vec,
  # else alpha becomes C and joins error vectors
  loop_var = tf.cond(
        beta_[min_indx+1] < 0, 
        lambda: add_to_rem(params["eps"], 
          tf.constant(0.), marg_vec_[min_indx], loop_var),
        lambda: add_to_err(tf.constant(0.), marg_vec_[min_indx], loop_var)
        )
  # Remove vector from R_ and marg_vec_
  # with tf.control_dependencies([op1]):
  return rem_from_marg_sup(params["kernel"], min_indx, loop_var)

def update_g(g_tens, gamma_tens, min_alpha):
  iter_tensor = tf.range(tf.shape(g_tens)[0])
  return tf.cond(
    tf.equal(tf.shape(g_tens)[0], 0),
    lambda: tf.constant([]),
    lambda: tf.map_fn(
      lambda i: g_tens[i] + min_alpha*gamma_tens[i],
      iter_tensor, dtype=tf.float32)
  )

# Tensor should have non-zero shape 
def check_min(tens, val):
  min_indx = tf.argmin(tens)
  return tf.cond(tf.equal(tens[min_indx], val),
    lambda: tf.cast(min_indx, tf.int32),
    lambda: tf.shape(tens)[0] + 1)

# returns alpha_c and g_c
def mini_iter(params, iter_ct, loop_var):
  with tf.control_dependencies(None):
    marg_vec_ = loop_var.marg_vec
    err_vec_ = loop_var.err_vec
    rem_vec_ = loop_var.rem_vec
    alpha_marg_ = loop_var.alpha_marg
    g_err_ = loop_var.g_err
    g_rem_ = loop_var.g_rem
    g_c_ = loop_var.g_c
    alpha_c_ = loop_var.alpha_c
    b_ = loop_var.b
    R_ = loop_var.R

    x_all_ = loop_var.x_all
    y_all_ = loop_var.y_all 
    n = loop_var.n 


    printer = tf.Print(tf.constant(0), [tf.constant("AAAAAAAAAAAA")])
    # Handle case of matrix containing inf
    R_ = tf.cond(
      tf.equal(tf.shape(loop_var.marg_vec)[0], 0),
      lambda: tf.constant([[-INF]]),
      lambda: loop_var.R)
    loop_var = loop_var._replace(R=R_)
    
    print_op = tf.Print(tf.constant(1.), [tf.constant("----------Line 348 ------: "),
      tf.constant("MARG: "), loop_var.marg_vec,
      tf.constant("ERR: "), loop_var.err_vec,
      tf.constant("REM: "), loop_var.rem_vec,
      tf.constant("ALPHA: "), loop_var.alpha_marg,
      tf.constant("B: "), loop_var.b,
      tf.constant("R: "), loop_var.R,
      iter_ct])

    # Get beta and gamma necessary for the iteration
    with tf.control_dependencies([print_op]):
      beta_ = get_beta(params["kernel"], x_all_, y_all_,
        n-1, marg_vec_, R_)
      # Gamma for err_vec
    gamma_err_ = get_gamma(params["kernel"], err_vec_, beta_,
      x_all_, y_all_, n, marg_vec_)
    # Gamma for rem_vec
    gamma_rem_ = get_gamma(params["kernel"], rem_vec_, beta_,
      x_all_, y_all_, n, marg_vec_)
    # Gamma for candidate vector
    n_temp = n-1
    gamma_c_ = get_gamma(params["kernel"], tf.reshape(n_temp, [1]), beta_,
      x_all_, y_all_, n, marg_vec_)

    with tf.control_dependencies([beta_]): 
      return (iter_ct+1, loop_var)

    # # Book-keeping -
    # # For margin support vectors, if Beta_s > 0, alpha_c can go to C or 
    # # if Beta_s < 0, alpha_s can go to 0, causing a transition in state
    # iter_tensor_a = tf.range(tf.shape(marg_vec_)[0])
    # # Add extra condition to prevent map_fn on empty iter_tensor
    # marg_trans_alpha = tf.cond(tf.equal(tf.shape(iter_tensor_a)[0], 0),
    #   lambda: tf.constant([]),
    #   lambda: tf.map_fn(
    #     lambda i: tf.cond(beta_[i+1] < 0,
    #       # Alpha goes to zero
    #       lambda: -1 * alpha_marg_[i] / beta_[i+1],
    #       lambda: tf.cond(beta_[i+1] > 0,
    #         # Alpha goes to C
    #         lambda: (params["C"] - alpha_marg_[i]) / beta_[i+1],
    #         # Beta is zero
    #         lambda: tf.constant(float("Inf"))
    #         )
    #       ),
    #     iter_tensor_a, dtype=tf.float32)
    # )
    # # For error support vectors, if Gamma_i > 0, then g_i increases to 0
    # # and causes state change
    # iter_tensor_e = tf.range(tf.shape(err_vec_)[0])
    # err_trans_alpha = tf.cond(tf.equal(tf.shape(iter_tensor_e)[0], 0),
    #   lambda: tf.constant([]),
    #   lambda: tf.map_fn(
    #     lambda i: tf.cond(gamma_err_[i] > 0,
    #       lambda: -1 * g_err_[i] / gamma_err_[i],
    #       lambda: tf.constant(float("Inf"))
    #       ),
    #     iter_tensor_e, dtype=tf.float32)
    #   )

    # # For the remaining vectors, if Gamma_i < 0, then g_i decreases to 0
    # iter_tensor_r = tf.range(tf.shape(rem_vec_)[0])
    # rem_trans_alpha = tf.cond(tf.equal(tf.shape(iter_tensor_r)[0], 0),
    #   lambda: tf.constant([]),
    #   lambda: tf.map_fn(
    #     lambda i: tf.cond(gamma_rem_[i] < 0,
    #       lambda: -1 * g_rem_[i] / gamma_rem_[i],
    #       lambda: tf.constant(float("Inf"))
    #       ),
    #     iter_tensor_r, dtype=tf.float32)
    #   )
    # # For candidate vector, if gamma_i > 0, then g_i increases to 0
    # # or alpha_c increases to C_svm
    # candidate_trans_alpha = tf.minimum(
    #   tf.cond(gamma_c_[0] > 0,
    #     lambda: -1*g_c_/gamma_c_[0],
    #     lambda: tf.constant(float("Inf"))),
    #   params["C"] - alpha_c_)

    # # Compare the min and do the necessary transitions
    # all_trans_alpha = tf.concat(
    #   [marg_trans_alpha, err_trans_alpha, rem_trans_alpha, 
    #   tf.reshape(candidate_trans_alpha, [1,])], 0)
    # min_alpha = tf.reduce_min(all_trans_alpha)

    # # CALCULATING NEW VALUES
    # # Calculate new b
    # b_ = b_ + beta_[0]*min_alpha
    # loop_var = loop_var._replace(b=b_)
    # # Calculate new values of alpha for marg vec
    # iter_tensor = tf.range(tf.shape(alpha_marg_)[0])
    # # Condition to not run the map fn on zero sz tensor
    # alpha_marg_ = tf.cond(tf.equal(tf.shape(iter_tensor)[0], 0),
    #   lambda: tf.constant([]),
    #   lambda: tf.map_fn(
    #     lambda i: alpha_marg_[i] + beta_[i+1]*min_alpha,
    #     iter_tensor, dtype=tf.float32)
    #   )
    # loop_var = loop_var._replace(alpha_marg=alpha_marg_)
    # # Calculate new value of alpha_c
    # alpha_c_ = alpha_c_ + min_alpha
    # loop_var = loop_var._replace(alpha_c=alpha_c_)

    # # Calculate new values of g_all
    # g_err_ = update_g(g_err_, gamma_err_, min_alpha)
    # g_rem_ = update_g(g_rem_, gamma_rem_, min_alpha)
    # g_c_ = g_c_ + gamma_c_[0]*min_alpha
    # loop_var = loop_var._replace(g_rem=g_rem_, g_err=g_err_, g_c=g_c_)

    # print_op2 = tf.Print(tf.constant(1.),
    #   [tf.constant("BETA: "), beta_,
    #   tf.constant("GAMMA_c: "), gamma_c_,
    #   tf.constant("ALL TRANS: "), all_trans_alpha])
    # with tf.control_dependencies([print_op2]):
    #   # MOVING VECTORS
    #   # Check if min is in marg sup vectors
    #   # Run get index only if tnsor sz is non-zero
    #   min_indx = tf.cond(tf.equal(tf.shape(marg_trans_alpha)[0], 0),
    #     lambda: tf.constant(1),
    #     lambda: check_min(marg_trans_alpha, min_alpha),
    #     )
    #   loop_var = tf.cond(
    #     min_indx < tf.shape(marg_vec_)[0],
    #     lambda: handle_marg(params, min_indx, beta_, loop_var),
    #     lambda: loop_var,
    #     name="476")

    #   # # Check if min is in err sup vectors
    #   # def handle_err(loc_index, loop_var):
    #   #   loop_var = add_to_marg_supp(params["kernel"], err_vec_[loc_index],
    #   #     gamma_err_[loc_index], params["C"], x_all_, y_all_, loop_var)
    #   #   loop_var = rem_from_err(loc_index, loop_var)
    #   #   return loop_var

    #   # min_indx = tf.cond(tf.equal(tf.shape(err_trans_alpha)[0], 0),
    #   #   lambda: tf.constant(1),
    #   #   lambda: check_min(err_trans_alpha, min_alpha)
    #   #   )

    #   # loop_var = tf.cond(
    #   #   min_indx < tf.shape(err_trans_alpha)[0],
    #   #   lambda: handle_err(min_indx, loop_var),
    #   #   lambda: loop_var,)
      
    #   # # Check if min is in remaining vectors
    #   # def handle_rem(loc_index, loop_var):
    #   #   loop_var = add_to_marg_supp(params["kernel"], rem_vec_[loc_index],
    #   #     gamma_rem_[loc_index], 0., x_all_, y_all_, loop_var)
    #   #   loop_var = rem_from_rem(loc_index, loop_var)
    #   #   return loop_var

    #   # min_indx = tf.cond(tf.equal(tf.shape(rem_trans_alpha)[0], 0),
    #   #   lambda: tf.constant(1),
    #   #   lambda: check_min(rem_trans_alpha, min_alpha)
    #   #   )
    #   # loop_var = tf.cond(
    #   #   min_indx < tf.shape(rem_trans_alpha)[0],
    #   #   lambda: handle_rem(min_indx, loop_var),
    #   #   lambda: loop_var)
    #   # Need to add the candidate vector here as I need g and gamma values
    #   # loop_var = tf.cond(
    #   #   termn_condn(params["C"], loop_var),
    #   #   lambda: tf.cond(
    #   #     tf.equal(loop_var.g_c, 0.),
    #   #     lambda: add_to_marg_supp(params["kernel"],
    #   #       n-1, gamma_c_[0], loop_var.alpha_c, x_all_, y_all_, loop_var),
    #   #     lambda: add_to_err(loop_var.g_c, n-1, loop_var)),
    #   #   lambda: loop_var)

    #   printer = tf.Print(tf.constant(1.),
    #       [tf.constant("----------------Line 512----: "), n, loop_var.marg_vec,
    #       loop_var.err_vec, loop_var.rem_vec, loop_var.alpha_marg, loop_var.g_err,
    #       loop_var.R, loop_var.b])
    #   with tf.control_dependencies([printer]):
    #     return loop_var

def fit_point(params):
  marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32)
  err_vec_ = tf.get_variable("err_vec", dtype=tf.int32)
  rem_vec_ = tf.get_variable("rem_vec", dtype=tf.int32)
  alpha_marg_ = tf.get_variable("alpha_marg")
  g_err_ = tf.get_variable("g_err")
  g_rem_ = tf.get_variable("g_rem")
  g_c_ = tf.get_variable("g_c")
  alpha_c_ = tf.get_variable("alpha_c")
  b_ = tf.get_variable("b")
  R_ = tf.get_variable("R")

  # While loop supports named tuples and not dicts
  LoopVars = namedtuple("LoopVars",
    "x_all, y_all, n, marg_vec, err_vec, rem_vec, alpha_marg, g_err, g_rem, g_c, alpha_c, b, R")
  init_val = (tf.constant(0), LoopVars(
    x_all=tf.get_variable("x_all").read_value(),
    y_all=tf.get_variable("y_all").read_value(),
    n=tf.get_variable("n", dtype=tf.int32).read_value(),
    marg_vec=marg_vec_.read_value(),
    err_vec=err_vec_.read_value(),
    rem_vec=rem_vec_.read_value(),
    alpha_marg=alpha_marg_.read_value(),
    g_err=g_err_.read_value(),
    g_rem=g_rem_.read_value(),
    g_c=g_c_.read_value(),
    alpha_c=alpha_c_.read_value(),
    b=b_.read_value(),
    R=R_.read_value(),
    ))

  # init_val = tf.Print(init_val, [tf.constant("----------------Line 509--- "),
  #   fixed_vars["n"], fixed_vars["x_all"], fixed_vars["y_all"],
  #   tf.constant("------G_c: "), init_val])
  fin_i, fin_vars = tf.while_loop(
    cond=lambda i, p: tf.logical_and(tf.logical_not(termn_condn(params["C"], p)),
      tf.less(i, 1)),
    body=lambda i, p: mini_iter(params, i, p), 
    loop_vars=init_val,
    parallel_iterations=1,
    back_prop=False,
    name="FIT_WHILE"
    )

  upd_marg_vec = tf.assign(marg_vec_, fin_vars.marg_vec, validate_shape=False)
  upd_err_vec = tf.assign(err_vec_, fin_vars.err_vec, validate_shape=False)
  upd_rem_vec = tf.assign(rem_vec_, fin_vars.rem_vec, validate_shape=False)
  upd_alpha_marg = tf.assign(alpha_marg_, fin_vars.alpha_marg,
    validate_shape=False)
  upd_g_err = tf.assign(g_err_, fin_vars.g_err, validate_shape=False)
  upd_g_rem = tf.assign(g_rem_, fin_vars.g_rem, validate_shape=False)
  upd_alpha_c = tf.assign(alpha_c_, fin_vars.alpha_c, validate_shape=False)
  upd_g_c = tf.assign(g_c_, fin_vars.g_c, validate_shape=False)
  upd_b = tf.assign(b_, fin_vars.b, validate_shape=False)
  upd_R = tf.assign(R_, fin_vars.R, validate_shape=False)

  with tf.control_dependencies([upd_marg_vec, upd_err_vec, upd_rem_vec,
    upd_alpha_marg, upd_g_err, upd_g_rem, upd_alpha_c, upd_g_c, upd_b, upd_R]):
    return fin_i

def create_svm_variables(x_shape):
  with tf.variable_scope("svm_model") as scope:
    # Variable creation
    # count of the number of data points seen
    n_ = tf.get_variable("n", initializer=tf.constant(0), trainable=False,
      dtype=tf.int32)
    # offset
    b_ = tf.get_variable("b", initializer=tf.constant(0.), trainable=False,)
    # Variable tensor - containing all the data points and labels
    x_all_ = tf.get_variable("x_all", shape=[0, x_shape],
      validate_shape=False)
    y_all_ = tf.get_variable("y_all", shape=[0],
      validate_shape=False)

    # Variable tensor - representing the Margin Support vector indices
    marg_vec_ = tf.get_variable("marg_vec", shape=[0], dtype=tf.int32,
      validate_shape=False) 
    # Variable tensor - representing the Error Support vector indices
    err_vec_ = tf.get_variable("err_vec", shape=[0], dtype=tf.int32,
      validate_shape=False)
    # Variable tensor - representing the Remaining vector indices
    rem_vec_ = tf.get_variable("rem_vec", shape=[0], dtype=tf.int32,
      validate_shape=False)

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

    # Variable denoting the alpha of new candidate
    alpha_c_ = tf.get_variable("alpha_c", initializer=0.)
    # Variable denoting the g of candidate vector
    g_c_ = tf.get_variable("g_c", shape=())

    return scope

def return_w_b(scope, params):
  # get a1 and a2 for a1x+a2y+b=0
  with tf.variable_scope(scope, reuse=True):
    marg_vec_ = tf.get_variable("marg_vec")
    err_vec_ = tf.get_variable("err_vec")
    alpha_marg_ = tf.get_variable("alpha_marg")
    x_all_ = tf.get_variable("x_all")
    y_all_ = tf.get_variable("y_all")
    b_ = tf.get_variable("b")
    n_ = tf.get_variable("n")

    w = tf.zeros([2])

    iter_tensor = tf.range(tf.shape(marg_vec_)[0])
    temp_marg = tf.map_fn(
      lambda i: y_all_[marg_vec_[i]] * x_all_[marg_vec_[i]] * alpha_marg_[i],
      iter_tensor)

    iter_tensor = tf.range(tf.shape(err_vec_)[0])
    temp_err = tf.map_fn(
      lambda i: y_all_[err_vec_[i]] * x_all_[err_vec_[i]] * params["C"],
      iter_tensor)

    w = tf.reduce_sum(tf.concat([temp_marg, temp_err], 0), 0)
    return (w, b_, x_all_, y_all_, n_) 
  
def draw_dec_bdry(w, b, x_all, y_all, n):
  import numpy as np

  from matplotlib import colors
  from matplotlib import pyplot as plt
  
  x_all = np.array(x_all)
  y_all = np.array(y_all)

  plt.scatter(x_all[:n, 0], x_all[:n, 1], c=y_all[:n],
    cmap=colors.ListedColormap(['orange', 'red', 'green']))
  plt.plot()

  start = 0
  last = 1
  xl = [start, last]
  yl = []
  for xx in xl:
    yl.append((-b - w[0]*xx)/w[1])
  plt.plot(xl, yl)
  plt.show()

def svm_train(x_, y_, params):
  x_all_ = tf.get_variable("x_all")
  y_all_ = tf.get_variable("y_all")
  n_ = tf.get_variable("n", dtype=tf.int32)
  alpha_c_ = tf.get_variable("alpha_c")
  g_c_ = tf.get_variable("g_c")

  #  Concatanate x_all, y_all to include the new vector and it's labe
  op1 = tf.assign(x_all_, 
    tf.concat([x_all_, tf.reshape(x_, [1, params["shape"]])], 0), 
    validate_shape=False)
  op2 = tf.assign(y_all_, 
    tf.concat([y_all_, tf.reshape(y_, [1])], 0), 
    validate_shape=False, name="y_assign531")
  # Update count of data points seen
  op3 = tf.assign_add(n_, 1)
  # Reset alpha_c
  op4 = tf.assign(alpha_c_, 0.)
  # Force the above operations
  with tf.control_dependencies([op1, op2, op3, op4]):
    # g_c = tf.assign(g_c_, 0)
    g_c = tf.assign(g_c_, calc_g(params, x_, y_),
        validate_shape=False)
    return tf.cond(
      tf.greater(g_c, 0),
      lambda: add_to_rem(params["eps"], g_c, n_.read_value()-1),
      lambda: fit_point(params))

def svm_model_fn():
  train_file = "data_1.csv"
  x_train, y_train = extract_data(train_file, "csv")
  model_params["shape"] = x_train.shape[1]

  x_ = tf.placeholder(tf.float32, shape=(x_train.shape[1]))
  y_ = tf.placeholder(tf.float32, shape=())
  scope = create_svm_variables(x_shape=x_train.shape[1])

  with tf.variable_scope(scope, reuse=True):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      tr = svm_train(x_, y_, model_params)
      for i in range(1):
        print("\n \n")
        print(sess.run(tr, feed_dict={x_ : x_train[i], y_ : y_train[i]}))
        # y_all_ = tf.get_variable("y_all")
        # print(sess.run(y_all_))
      # writer = tf.summary.FileWriter(path_, sess.graph)
      # writer.close()
        # print(sess.run(r[0]))

svm_model_fn()    


# class Inc_SVM():
#   def __init__(train_file, C_svm, eps_thresh, kernel_fn):
#     self.params = {"C" : C_svm, "eps": eps_thresh, "kernel": kernel_fn}