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
# than loop vars no other ones run



import tensorflow as tf

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

def rem_from(vec_set, g_vec, loc_index):
  # Remove vector from vec_set
  vec_set_tens = vec_set.read_value()
  upd_vec = tf.assign(vec_set, 
    tf.concat([vec_set_tens[:loc_index], vec_set_tens[loc_index+1:]], 0))
  # REmove g val of the vector
  g_vec_tens = g_vec.read_value()
  upd_g = tf.assign(g_vec,
    tf.concat([g_vec_tens[:loc_index], g_vec_tens[loc_index+1:]], 0))
  # Force the above two operations
  with tf.control_dependencies([upd_vec, upd_g]):
    return tf.constant(1.)

def add_to_err(g_c, glob_index):
    err_vec_ = tf.get_variable("err_vec", dtype=tf.int32)
    g_err_ = tf.get_variable("g_err")
    # Add new index to err_vec
    op1 = tf.assign(err_vec_, 
      tf.concat([err_vec_.read_value(), [glob_index]], 0),
      validate_shape=False)
    # Add new index g to g_err
    op2 = tf.assign(g_err_,
      tf.concat([g_err_.read_value(), [g_c]], 0),
      validate_shape=False)
    with tf.control_dependencies([op1, op2]):
      return tf.constant(1.)

# TODO: Add the reserve threshold check
def add_to_rem(eps, g_c, glob_index):
  rem_vec_ = tf.get_variable("rem_vec", dtype=tf.int32)
  g_rem_ = tf.get_variable("g_rem")
  # Add new index to rem_vec
  op1 = tf.assign(rem_vec_,
    tf.concat([rem_vec_.read_value(), tf.reshape(glob_index, [1])], 0),
    validate_shape=False)
  # Add g_c to g_rem
  op2 = tf.assign(g_rem_, 
    tf.concat([g_rem_.read_value(), tf.reshape(g_c, [1])], 0),
    validate_shape=False)
  # Force the above operations to run
  with tf.control_dependencies([op1, op2]):
    # ans = tf.Print(tf.constant(1.), [op1, op2])
    return tf.constant(1.)

# Terminate if candidate g_c equals 0 or candidate alpha_c equals C
def termn_condn(C_svm, alpha_c, g_c):
  print_op = tf.Print(tf.constant(0), [tf.constant("TREM CONDN"), alpha_c, g_c])
  with tf.control_dependencies([print_op]):
    return tf.logical_or(tf.equal(g_c, 0), tf.equal(alpha_c, C_svm))

# TODO: do matrix inverse when size is one/inf
# Calculates the co-efficient sensitivities
def get_beta(kernel_fn, x_all_, y_all_, glob_index):
  marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32).read_value()
  R_ = tf.get_variable("R").read_value()

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
def gamma_helper_fn(kernel_fn, beta_, indx, x_all_, y_all_, n):
  # with tf.variable_scope(scope, reuse=True):
    marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32).read_value()
    # x_all_ = tf.get_variable("x_all")
    # y_all_ = tf.get_variable("y_all")
    # n = tf.get_variable("n").read_value()

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
def get_gamma(kernel_fn, vec, beta_, x_all_, y_all_, n):
  x_c = x_all_[n-1]
  y_c = y_all_[n-1]
  # vec = tf.Print(vec_1, [tf.constant("---------------Line 174"), beta_,
  #   y_all_, tf.shape(y_all_), tf.shape(x_all_), tf.shape(beta_)])
  return tf.cond(tf.equal(tf.shape(vec)[0], 0),
    lambda: tf.constant([]),
    lambda: tf.map_fn(
      lambda i: (y_all_[i]*y_c*kernel_fn(x_all_[i], x_c) + 
        beta_[0]*y_all_[i] +
        gamma_helper_fn(kernel_fn, beta_, i, x_all_, y_all_, n)),
      vec,
      dtype=tf.float32)
    )

# TODO: Take beta as arg to prevent recalculation in case of new candidate vector
def add_to_marg_supp(kernel_fn, glob_index, gamma, alpha, x_all_, y_all_):
  # with tf.variable_scope(scope, reuse=True):
  R_ = tf.get_variable("R")
  marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32)
  alpha_marg_ = tf.get_variable("alpha_marg")

  beta_indx = get_beta(kernel_fn, x_all_, y_all_, glob_index)
  beta_indx = tf.concat([beta_indx, [1]], 0)
  beta_shp = tf.shape(beta_indx)[0]

  beta_mat = tf.matmul(
    tf.reshape(beta_indx, [beta_shp, 1]),
    tf.reshape(beta_indx, [1, beta_shp]))

  beta_mat = 1/gamma * beta_mat

  # reshape R - will this be a problem while distributin???
  R_tens = R_.read_value()
  R_tens = tf.concat([R_tens, tf.zeros([1, beta_shp-1])], 0)
  R_tens = tf.concat([R_tens, tf.zeros([beta_shp, 1])], 1)

  upd_R = tf.assign(R_, R_tens + beta_mat, validate_shape=False)
  # add index to marg_vec_
  upd_marg = tf.assign(marg_vec_,
    tf.concat([marg_vec_.read_value(), tf.reshape(glob_index, [1])], 0),
    validate_shape=False)
  # add alpha to alpha_marg_
  upd_alpha = tf.assign(alpha_marg_,
    tf.concat([alpha_marg_.read_value(), tf.reshape(alpha, [1])], 0),
    validate_shape=False)
  # Force the above two operations
  with tf.control_dependencies([upd_R, upd_alpha, upd_marg]):
    return tf.constant(1.)

# I need to test this function
def rem_helper(elem, R_, sz, k):
  i = elem//sz
  j = elem%sz
  return R_[i][j] - (1/R_[k][k])*R_[i][k]*R_[k][j]

# Need to assert that this value is same inverting Q_
def rem_from_marg_sup(kernel_fn, loc_index):
  # with tf.variable_scope(scope, reuse=True):
  R_ = tf.get_variable("R")
  marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32)
  alpha_marg_ = tf.get_variable("alpha_marg")

  k = loc_index + 1

  sz = tf.shape(R_.read_value())[0]
  iter_tensor = tf.range(sz*sz)
  # iter_tensor = tf.reshape(iter_tensor, [sz, sz])

  R_tens = R_.read_value()
  # R_new = tf.Print(R_tens, [tf.constant("----------Line 252---"),
  #  tf.shape(R_tens), R_tens[0][0], sz])
  R_new = tf.map_fn(lambda i: rem_helper(i, R_tens, sz, k), 
    iter_tensor, dtype=tf.float32)
  R_new = tf.reshape(R_new, [sz, sz])
  # # drop k th row
  R_new = tf.concat([R_new[:k, :], R_new[k+1:, :]], 0)
  # # drop kth col
  R_new = tf.concat([R_new[:, :k], R_new[:, k+1:]], 1)

  upd_r = tf.assign(R_, R_new, validate_shape=False)

  # remove index from marg_vec_
  marg_vec_tens = marg_vec_.read_value()
  upd_marg = tf.assign(marg_vec_,
    tf.concat([marg_vec_tens[:loc_index], marg_vec_tens[loc_index+1:]], 0),
    validate_shape=False)
  # remove alpha value from alpha_marg
  alpha_marg_tens = alpha_marg_.read_value()
  upd_alpha = tf.assign(alpha_marg_,
    tf.concat([alpha_marg_tens[:loc_index], alpha_marg_tens[loc_index+1:]], 0),
    validate_shape=False)
  # Force the update operations
  with tf .control_dependencies([upd_r, upd_marg, upd_alpha]):
    return tf.constant(1.)

def handle_marg(params, min_indx, beta_):
  # with tf.variable_scope(scope, reuse=True):
  marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32).read_value()
  # If beta[min_indx+1] < 0, then alpha becomes 0 and joins remaining vec,
  # else alpha becomes C and joins error vectors
  op1 = tf.cond(
        beta_[min_indx+1] < 0, 
        lambda: add_to_rem(params["eps"], 
          tf.constant(0.), marg_vec_[min_indx]),
        lambda: add_to_err(tf.constant(0.), marg_vec_[min_indx])
        )
  # Remove vector from R_ and marg_vec_
  with tf.control_dependencies([op1]):
    op2 = rem_from_marg_sup(params["kernel"], min_indx)

    with tf.control_dependencies([op2]):
      return tf.constant(1.)


def update_g(g_var, gamma_tens, min_alpha):
  g_tens = g_var.read_value()
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
def mini_iter(params, i, j, x_all_, y_all_, n):
  with tf.control_dependencies(None):
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

    printer = tf.Print(tf.constant(0), [tf.constant("AAAAAAAAAAAA")])
    # Handle case of matrix containing inf
    inf_op = tf.cond(
      tf.equal(tf.shape(marg_vec_)[0], printer),
      lambda: tf.assign(R_, tf.constant([[-INF]])),
      lambda: tf.constant([[-1.]]))
    
    print_op = tf.Print(tf.constant(1.), [tf.constant("----------Line 334 ------: "),
      tf.constant("MARG: "), marg_vec_.read_value(),
      tf.constant("ERR: "), err_vec_.read_value(), 
      tf.constant("REM: "), rem_vec_.read_value(),
      tf.constant("ALPHA: "), alpha_marg_.read_value(),
      tf.constant("B: "), b_.read_value()])
    # Get beta and gamma necessary for the iteration
    with tf.control_dependencies([inf_op, print_op]):
      beta_ = get_beta(params["kernel"], x_all_, y_all_, n-1)
    # Gamma for err_vec
    gamma_err_ = get_gamma(params["kernel"], err_vec_.read_value(), beta_,
      x_all_, y_all_, n)
    # Gamma for rem_vec
    gamma_rem_ = get_gamma(params["kernel"], rem_vec_.read_value(), beta_,
      x_all_, y_all_, n)
    # Gamma for candidate vector
    n_temp = n-1
    gamma_c_ = get_gamma(params["kernel"], tf.reshape(n_temp, [1]), beta_,
      x_all_, y_all_, n)

    # Book-keeping -
    # For margin support vectors, if Beta_s > 0, alpha_c can go to C or 
    # if Beta_s < 0, alpha_s can go to 0, causing a transition in state
    iter_tensor_a = tf.range(tf.shape(marg_vec_.read_value())[0])
    # read value to get the latest value of the variable
    alpha_marg_tens = alpha_marg_.read_value()
    # Add extra condition to prevent map_fn on empty iter_tensor
    marg_trans_alpha = tf.cond(tf.equal(tf.shape(iter_tensor_a)[0], 0),
      lambda: tf.constant([]),
      lambda: tf.map_fn(
        lambda i: tf.cond(beta_[i+1] < 0,
          # Alpha goes to zero
          lambda: -1 * alpha_marg_tens[i] / beta_[i+1],
          lambda: tf.cond(beta_[i+1] > 0,
            # Alpha goes to C
            lambda: (params["C"] - alpha_marg_tens[i]) / beta_[i+1],
            # Beta is zero
            lambda: tf.constant(float("Inf"))
            )
          ),
        iter_tensor_a, dtype=tf.float32)
    )
    # For error support vectors, if Gamma_i > 0, then g_i increases to 0
    # and causes state change
    iter_tensor_e = tf.range(tf.shape(err_vec_.read_value())[0])
    g_err_tens = g_err_.read_value()
    err_trans_alpha = tf.cond(tf.equal(tf.shape(iter_tensor_e)[0], 0),
      lambda: tf.constant([]),
      lambda: tf.map_fn(
        lambda i: tf.cond(gamma_err_[i] > 0,
          lambda: -1 * g_err_tens[i] / gamma_err_[i],
          lambda: tf.constant(float("Inf"))
          ),
        iter_tensor_e, dtype=tf.float32)
      )

    # For the remaining vectors, if Gamma_i < 0, then g_i decreases to 0
    iter_tensor_r = tf.range(tf.shape(rem_vec_)[0])
    g_rem_tens = g_rem_.read_value()
    rem_trans_alpha = tf.cond(tf.equal(tf.shape(iter_tensor_r)[0], 0),
      lambda: tf.constant([]),
      lambda: tf.map_fn(
        lambda i: tf.cond(gamma_rem_[i] < 0,
          lambda: -1 * g_rem_tens[i] / gamma_rem_[i],
          lambda: tf.constant(float("Inf"))
          ),
        iter_tensor_r, dtype=tf.float32)
      )
    # For candidate vector, if gamma_i > 0, then g_i increases to 0
    # or alpha_c increases to C_svm
    candidate_trans_alpha = tf.minimum(
      tf.cond(gamma_c_[0] > 0,
        lambda: -1*g_c_.read_value()/gamma_c_[0],
        lambda: tf.constant(float("Inf"))),
      params["C"] - alpha_c_.read_value())

    # Compare the min and do the necessary transitions
    all_trans_alpha = tf.concat(
      [marg_trans_alpha, err_trans_alpha, rem_trans_alpha, 
      tf.reshape(candidate_trans_alpha, [1,])], 0)
    min_alpha = tf.reduce_min(all_trans_alpha)

    # CALCULATING NEW VALUES
    # Calculate new b
    b_new = b_.read_value() + beta_[0]*min_alpha
    upd_b = tf.assign(b_, b_new)
    # Calculate new values of alpha for marg vec
    alpha_marg_tens = alpha_marg_.read_value()
    iter_tensor = tf.range(tf.shape(alpha_marg_tens)[0])
    # Condition to not run the map fn on zero sz tensor
    alpha_marg_new = tf.cond(tf.equal(tf.shape(iter_tensor)[0], 0),
      lambda: tf.constant([]),
      lambda: tf.map_fn(
        lambda i: alpha_marg_tens[i] + beta_[i+1]*min_alpha,
        iter_tensor, dtype=tf.float32)
      )
    upd_alpha = tf.assign(alpha_marg_, alpha_marg_new)
    # Calculate new value of alpha_c
    upd_alpha_c = tf.assign(alpha_c_, alpha_c_.read_value() + min_alpha)

    # Calculate new values of g_all
    upd_g_err = tf.assign(g_err_, update_g(g_err_, gamma_err_, min_alpha))
    upd_g_rem = tf.assign(g_rem_, update_g(g_rem_, gamma_rem_, min_alpha))
    upd_g_c = tf.assign(g_c_, g_c_.read_value() + gamma_c_[0]*min_alpha)

    print_op2 = tf.Print(tf.constant(1.),
      [tf.constant("BETA: "), beta_,
      tf.constant("GAMMA_c: "), gamma_c_,
      tf.constant("ALL TRANS: "), all_trans_alpha])
    with tf.control_dependencies([upd_b, upd_alpha, upd_alpha_c, upd_g_err,
      upd_g_rem, upd_g_c, print_op2]):
      # MOVING VECTORS
      # Check if min is in marg sup vectors
      # Run get index only if tnsor sz is non-zero
      min_indx = tf.cond(tf.equal(tf.shape(marg_trans_alpha)[0], 0),
        lambda: tf.constant(1),
        lambda: check_min(marg_trans_alpha, min_alpha),
        )
      mov_marg = tf.cond(
        min_indx < tf.shape(marg_vec_.read_value())[0],
        lambda: handle_marg(params, min_indx, beta_),
        lambda: tf.constant(-1.))

      # Check if min is in err sup vectors
      def handle_err(loc_index):
        op1 = add_to_marg_supp(params["kernel"], err_vec_.read_value()[loc_index],
          gamma_err_[loc_index], params["C"], x_all_, y_all_)
        op2 = rem_from(err_vec_, g_err_, loc_index)
        return op1 + op2

      min_indx = tf.cond(tf.equal(tf.shape(err_trans_alpha)[0], 0),
        lambda: tf.constant(1),
        lambda: check_min(err_trans_alpha, min_alpha)
        )

      mov_err = tf.cond(
        min_indx < tf.shape(err_vec_)[0],
        lambda: handle_err(min_indx),
        lambda: tf.constant(-1.))
      
      # Check if min is in remaining vectors
      def handle_rem(loc_index):
        op1 = add_to_marg_supp(params["kernel"], rem_vec_.read_value()[loc_index],
          gamma_rem_[loc_index], 0.,x_all_, y_all_)
        op2 = rem_from(rem_vec_, g_rem_, loc_index)
        return op1 + op2

      min_indx = tf.cond(tf.equal(tf.shape(rem_trans_alpha)[0], 0),
        lambda: tf.constant(1),
        lambda: check_min(rem_trans_alpha, min_alpha)
        )
      mov_rem = tf.cond(
        min_indx < tf.shape(rem_vec_)[0],
        lambda: handle_rem(min_indx),
        lambda: tf.constant(-1.))
      # Need to add the candidate vector here as I need g and gamma values
      mov_c = tf.cond(
        termn_condn(params["C"], upd_alpha_c, upd_g_c),
        lambda: tf.cond(
          tf.equal(upd_g_c, 0),
          lambda: add_to_marg_supp(params["kernel"],
            n-1, gamma_c_[0], upd_alpha_c, x_all_, y_all_),
          lambda: add_to_err(upd_g_c, n-1)),
        lambda: tf.constant(-1.))

      with tf.control_dependencies([mov_marg, mov_err, mov_rem, mov_c]):
        upd_alpha_c1 = tf.Print(upd_alpha_c,
          [tf.constant("----------------Line 493---- "), n,
          min_alpha, mov_marg, mov_err, mov_rem, mov_c, upd_alpha_c,
          upd_g_c, marg_vec_.read_value(), err_vec_.read_value(), rem_vec_.read_value()])
        return [upd_alpha_c1, upd_g_c]

def fit_point(params):
  mini_params = {}

  alpha_c = tf.get_variable("alpha_c").read_value()
  g_c = tf.get_variable("g_c").read_value()
  x_all_ = mini_params["x_all_"] = tf.get_variable("x_all").read_value()
  y_all_ = tf.get_variable("y_all").read_value()
  n = tf.get_variable("n", dtype=tf.int32).read_value()

  # marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32)
  # err_vec_ = tf.get_variable("err_vec", dtype=tf.int32)
  # rem_vec_ = tf.get_variable("rem_vec", dtype=tf.int32)
  # alpha_marg_ = tf.get_variable("alpha_marg")
  # g_err_ = tf.get_variable("g_err")
  # g_rem_ = tf.get_variable("g_rem")
  # g_c_ = tf.get_variable("g_c")
  # alpha_c_ = tf.get_variable("alpha_c")
  # b_ = tf.get_variable("b")
  # R_ = tf.get_variable("R")

  alpha_c = tf.Print(alpha_c, [tf.constant("----------------Line 509--- "),
    n, x_all_, y_all_, tf.constant("------G_c: "), g_c, alpha_c])
  fin_alpha, fin_g = tf.while_loop(
    cond=lambda i, j: tf.logical_not(termn_condn(params["C"], i, j)),
    body=lambda i, j: mini_iter(params, i, j, x_all_, y_all_, n), 
    loop_vars=[alpha_c, g_c],
    # parallel_iterations=1,
    # back_prop=False,
    name="FIT_WHILE"
    )

  return (fin_alpha)

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
      for i in range(3):
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