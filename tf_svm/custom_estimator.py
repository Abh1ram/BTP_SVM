import tensorflow as tf


C_SVM = 5.
RESERVE_THRESHOLD = -1

INF = 10**12

# Simple dot product
def simple_kernel(x1, x2):
  return tf.tensordot(x1, x2, [0,0])[0]

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

def calc_f(scope, params, x_):
  with tf.variable_scope(scope, reuse=True):
    x_all_ = tf.get_variable("x_all")
    y_all_ = tf.get_variable("y_all")
    b_ = tf.get_variable("b")
    marg_vec_ = tf.get_variable("marg_vec", dtype=tf.int32)
    err_vec_ = tf.get_variable("err_vec", dtype=tf.int32)
    alpha_marg_ = tf.get_variable("alpha_marg")

    alpha_err_ = tf.map_fn(lambda x: params["C"], err_vec_, dtype=tf.float32)
    # f(x_) = SUM(alpha[j] * y[j] * K(x_all[j] * x_)) + b
    
    # CHECK: Should I concat both the vectors to get all supp vectors
    # and then do the mulitplication (more parallelism) vs do them individually
    # and then join them (more distributed)??

    # More parallelism
    supp_vec_ = tf.concat([marg_vec_, err_vec_], 0)
    alpha_supp_ = tf.concat([alpha_marg_, alpha_err_], 0)

    # supp_vec2_ = tf.Print(supp_vec_, [tf.constant([1,2,3]),supp_vec_, tf.shape(y_all_),
      # tf.shape(x_all_), tf.shape(x_), tf.shape(supp_vec_)])
    temp = tf.map_fn(
      lambda i: y_all_[i] 
      # * params["kernel"](x_all_[i], x_)
      ,
      supp_vec_, dtype=tf.float32, name="Dipshit")
    # supp_vec2_ = supp_vec2_ + 1
    # return (supp_vec2_)
    return (tf.tensordot(alpha_supp_, temp, [0,0]) + b_)

# x_ - tensor
# y_ - tensor of rank 0/scalar
def calc_g(scope, params, x_, y_):
  # g(x, y) = f(x)*y - 1 
  return (calc_f(scope, params, x_) * y_) - 1.

def rem_from(vec_set, g_vec, loc_index):
  _ = tf.assign(vec_set, 
    tf.concat([vec_set[:loc_index], vec_set[loc_index+1:]], 0))

  _ = tf.assign(g_vec, 
    tf.concat([g_vec[:loc_index], g_vec[loc_index+1:]], 0))

def add_to_err(scope, g_c, glob_index):
  with tf.variable_scope(scope, reuse=True):
    err_vec_ = tf.get_variable("err_vec")
    g_err_ = tf.get_variable("g_err")

    _ = tf.assign(err_vec_, tf.concat([err_vec_, [glob_index]], 0),
      validate_shape=False)
    _ = tf.assign(g_err_, tf.concat([g_err_, [g_c]], 0))

# TODO: Add the reserve threshold check
def add_to_rem(scope, eps, g_c, glob_index):
  with tf.variable_scope(scope, reuse=True):
    rem_vec_ = tf.get_variable("rem_vec")
    g_rem_ = tf.get_variable("g_rem")

    _ = tf.assign(rem_vec_, tf.concat([rem_vec_, [glob_index]], 0),
      validate_shape=False)
    _ = tf.assign(g_rem_, tf.concat([g_rem_, [g_c]], 0))

# Terminate if candidate g_c equals 0 or candidate alpha_c equals C
def termn_condn(C_svm, alpha_c, g_c):
  return tf.logical_or(tf.equal(g_c, 0), tf.equal(alpha_c, C_svm))

# Calculates the co-efficient sensitivities
def get_beta(scope, kernel_fn, glob_index):
  with tf.variable_scope(scope, reuse=True):
    marg_vec_ = tf.get_variable("marg_vec")
    x_all_ = tf.get_variable("x_all")
    y_all_ = tf.get_variable("y_all")
    n = tf.get_variable("n").read_value()
    R_ = tf.get_variable("R")

    x_c = x_all_[glob_index]
    y_c = y_all_[glob_index]

    # Calculating Q_(s_i, c) = y_all[s_i]*y_c*
    temp_q = tf.map_fn(
      lambda i: y_c * y_all_[i] * kernel_fn(x_c, x_all_[i]), 
      marg_vec_)
    # Add y_c as the first element
    temp_q = tf.concat([[y_c], temp_q], 0)
    # reshape temp_q from (n,) to (n,1)/ rank 1 to rank 2 for matmul
    temp_q = tf.reshape(temp_q, [tf.shape(temp_q)[0], 1])

    beta = -1 * tf.matmul(R, temp_q)
    return tf.reshape(beta, [tf.shape(beta)[0]])

# Computes SUM(Q_ij*B_j)
def gamma_helper_fn(scope, kernel_fn, beta_, indx):
  with tf.variable_scope(scope, reuse=True):
    marg_vec_ = tf.get_variable("marg_vec")
    x_all_ = tf.get_variable("x_all")
    y_all_ = tf.get_variable("y_all")
    n = tf.get_variable("n").read_value()

    iter_tensor = tf.range(tf.shape(marg_vec_)[0])

    return tf.reduce_sum(
      tf.map_fn(
        lambda i: ( y_all_[indx] * y_all_[marg_vec_[i]] * beta_[i + 1] *
          kernel_fn(x_all_[indx], x_all_[marg_vec_[i]]) ),
        iter_tensor,
        )
    )

# Making this more distributed
# vec is a tensor of indices for which gamma needs to be computed
def get_gamma(scope, kernel_fn, vec, beta_):
  with tf.variable_scope(scope, reuse=True):
    marg_vec_ = tf.get_variable("marg_vec")
    x_all_ = tf.get_variable("x_all")
    y_all_ = tf.get_variable("y_all")
    n = tf.get_variable("n").read_value()

    x_c = x_all_[n-1]
    y_c = y_all_[n-1]


    return tf.map_fn(
      lambda i: (y_all_[i]*y_c*kernel_fn(x_all_[i], x_c) + 
        beta_[0]*y_all_[i] +
        gamma_helper_fn(scope, kernel_fn, beta_, i)),
      vec)


def add_to_marg_supp(scope, kernel_fn, glob_index, gamma, alpha):
  with tf.variable_scope(scope, reuse=True):
    R_ = tf.get_variable("R")
    marg_vec_ = tf.get_variable("marg_vec")
    alpha_marg_ = tf.get_variable("alpha_marg_")

    beta_indx = get_beta(scope, kernel_fn, glob_index)
    beta_indx = tf.concat([beta_indx, [1]], 0)
    beta_shp = tf.shape(beta_indx)[0]

    beta_mat = tf.matmul(
      tf.reshape(beta_indx, [beta_shp, 1]),
      tf.reshape(beta_indx, [1, beta_shp]))

    beta_mat = 1/gamma * beta_mat

    # reshape R - will this be a problem while distributin???
    _ = tf.assign(R_, tf.concat([R_, tf.zeros([1, beta_shp-1])], 0))
    _ = tf.assign(R_, tf.concat([R_, tf.zeros([beta_shp, 1])], 1))

    _ = tf.assign(R_, R_.read_value() + beta_mat)
    # add index to marg_vec_
    _ = tf.assign(marg_vec_, tf.concat([marg_vec_, [glob_index]], 0))
    # add alpha to alpha_marg_
    _ = tf.assign(alpha_marg_, tf.concat([alpha_marg_, [alpha]], 0))

def get_index(tens_, val):
  i = tf.constant(0)
  sz = tf.shape(tens_)[0]
  c = lambda i: tf.logical_and(tf.less(i ,sz), 
    tf.not_equal(tens_[i], val))
  b = lambda i: tf.add(i, 1)
  r = tf.while_loop(c, b, [i])
  # assert r is  tensor of rank 0
  return r

# I need to test this function
def rem_helper(elem, R_, sz, k):
  i = elem/sz
  j = elem%sz
  return R_[i][j] - (1/R_[k][k])*R_[i][k]*R_[k][j]

# Need to assert that this value is same inverting Q_
def rem_from_marg_sup(scope, kernel_fn, loc_index):
  with tf.variable_scope(scope, reuse=True):
    R_ = tf.get_variable("R")
    marg_vec_ = tf.get_variable("marg_vec")
    alpha_marg_ = tf.get_variable("alpha_marg")

    k = loc_index + 1

    sz = tf.shape(R_)[0]
    iter_tensor = tf.range(sz*sz)
    iter_tensor = tf.reshape(iter_tensor, [sz, sz])

    R_new = tf.map_fn(lambda i: rem_helper(i, R_.read_value, sz, k), 
      iter_tensor)
    # drop k th row
    R_new = tf.concat([R_new[:k, :], R_new[k+1:, :]], 0)
    # drop kth col
    R_new = tf.concat([R_new[:, :k], R_new[:, k+1:]], 1)

    _ = tf.assign(R_, R_new)

    # remove index from marg_vec_
    _ = tf.assign(marg_vec_,
      tf.concat([marg_vec_[:loc_index], marg_vec_[loc_index+1:]], 0))
    # remove alpha value from alpha_marg
    _ = tf.assign(alpha_marg_,
      tf.concat([alpha_marg_[:loc_index], alpha_marg_[loc_index+1:]], 0))

def handle_marg(scope, params, min_indx, beta_):
  with tf.variable_scope(scope, reuse=True):
    marg_vec_ = tf.get_variable("marg_vec")
    # If beta[min_indx+1] < 0, then alpha becomes 0 and joins remaining vec,
    # else alpha becomes C and joins error vectors
    _ = tf.cond(
          beta_[min_indx+1] < 0, 
          lambda: add_to_rem(scope, params["eps"], 
            tf.constant(0), marg_vec_[min_indx]),
          lambda: add_to_err(scope, tf.constant(0), marg_vec_[min_indx])
          )
    # Remove vector from R_ and marg_vec_
    _ = rem_from_marg_sup(scope, params["kernel"], min_indx)

def update_g(g_tens, gamma_tens, min_alpha):
  iter_tensor = tf.range(tf.shape(g_tens)[0])
  return tf.map_fn(
    lambda i: g_tens[i] + min_alpha*gamma_tens[i],
    iter_tensor)

def free_move(scope, params):
  with tf.variable_scope(scope, reuse=True):
    x_all_ = tf.get_variable("x_all")
    y_all_ = tf.get_variable("y_all")
    marg_vec_ = tf.get_variable("marg_vec")
    err_vec_ = tf.get_variable("err_vec")
    rem_vec_ = tf.get_variable("rem_vec")
    alpha_marg_ = tf.get_variable("alpha_marg")
    g_err_ = tf.get_variable("g_err")
    g_rem_ = tf.get_variable("g_rem")
    R_ = tf.get_variable("R")

    def move_vec(loc_index, vec_set, g_vec, alpha_c):
      glob_index = vec_set[loc_index]
      x_c = x_all_[glob_index]
      y_c = y_all_[glob_index]
      Q = tf.constant(
        [[0, y_c],
        [y_c, y_c*y_c*params["kernel"](x_c, x_c)]])
      _ = tf.assign(R_, tf.matrix_inverse(Q))

      _ = tf.assign(marg_vec_, tf.concat([marg_vec_, [glob_index]], 0))
      # add alpha to alpha_marg_
      _ = tf.assign(alpha_marg_, tf.concat([alpha_marg_, [alpha_c]], 0))

      rem_from(vec_set, g_vec, loc_index)



    zero_indx = get_index(g_err_, 0.)
    _ = tf.cond(zero_indx < tf.shape(g_err_)[0],
      lambda: move_vec(zero_indx, err_vec_, g_err_, alpha_c=5.),
      lambda: -1)

    zero_indx = get_index(g_rem_, 0.)
    _ = tf.cond(
      tf.logical_and(
                  tf.equal(tf.shape(marg_vec_)[0], 0),
                  tf.less(zero_indx, tf.shape(rem_vec_)[0])),
      lambda: move_vec(zero_indx, rem_vec_, g_rem_, alpha_c=0.),
      lambda: -1)


def handle_inf_matrix(scope, params):
  with tf.variable_scope(scope, reuse=True):
    err_vec_ = tf.get_variable("err_vec")
    rem_vec_ = tf.get_variable("rem_vec")
    g_err_ = tf.get_variable("g_err")
    g_rem_ = tf.get_variable("g_rem")

    # free move if g = 0
    _ = free_move(scope, params)


# returns alpha_c and g_c
def mini_iter(scope, params, i):
  with tf.variable_scope(scope, reuse=True):
    x_all_ = tf.get_variable("x_all")
    y_all_ = tf.get_variable("y_all")
    n = tf.get_variable("n").read_value()
    marg_vec_ = tf.get_variable("marg_vec")
    err_vec_ = tf.get_variable("err_vec")
    rem_vec_ = tf.get_variable("rem_vec")
    alpha_marg_ = tf.get_variable("alpha_marg")
    g_err_ = tf.get_variable("g_err")
    g_rem_ = tf.get_variable("g_rem")
    g_c_ = tf.get_variable("g_c")
    alpha_c_ = tf.get_variable("alpha_c")
    b_ = tf.get_variable("b")
    R_ = tf.get_variable("R")

    # Handle case of matrix containing inf
    _ = tf.cond(
      tf.equal(tf.shape(marg_vec_)[0], 0), 
      lambda: tf.assign(R_, tf.constant([[INF]])),
      lambda: tf.constant(-1))
    # Get beta and gamma necessary for the iteration
    beta_ = get_beta(scope, params["kernel"], n-1)
    gamma_err_ = get_gamma(scope, params["kernel"], err_vec_, beta_)
    gamma_rem_ = get_gamma(scope, params["kernel"], rem_vec_, beta_)
    gamma_c_ = get_gamma(scope, params["kernel"], tf.constant([n-1]), beta_)

    # Book-keeping -
    # For margin support vectors, if Beta_s > 0, alpha_c can go to C or 
    # if Beta_s < 0, alpha_s can go to 0, causing a transition in state
    iter_tensor = tf.range(tf.shape(marg_vec_)[0])
    marg_trans_alpha = tf.map_fn(
      lambda i: tf.cond(beta_[i+1] < 0, 
        lambda: -1*alpha_marg_[i]/beta_[i+1],
        lambda: tf.cond(beta_[i+1] > 0,
          lambda: (params["C"]-alpha_marg_[i])/beta_[i+1],
          lambda: tf.constant(float("Inf"))
          )
        ),
      iter_tensor)

    # For error support vectors, if Gamma_i > 0, then g_i increases to 0
    # and causes state change
    iter_tensor = tf.range(tf.shape(err_vec_)[0])
    err_trans_alpha = tf.map_fn(
      lambda i: tf.cond(gamma_err_[i] > 0,
        lambda: -1*g_err_[i]/gamma_err_[i],
        lambda: tf.constant(float("Inf"))
        ),
      iter_tensor)

    # For the remaining vectors, if Gamma_i < 0, then g_i decreases to 0
    iter_tensor = tf.range(tf.shape(rem_vec_)[0])
    rem_trans_alpha = tf.map_fn(
      lambda i: tf.cond(gamma_rem_[i] < 0,
        lambda: -1*g_rem_[i]/gamma_rem_[i],
        lambda: tf.constant(float("Inf"))
        ),
      iter_tensor)

    # For candidate vector, if gamma_i > 0, then g_i increases to 0
    # or alpha_c increases to C_svm
    candidate_trans_alpha = tf.min(
      tf.cond(gamma_c_[0] > 0,
        lambda: -1*g_c_/gamma_c_[0],
        lambda: tf.constant(float("Inf"))),
      params["C"] - alpha_c_.read_value())

    # Compare the min and do the necessary transitions
    min_alpha = tf.min(tf.concat(
      [marg_trans_alpha, err_trans_alpha, rem_trans_alpha, 
      candidate_trans_alpha]))

    # CALCULATING NEW VALUES
    # Calculate new b
    b_new = b_ + beta_[0]*min_alpha
    _ = tf.assign(b_, b_new)
    # Calculate new values of alpha for marg vec
    iter_tensor = tf.range(tf.shape(alpha_marg_)[0])
    alpha_marg_new = tf.map_fn(
      lambda i: alpha_marg_[i] + beta_[i+1]*min_alpha,
      iter_tensor)
    _ = tf.assign(alpha_marg_, alpha_marg_new)
    # Calculate new value of alpha_c
    _ = tf.assign(alpha_c_, alpha_c_ + min_alpha)

    # Calculate new values of g_all
    _ = tf.assign(g_err_, update_g(g_err_, gamma_err_, min_alpha))
    _ = tf.assign(g_rem_, update_g(g_rem_, gamma_rem_, min_alpha))
    _ = tf.assign(g_c_, g_c_ + gamma_c_[0]*min_alpha)

    # MOVING VECTORS
    # Check if min is in marg sup vectors
    min_indx = get_index(marg_trans_alpha, min_alpha)
    _ = tf.cond(
      min_indx < tf.shape(marg_vec_)[0],
      lambda: handle_marg(scope, params, min_indx, beta_),
      lambda: tf.constant(-1))

    # Check if min is in err sup vectors
    def handle_err(loc_index):
      add_to_marg_supp(scope, params["kernel"], err_vec_[loc_index],
        gamma_err_[loc_index], params["C"])
      rem_from(err_vec_, g_err_, loc_index)

    min_indx = get_index(err_trans_alpha, min_alpha)
    _ = tf.cond(
      min_indx < tf.shape(err_vec_)[0],
      lambda: handle_err(min_indx),
      lambda: tf.constant(-1))
    
    # Check if min is in remaining vectors
    def handle_rem(loc_index):
      add_to_marg_supp(scope, params["kernel"], rem_vec_[loc_index],
        gamma_rem_[loc_index], 0.)
      rem_from(rem_vec_, g_rem_, loc_index)

    min_indx = get_index(rem_trans_alpha, min_alpha)
    _ = tf.cond(
      min_indx < tf.shape(rem_vec_)[0],
      lambda: handle_rem(min_indx),
      lambda: tf.constant(-1))
    # Need to add the candidate vector here as I need g and gamma values
    _ = tf.cond(
      termn_condn(params["C"], alpha_c_, g_c_),
      lambda: tf.cond(
        tf.equal(g_c_, 0),
        lambda: add_to_marg_supp(scope, params["kernel"],
          n-1, gamma_c_[0], alpha_c_),
        lambda: add_to_err(scope, g_c_, n-1)),
      lambda: -1)

    return (alpha_c_.read_value(), g_c_.read_value())

def fit_point(scope, params):
  with tf.variable_scope(scope, reuse=True):
    alpha_c = tf.get_variable("alpha_c").read_value()
    n = tf.get_variable("n").read_value()
    g_c = tf.get_variable("g_c").read_value()

    fin_alpha, fin_g = tf.while_loop(
      cond=lambda i, j: tf.logical_not(termn_condn(params["C"], i, j)),
      body=lambda i, j: mini_iter(scope, params, i, j), 
      loop_vars=[alpha_c, g_c],
      parallel_iterations=1,
      back_prop=False,
      )

# Allows incremental training of the model
def svm_model_fn(
  features, # Receives one new feature from the input_fn 
  labels,   # Label corresponding to the feature from the input_fn
  mode,     # An instance of tf.estimator.ModeKeys
  params):  # Additional configuration

  with tf.variable_scope("svm_model") as scope:
    # Variable creation
    # count of the number of data points seen
    n_ = tf.get_variable("n", initializer=tf.constant(0), trainable=False,
      dtype=tf.int32)
    # offset
    b_ = tf.get_variable("b", initializer=tf.constant(0.), trainable=False,)
    # Variable tensor - containing all the data points and labels
    x_all_ = tf.get_variable("x_all", shape=[0, tf.shape(features["x"])[0]])
    y_all_ = tf.get_variable("y_all", shape=[0])
    
    # Variable tensor - representing the Margin Support vector indices
    marg_vec_ = tf.get_variable("marg_vec", 
      shape=[0, tf.shape(features["x"])[0]], dtype=tf.int32)
    # Variable tensor - representing the Error Support vector indices
    err_vec_ = tf.get_variable("err_vec", 
      shape=[0, tf.shape(features["x"])[0]], dtype=tf.int32)
    # Variable tensor - representing the Remaining vector indices
    rem_vec_ = tf.get_variable("rem_vec",
      shape=[0, tf.shape(features["x"])[0]], dtype=tf.int32) 

    # Variable denoting the alpha of new candidate
    alpha_c_ = tf.get_variable("alpha_c", initializer=0)
    # Variable denoting g of error support vectors
    g_err_ = tf.get_variable("g_err", shape=[0])
    # Variable denoting g of remaining vectors
    g_rem_ = tf.get_variable("g_rem", shape=[0])
    # Variable denoting the g of candidate vector
    g_c_ = tf.get_variable("g_c", shape=0)

    # Variable for alpha of margin vectors
    alpha_marg_ = tf.get_variable("alpha_marg", [0])
    # Variable for the inverse Jacobian matrix R - initially Inf
    R_ = tf.get_variable("R", initializer=tf.constant([[float("Inf")]]))

    # Starting the algorithm
    x_ = features["x"]
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode, predictions={"x":x_,
        "y": labels})


    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
        mode, loss=return_w_b(scope, params), eval_metric_ops=metrics)


    if mode == tf.estimator.ModeKeys.TRAIN:
      y_ = labels
      # First add the new input to x_all_
      _ = tf.assign(x_all_, tf.concat([x_all_, tf.reshape(x_, [1,2])], 0), 
        validate_shape=False)
      _ = tf.assign(y_all_, tf.concat([y_all_, labels], 0), 
        validate_shape=False)
      # Update count of data points seen
      n = tf.assign_add(n_, 1)
      # First calculate g_c
      return tf.Print(g_c_, [y_, y_all_.read_value(), x_all_.read_value(), x_])
      g_c = tf.assign(g_c_, calc_g(scope, params, x_, y_),
        validate_shape=False)
      # reset alpha_c
      _ = tf.assign(alpha_c_, 0)
      # Add condition to update remaining vectors if g_c > 0
      _ = tf.cond(
        tf.greater(g_c, 0),
        lambda: add_to_rem(scope, params["eps"], g_c, n-1),
        lambda: fit_point(scope, params))
      return -1

def svm_train(x_, y_):
  x_all_ = tf.get_variable("x_all")
  y_all_ = tf.get_variable("y_all")
  n_ = tf.get_variable("n", dtype=tf.int32)
  
  _ = tf.assign(x_all_, 
    tf.concat([x_all_, tf.reshape(x_, [1, tf.shape(x_all_)[1]])], 0),
    validate_shape=False)
  _ =  tf.assign(y_all_,
    tf.concat([y_all_, tf.reshape(y_, [1])], 0),
    validate_shape=False)
  _ = tf.assign_add(n_, 1)
  return tf.constant(1)
  
def svm_model(
  features, # Receives one new feature from the input_fn 
  labels,   # Label corresponding to the feature from the input_fn
  mode,     # An instance of tf.estimator.ModeKeys
  params):
  with tf.variable_scope("svm_model") as scope:
    # Variable creation
    # count of the number of data points seen
    n_ = tf.get_variable("n", initializer=tf.constant(0), trainable=False,
      dtype=tf.int32)
    # offset
    b_ = tf.get_variable("b", initializer=tf.constant(0.), trainable=False,)
    # Variable tensor - containing all the data points and labels
    x_all_ = tf.get_variable("x_all", shape=[0, model_params["x_shape"]])
    y_all_ = tf.get_variable("y_all", shape=[0])

    x_ = tf.to_float(features["x"])

    if mode == tf.estimator.ModeKeys.TRAIN:
      y_ = tf.to_float(labels)
      scope.reuse_variables()
      svm_train_op = svm_train(x_, y_)
      return tf.estimator.EstimatorSpec(mode, loss=tf.constant(1), train_op=svm_train_op)

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode, predictions={"x":x_all_,
        "y": y_all_})


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



def main():
  from data_loader import extract_data

  train_file = "data_1.csv"
  x_all, y_all = extract_data(train_file, "csv")
  print(y_all[:2])
  input()
  model_params["x_shape"] = x_all.shape[1]
  classifier = tf.estimator.Estimator(
        model_fn=svm_model,
        params=model_params)

  # with tf.Session() as sess:
    # print(sess.run(custom_input_fn(x_all[:2], y_all[:2])))
    # print(sess.run(custom_input_fn(x_all[:2], y_all[:2])))
  classifier.train(
    input_fn=lambda: custom_input_fn(x_all[:2], y_all[:2]))
  
  # predictions = classifier.predict(
  #   input_fn=lambda:custom_input_fn(x_all[:2], 
  #     y_all[:2]))

  # for dontcare in predictions:
    # print(dontcare)

main()
