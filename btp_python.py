# Store the Q calculation of each point with Support vectors
# as they could be used for add supp to R

# Online SVM - Cauwenberghs algorithm
import hashlib

import numpy as np

from matplotlib import colors
from matplotlib import pyplot as plt
from scipy import sparse
from time import time

from data_loader import extract_data

float_threshold = 10**-8
INF = 10**10
# float('Inf')
eps2 = 2*(10**-6)

def is_close(a, b):
  if abs(a-b) < float_threshold:
    return True

# def memoize(kernel_fn):
#   cache = dict()

#   def memoized_fn(x1, x2):
#     x1_h = x1.view(np.uint8)
#     x1_h = hashlib.sha1(x1_h).hexdigest()

#     x2_h = x2.view(np.uint8)
#     x2_h = hashlib.sha1(x2_h).hexdigest()
#     if (x1_h, x2_h) in cache:
#       return cache[(x1_h,x2_h)]
#     else:
#       result = kernel_fn(x1, x2)
#       cache[(x1_h, x2_h)] = result
#       return result

#   return memoized_fn

# @memoize
def kernel(x1, x2):
  # apply the kernel
  return np.dot(x1, x2)

# @cache this
  # .astype(np.float32) + eps


class SVM_Online:
  def __init__(self, X=None, y=None, filename="data_1.csv",
      file_type="csv", C_svm=5, reserve_threshold=-1):

    if X is None and y is None:
      self.file_type = file_type
      # All data
      self.x_all, self.y_all = extract_data(filename, file_type)
      # self.x_all = self.x_all[:500]
      # self.y_all = self.y_all[:500]


    else:
      self.x_all = X
      self.y_all = y
      self.file_type = None
    # Classes of vector - stores indices of vectors
    self.Margin_v = []
    self.Remain_v = []
    self.Error_v = []

    # Hessian of margin support vectors
    self.Q_ = np.array([[0]])
    # Inverse Hessian
    self.R_ = np.array([[-float("Inf")]])
    
    # KT conditions to be stored
    # g = y*f(x) - 1
    self.g_all = []
    # Lagrange multiplier
    self.alpha_all = []

    # offset
    self.b_ = 0.
    # Regularization parameter
    self.C_svm = C_svm
    # points seen
    self.n = 0
    # Reserve threshold to drop farther Remain vectors
    self.reserve_threshold = reserve_threshold
    # DEBUG purpose - iteeration count
    self.tot_iter = 0
    # NUMERIC INSTABILITY CORRECTION
    self.cor_eps = 2*(10**-6)/self.C_svm
    # print(self.cor_eps)
    # input()

  def calc_Q(self, x1, y1, x2, y2):
    return np.array(y1*y2*kernel(x1, x2)) + self.cor_eps

  def add_point(self, x_c, y_c):
    self.y_all.append(y_c)
    
    if self.file_type == "svmlight":
      if type(x_c) == np.ndarray:
        x_c = sparse.csr_matrix(x_c)
      if type(x_c) != sparse.csr_matrix:
        return False

    else:
      if type(x_c) != np.ndarray:
        return False
    self.x_all.append(x_c)

    self.__learn(self.get_x(self.n), y_c)
    return True

# Cache this - useful in case of svmlight
  def get_x(self, indx):
    x = self.x_all[indx]
    if self.file_type == "svmlight" and type(x) != np.ndarray:
      # converting csr matrix element to np arrayof shape (n_features,)
      x = x.toarray()[0]
    return x

  def calc_f(self, x_c):
    ans = self.b_
    for indx in self.Margin_v:
      ans += (self.alpha_all[indx])*(self.y_all[indx])*kernel(self.get_x(indx), x_c)
    # Error vectors have non-zero alpha
    for indx in self.Error_v:
      ans += (self.alpha_all[indx])*(self.y_all[indx])*kernel(self.get_x(indx), x_c)
    return ans

  def predict(self, x_c):
    ans = self.calc_f(x_c)
    if ans >= 0:
      return 1
    else:
      return -1

  def add_support_R(self, elem_index):
    x_c = self.get_x(elem_index)
    y_c = self.y_all[elem_index]

    Q_cs = np.zeros((len(self.Margin_v)+1, 1))
    Q_cs[0][0] = y_c
    for i in range(len(self.Margin_v)):
      indx = self.Margin_v[i]
      Q_cs[i+1][0] = self.calc_Q(self.get_x(indx), self.y_all[indx], x_c, y_c)

    Q_cs = np.append(Q_cs, [[self.calc_Q(x_c, y_c, x_c, y_c)]], axis=0)
    # Updating Q
    Q_temp = np.concatenate( (self.Q_, Q_cs.transpose()[:,:-1]), axis=0)
    self.Q_ = np.concatenate( (Q_temp, Q_cs), axis=1)
    # To calculate R:
    if len(self.Margin_v) == 0:
      # print("HREH")
      self.R_ = np.linalg.inv(self.Q_)
    else:
      R_ = self.R_
      Q_cs = Q_cs[:-1, :]
      beta = np.dot(-1*R_, Q_cs)

      # calc pivot/gamma
      gamma_c = beta[0][0] * y_c + self.calc_Q(x_c, y_c, x_c, y_c)
      for i in range(len(self.Margin_v)):
        indx = self.Margin_v[i]
        gamma_c += self.calc_Q(self.get_x(indx), self.y_all[indx], x_c, y_c)*beta[i+1][0]
      # pivot check
      
      if gamma_c < self.cor_eps:
        print("Training error: ", gamma_c)
        gamma_c = self.cor_eps
      # Calc R
      beta = np.append(beta, [[1.]], axis=0)
      n = self.R_.shape[0]
      R_ = np.zeros((n+1, n+1))
      R_[:-1, :-1] = self.R_
      # print(beta)
      # print(gamma_c)
      self.R_ = R_ + 1/gamma_c*beta*beta.transpose()
    self.stabilize_R()


  def rem_support_R(self, elem_index):
    # print("Removing...")
    R_mat = self.R_
    k = self.Margin_v.index(elem_index) + 1
    if R_mat[k][k] == 0:
      # print("Training error - R_kk is zero")
      R_mat[k][k] = 10**-8
    if R_mat[k][k] != 0:
      for i in range((R_mat.shape[0])):
        for j in range((R_mat.shape[1])):
          if(i!=k and j!= k):
            R_mat[i][j] -= 1/R_mat[k][k]*R_mat[i][k]*R_mat[k][j]
      R_mat = np.delete(R_mat, k, axis=0)
      R_mat = np.delete(R_mat, k, axis=1)

    self.Q_ = np.delete(self.Q_, k, axis=0)
    self.Q_ = np.delete(self.Q_, k, axis=1)
    # self.R_ = R_mat
    if self.Q_.shape[0] == 1:
      self.R_ = np.array([[-INF]])
    else:
      self.R_ = R_mat
    self.stabilize_R()
    # numerical stability correction for R
    
  def stabilize_R(self):
    R_ = self.R_
    # .astype(np.float32)
    Q_ = self.Q_
    # .astype(np.float32)
    R_tr = R_.transpose()
    self.R_ = R_ + R_tr - np.dot(np.dot(R_, Q_), R_tr)


  def calc_g(self, x_c, y_c):
    # f(x) = (Sum alpha_j*y_j*kernel(x_c, x_j)) + b
    # g(x) = y_c*f(x_c) - 1
    ans = self.calc_f(x_c)
    return (ans*y_c - 1)

  def get_beta_gamma(self, x_c, y_c):
    n = self.n
    Q_cs = np.zeros((len(self.Margin_v)+1, 1))
    Q_cs[0][0] = y_c
    # -------- TO DO ---------------
    # To make this vector operation
    for i in range(len(self.Margin_v)):
      indx = self.Margin_v[i]
      Q_cs[i+1][0] = self.calc_Q(self.get_x(indx), self.y_all[indx], x_c, y_c)

    if self.R_.shape[0] == 1:
      self.R_[0][0] = -INF
    
    beta_sup = np.dot(-1*(self.R_), Q_cs)

    beta_ = [0 for i in range(n)]
    # for Margin support vectors
    for i in range(len(self.Margin_v)):
      indx = self.Margin_v[i]
      beta_[indx] = beta_sup[i+1][0]

    # calculate gamma
    gamma_ = [0 for i in range(n)]
    for i in range(n):
      if i not in self.Margin_v:
        gamma_[i] = self.calc_Q(self.get_x(i), self.y_all[i], x_c, y_c)
        gamma_[i] += beta_sup[0][0]*(self.y_all[i])

        for j, indx in enumerate(self.Margin_v):
          gamma_[i] += (beta_sup[j+1][0] * self.calc_Q(
            self.get_x(indx), self.y_all[indx], self.get_x(i), self.y_all[i]))

    return beta_, gamma_, beta_sup 

  def dec_bdry(self, start=0, last=1):
    if self.x_all.shape[1] != 2:
      return
    # get a1 and a2 for a1x+a2y+b=0
    w = np.zeros((2,))
    for _, indx in enumerate(self.Margin_v + self.Error_v):
      w += (self.y_all[indx])*(self.alpha_all[indx])*(self.get_x(indx))

    plt.scatter(self.x_all[:self.n, 0], self.x_all[:self.n, 1], c=self.y_all[:self.n],
      cmap=colors.ListedColormap(['orange', 'red', 'green']))
    plt.plot()

    xl = [start, last]
    yl = []
    for xx in xl:
      yl.append((-self.b_ - w[0]*xx)/w[1])
    plt.plot(xl, yl)
    plt.show()


  def train_all(self):
    t1 = time()
    num_pts = 0
    if self.file_type == "svmlight":
      num_pts = self.x_all.shape[0]
      print(num_pts)
    else:
      num_pts = len(self.x_all)
    for ii in range(num_pts):
      if (ii+1)%200 == 0:
        print("AT: ", ii)
        print("Margin_v: ", len(self.Margin_v))
        print("Iterations: ", self.tot_iter)
      self.__learn(self.get_x(ii), self.y_all[ii])

    print("Time taken: ", time()-t1)
    # print("----------------------DONE---------------------------")
    print("Iterations : ", self.tot_iter)
    # print("SUP_V: ", self.Margin_v)
    # print("Err_v: ", self.Error_v)
    # print(self.g_all, self.alpha_all)
    # self.dec_bdry()

  def remove_pt(self, rem_indx):
    self.x_all.remove(rem_indx)
    self.y_all.remove(rem_indx)
    self.alpha_all.remove(rem_indx)
    self.g_all.remove(rem_indx)

  # Leave-one-out decremental unlearning
  # gives False if classification error occurs
  def unlearn(self, rem_indx):
    x_c = self.get_x(rem_indx)
    y_c = self.y_all[rem_indx]
    # If point is not Suppport Vector, there is nothing ot do
    if self.g_all[rem_indx] not in (Margin_v+Error_v):
      self.remove_pt(rem_indx)
      return True

    else:
      if self.g_all[rem_indx] < -1:
        self.remove_pt(rem_indx)
        return False

    classification_flag = True
    while True:
      # Keep decrementing alpha
      # IF candidate's g reaches -1 first, classify as wrong
      if self.g_all[rem_indx] <= -1:
        classification_flag = False
        break
      # If candidate's alpha becomes zero, classify as right
      if is_close(self.alpha_all[rem_indx], 0):
        classification_flag = True
        break

      beta_, gamma_, beta_sup = self.get_beta_gamma(x_c, y_c)
      # beta_ for candidate is 1
      beta_[rem_indx] = 1

      transition_alpha = [-INF for i in range(n)]
      # book-keeping
      # for error_vectors: g_i <= 0 and become margin vectors at equality
      for _, indx in enumerate(self.Error_v):
        if(gamma_[indx] < 0):
          transition_alpha[indx] = -(self.g_all[indx])/gamma_[indx]
      # for rem vectors: g_i >=0 and become margin vectors at equality
      for _, indx in enumerate(self.Remain_v):
        if(gamma_[indx] > 0):
          transition_alpha[indx] = self.g_all[indx]/(-gamma_[indx])

      # for Margin Support vectors
      for i, indx in enumerate(self.Margin_v):
        # Alpha increases to C_svm - transition to Error Vectors
        if beta_sup[i+1][0] < 0:
          transition_alpha[indx] = ((self.C_svm - self.alpha_all[indx])/
                                        beta_sup[i+1][0])
        # Alpha decreases to 0 - transition to Remaining Vectors
        elif beta_sup[i+1][0] > 0:
          transition_alpha[indx] = (-self.alpha_all[indx]/ beta_sup[i+1][0])
      
      # for candidate vector - transition to Remaining vector
      # alpha becomes 0
      transition_alpha[n-1] = -self.alpha_all[rem_indx]
      # transition of candidate vector to Wrong Classified Vector -
      # when g becomes -1
      if (gamma_[n-1] > 0):
        transition_alpha[rem_indx] = max(transition_alpha[rem_indx], 
                                    (-1 - self.g_all[n-1])/gamma_[rem_indx])

      # Should make this a list of elements
      max_alpha = max(transition_alpha)
      transition_vectors = []

      for i in range(n):
        if transition_alpha[i] == min_alpha:
          transition_vectors.append(i)
          # ------ REMOVE THIS
          break

      flag = False
      for elem_index in transition_vectors:
        if elem_index in self.Margin_v:
          # remove elem_index from R
          self.rem_support_R(elem_index)
          self.Margin_v.remove(elem_index)
          if beta_[elem_index] < 0:
            self.Error_v.append(elem_index)
          else:
            self.Remain_v.append(elem_index)

        elif elem_index in self.Error_v:
          # add elem_index to Margin vector
          self.add_support_R(elem_index)
          self.Error_v.remove(elem_index)
          self.Margin_v.append(elem_index)

        elif elem_index in self.Remain_v:
          # add elem_index to Margin vector
          self.add_support_R(elem_index)
          self.Remain_v.remove(elem_index)
          self.Margin_v.append(elem_index)

        elif elem_index == rem_indx:
          if transition_alpha[rem_indx] == (-self.alpha_all[n-1]):
            classification_flag = True
          else:
            classification_flag = False
          flag = True

      # recompute alpha and g
      for i in range(n):
        self.alpha_all[i] += max_alpha * beta_[i]
        self.g_all[i] += max_alpha * gamma_[i]

      # recompute b
      self.b_ += beta_sup[0][0] * max_alpha
      if (flag):
        break
    # seems inefficient
    self.remove_pt(rem_indx)
    return classification_flag


  def handle_empty_marg_set(self):
    # print("Inside empty handler")
    n = self.n
    x_c = self.x_all[n-1]
    y_c = self.y_all[n-1]
    # calculate min possible b from candidate vector
    min_del_b = -self.g_all[n-1]/y_c
    # calc min possible increment in b from remain vectors
    for indx in self.Error_v:
      if self.y_all[indx]*y_c > 0:
        del_b = -self.g_all[indx]/self.y_all[indx]
        if min_del_b >= 0:
          min_del_b = min(min_del_b, del_b)
        else:
          # It should actually be -1* min(abs(min_del_b), abs(del_b)))
          min_del_b = max(min_del_b, del_b)

    for indx in self.Remain_v:
      if self.y_all[indx]*y_c < 0:
        del_b = -self.g_all[indx]/self.y_all[indx]
        if min_del_b >= 0:
          min_del_b = min(min_del_b, del_b)
        else:
          # It should actually be -1* min(abs(min_del_b), abs(del_b)))
          min_del_b = max(min_del_b, del_b)
    # update b
    self.b_ += min_del_b
    for indx in self.Error_v + self.Remain_v + [n-1]:
      self.g_all[indx] += self.y_all[indx]*min_del_b
    # Move zero g vectors to margin support vector set if they satisy the direction
    # of movement 
    # if self.g_all[n-1] == 0:
    #   self.add_support_R(n-1)
    #   print("Free moved ", n-1)

    for indx in self.Remain_v:
      if self.y_all[indx]*y_c < 0 and self.g_all[indx] <= 0:
        self.add_support_R(indx)
        self.Margin_v.append(indx)
        self.Remain_v.remove(indx)

    for indx in self.Error_v:
      if self.y_all[indx]*y_c > 0 and self.g_all[indx] >= 0:
        self.add_support_R(indx)
        self.Margin_v.append(indx)
        self.Error_v.remove(indx)

# Not to be called directly
  def __learn(self, x_c, y_c):
    # Assumption: the point has already been added
    self.n += 1
    n = self.n
    # new candidate starts with zero alpha
    self.alpha_all.append(0)

    g_c = self.calc_g(x_c, y_c)
    self.g_all.append(g_c)
    # Add candidate to Remaining vector      
    if (g_c > 0):
      # add threshold check to add into the list
      self.Remain_v.append(n-1)

    else:
      # Keep performing steps until x_c becomes Margin or Error Support Vector
      while True:
        # DEBUG: Counting number of iterations
        self.tot_iter += 1
        # Handling zero support vectors case
        if len(self.Margin_v) == 0:
          self.handle_empty_marg_set()

        beta_, gamma_, beta_sup = self.get_beta_gamma(x_c, y_c)
        # 1 for candidate - inconseq
        beta_[n-1] = 1

        transition_alpha = [INF for i in range(n)]
        # book-keeping
        # for error_vectors: g_i <= 0 and become margin vectors at equality
        for _, indx in enumerate(self.Error_v):
          if(gamma_[indx] > 0):
            transition_alpha[indx] = -(self.g_all[indx])/gamma_[indx]
        # for rem vectors: g_i >=0 and become margin vectors at equality
        for _, indx in enumerate(self.Remain_v):
          if(gamma_[indx] < 0):
            transition_alpha[indx] = self.g_all[indx]/(-gamma_[indx])

        # for Margin Support vectors
        for i, indx in enumerate(self.Margin_v):
          # Alpha increases to C_svm - transition to Error Vectors
          if beta_sup[i+1][0] > 0:
            transition_alpha[indx] = ((self.C_svm - self.alpha_all[indx])/
                                          beta_sup[i+1][0])
          # Alpha decreases to 0 - transition to Remaining Vectors
          elif beta_sup[i+1][0] < 0:
            transition_alpha[indx] = (-self.alpha_all[indx]/ beta_sup[i+1][0])
        
        # for candidate vector - transition to Error vector
        # alpha becomes C_svm
        transition_alpha[n-1] = self.C_svm - self.alpha_all[n-1]
        # transition of candidate vector to Support Vector -
        # when g becomes zero
        if (gamma_[n-1] > 0):
          transition_alpha[n-1] = min(transition_alpha[n-1], 
                                      -(self.g_all[n-1])/gamma_[n-1])

        # Should make this a list of elements
        min_alpha = min(transition_alpha)
        # print("\nTRANS: ", transition_alpha, self.Margin_v, self.Error_v,
        #   min_alpha)
        # print("BETA: ", beta_sup)
        # print("MARG: ", self.Margin_v, self.Error_v)
        # print("GAMMA: ", gamma_)
        # print("R: ", self.R_)
        transition_vectors = []
        for i in range(n):
          if transition_alpha[i] == min_alpha:
            transition_vectors.append(i)
            # ------ REMOVE THIS
            break

        # DEBUG
        if n >0:
          print("n: ", n)
          print("MArg: ", self.Margin_v)
          print("Err: ", self.Error_v)
          print("Remain: ", self.Remain_v)
          print("G: ", self.g_all)
          print("ALL ALPHA: ", self.alpha_all)
          print("gamma_: ", gamma_)
          print("TRANS: ", transition_alpha)
          print("VECS: ", transition_vectors)
          print("R: ", self.R_)
          input()

        # print(transition_vectors)
          # --------TO DO---------------------
          # When adding the element to Margin_v, pass beta_sup to prevent recalculation
          # elif elem_index == n-1:
          #   if transition_alpha[n-1] == (self.C_svm - self.alpha_all[n-1]):
          #     self.Error_v.append(elem_index)
          #   else:
          #     self.add_support_R(elem_index)
          #     self.Margin_v.append(elem_index)
          #   flag = True

        # recompute alpha and g
        for i in range(n):
          self.alpha_all[i] += min_alpha * beta_[i]
          self.g_all[i] += min_alpha * gamma_[i]

        # recompute b
        self.b_ += beta_sup[0][0] * min_alpha
        # Book-keeping
        for elem_index in transition_vectors:
          if elem_index in self.Margin_v:
            # remove elem_index from R
            self.rem_support_R(elem_index)
            self.Margin_v.remove(elem_index)
            if beta_[elem_index] > 0:
              self.Error_v.append(elem_index)
            else:
              self.Remain_v.append(elem_index)

          elif elem_index in self.Error_v:
            # add elem_index to Margin vector
            self.add_support_R(elem_index)
            self.Error_v.remove(elem_index)
            self.Margin_v.append(elem_index)

          elif elem_index in self.Remain_v:
            # add elem_index to Margin vector
            self.add_support_R(elem_index)
            self.Remain_v.remove(elem_index)
            self.Margin_v.append(elem_index)

        # Break conditions
        # IF candidate vector g is zero - Margin vector
        # Numerical floating point check
        if is_close(self.g_all[n-1], 0):
          self.add_support_R(n-1)
          self.Margin_v.append(n-1)
          break
        # If candidate vector alpha is equal to C_svm - Error vector
        # print("HERE: ", self.alpha_all[n-1], is_close(self.alpha_all[n-1], 5))
        if (is_close(self.alpha_all[n-1], self.C_svm)):
          self.Error_v.append(n-1)
          break
    # print("After\nMargin Vectors: %s \n Error vectors: %s" %(self.Margin_v,
    #   self.Error_v,))
    # print(self.x_all[:n], self.y_all[:n])
    # print("n: ", n)
    # print("ITER: ", self.tot_iter)
    # print("b: ", self.b_)
    # print("g: ", g_c)
    # print("alpha: ", self.alpha_all)
    # input()
    # print("\n\n")



if __name__ == "__main__":
  svm = SVM_Online(filename="data_1.csv", file_type="csv", C_svm=5.)
  svm.train_all()
  print(len(svm.Margin_v))
  print(len(svm.Error_v), len(svm.Remain_v))
  svm.dec_bdry()
  # print(svm.Margin_v)
  # print(len(svm.Error_v))
  # x_test, y_test = extract_data("diabetes.mat", "mat")
  # x_test = x_test[576:]
  # y_test = y_test[576:]
  # cor = 0
  # mismatch = 0
  # for i in range(x_test.shape[0]):
  #     if svm.predict(x_test[i]) == (y_test[i]):
  #         cor += 1
  #     if svm.predict(x_test[i]) != (y_test[i]):
  #         mismatch += 1
  # print(cor, mismatch)

