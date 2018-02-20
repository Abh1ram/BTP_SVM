# handling data from different inputs
import numpy as np

from scipy.io import arff
from sklearn.datasets import load_svmlight_file

def extract_data(filename, file_type):
  func_mapper = {
        "csv"       : get_csv_data,
        "arff"      : get_arff_data,
        "svmlight"  : get_svmlight_data,
  }
  return func_mapper[file_type](filename)

def get_arff_data(filename="iris.arff"):

  data, _ = arff.loadarff(filename)
  # class labels 
  class_name = data.dtype.names[-1]
  # all labels
  uniq_terms, y_all = np.unique(data[class_name], return_inverse=True)

  if len(uniq_terms) > 2:
    # Raise Exception - more than two class labels
    return None, None

  x_temp = data[list(data.dtype.names[:-1])]

  float_flag = True
  for i in range(len(x_temp.dtype)):
    if x_temp.dtype[i] != float:
      float_flag = False
      break

  # COntains string data type
  if not float_flag:
    return None, None

  x_all = x_temp.view(float).reshape(-1, len(x_temp.dtype))

  # To store dtypes for future need??
  # print(y_all*2 - 1, x_all, data)
  return x_all, y_all*2 - 1


def get_csv_data(filename="data_1.csv"):

  out = np.loadtxt(filename, delimiter=',');
  # Arrays to hold the labels and feature vectors.
  labels = out[:,0]
  fvecs = out[:,1:]
  # Check the number of labels
  # --------- TO DO --------- and their values
  uniq_vals = np.unique(labels) 
  if len(uniq_vals) > 2:
    # throw Exception
    return None, None

  # Check if labels are 0,1 instead of 1, -1
  if 0 in uniq_vals:
    labels = labels*2 - 1
  return fvecs, labels


# from sklearn.externals.joblib import Memory
# mem = Memory("./mycache")

# @mem.cache
def get_svmlight_data(filename="test_svm_light"):
    data = load_svmlight_file(filename)
    return data[0], data[1]

get_csv_data("data_1.csv")