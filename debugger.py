import pickle
import sys

import numpy as np

from btp_python import SVM_Online


C_SVM = 5.
RESERVE_THRESHOLD = float("Inf")

model_params = {
  "C" : C_SVM,
  "eps" : RESERVE_THRESHOLD,

  }

X_train, y_train = pickle.load(open("random_cancer.p", "rb"))
# scale y_train inputs
y_train = y_train*2-1

svm = SVM_Online(X=X_train, y=y_train, C_svm=C_SVM)
svm.train_all()
