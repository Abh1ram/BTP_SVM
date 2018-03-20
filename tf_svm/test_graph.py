import numpy as np
import pickle
import sys
import time

import tensorflow as tf

from sklearn import svm, datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from data_loader import extract_data
from var_free_graph import simple_kernel, svm_train, svm_eval


C_SVM = 5.
RESERVE_THRESHOLD = float("Inf")

model_params = {
  "C" : C_SVM,
  "eps" : RESERVE_THRESHOLD,
  "kernel" : simple_kernel,
  }


# x_train = x_train[1:]
# y_train = y_train[1:]

def compare(X, y, partial_=True, save_file=False):
    big = np.append(X, y.reshape(-1,1), axis=1)
    np.random.shuffle(big)

    X = big[:, :-1]
    y = big[:, -1]

    # n = round(X.shape[0]*0.7)    
    # print("Train size: ", n)
    # X_train = X[:n]
    # y_train = y[:n]
    # X_test = X[n:]
    # y_test = y[n:]
    
    # Applying k fold cross validation
    kf = KFold(n_splits=5)
    num_splits = kf.get_n_splits(X)
    acc_std, time_std, = 0, 0
    acc_mine, time_mine = 0, 0
    for train_index, test_index in kf.split(X):
        # print("Lengths of train and test: ", len(train_index), len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if save_file:
            pickle.dump((X_train, y_train), open("random_cancer.p", "wb"))

        n = len(X_train)
        print("Starting standard training", partial_)
        if not partial_:
            clf = svm.SVC(kernel="linear", shrinking=False)
            t1 = time.time()
            clf.fit(X_train, y_train)
            time_std += time.time()-t1
            # print((clf.support_))
            # print("Length: ")
            # print(len(clf.support_))
        else:
            # alpha = 1/X_train.shape[0]
            clf = linear_model.SGDClassifier()
            t1 = time.time()
            for jj in range(2):
                for ii in range(0, len(X_train), 15):
                    upper = min(ii+15, len(X_train))
                    clf.partial_fit(X_train[ii : upper], y_train[ii : upper], np.array([0, 1]))
            time_std += time.time()-t1
            # print("TIme taken for SGDClassifier: ", time.time()-t1)
        print("Starting standard prediction")
        pred = clf.predict(X_test)
        acc_std = accuracy_score(y_test, pred)
        # print("Standard accuracy: ", acc_std)
        # print("Time std: ", time_std)
        # Check performance of  my model
        # make labels -1, 1
        y_train = y_train*2 - 1
        y_test = y_test * 2 - 1
        with tf.device("/cpu:0"):
          time_mine,_ = svm_train(X_train, y_train, model_params, restart=True)
          acc_mine = svm_eval(X_test, y_test, model_params)
        # print("ACC MINE: ", acc_mine)
        # print("TIME MINE: ", time_mine)
        break

    num_splits = 1
    print("\n\n-----------------------------")
    print("Acc std: ", acc_std/num_splits)
    print("Acc mine: ", acc_mine/num_splits)
    print("Time std: ", time_std/num_splits)
    print("Time mine: ", time_mine/num_splits)


# try on different datasets
dataset_typ = int(sys.argv[1])
print("TYPE: ", dataset_typ)
input("ENTER STH TO START: ")
X, y = \
{
  0 : datasets.load_breast_cancer(True),
  1 : datasets.load_iris(True),
  2 : datasets.load_digits(10,True),
  3 : datasets.load_wine(True),
}[dataset_typ]
print("Size of dataset: ", X.shape)
y_uniq = np.unique(y)

for x in X:
    if sum(abs(x)) == 0:
        print(x)
        input()
# convert multi-class to two classes
if len(y_uniq) != 2:
    print(y_uniq)
    for i in range(y_uniq.shape[0]):
        print()
        print("-----------------------------------")
        print(i)
        y2 = np.zeros(y.shape[0])
        for j in range(y.shape[0]):
            if y[j] == i:
                y2[j] = 0
            else:
                y2[j] = 1
        compare(X, y2, partial_=True)

else:
    compare(X, y, partial_=False, save_file=True)