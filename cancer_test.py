import numpy as np
import pickle
import sys
import time

from sklearn import svm, datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from btp_python import SVM_Online

def compare(X, y, partial_=True, save_file=False, C=1.):
    # print(y.reshape(-1, 1)[:5,:])
    big = np.append(X, y.reshape(-1,1), axis=1)

    print(X.shape)
    tot_pts = X.shape[0]

    np.random.shuffle(big)

    X = big[:, :-1]
    y = big[:, -1]
    
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
            pickle.dump((X_train, y_train), open("train_data.p", "wb"))
            pickle.dump((X_test, y_test), open("test_data.p", "wb"))
        n = len(X_train)

        if not partial_:
            clf = svm.SVC(kernel="linear", shrinking=False, C=C)
            t1 = time.time()
            # clf.fit(X, y)
            clf.fit(X_train, y_train)
            time_std += time.time()-t1
            print("TIme taken for svc: ", time.time()-t1)
        # mesh_plot(X_train, y_train, clf)

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

        cor = 0
        pred = clf.predict(X_test)

        acc_std_iter = accuracy_score(pred, y_test)
        print("Accuracy of svc: ", acc_std_iter)
        acc_std += acc_std_iter

        mysvm = SVM_Online(X=X_train, y=y_train*2-1,C_svm=C)
        t1 = time.time()
        mysvm.train_all()
        time_mine += time.time() - t1
        # print("BDRY-------------")
        # mysvm.dec_bdry(12, 17)

        pos, neg = [],[]
        # print()
        # for x in sorted(mysvm.Margin_v+mysvm.Error_v):
        #     if mysvm.y_all[x] == 1:
        #         pos.append(x)
        #     else:
        #         neg.append(x)

        # print(pos, neg)

        pred_mine = []
        for x in X_test:
            pred_mine.append(mysvm.predict(x))
        acc_mine_iter = accuracy_score(pred_mine, y_test*2-1)
        print("My acc: ", acc_mine_iter)
        acc_mine += acc_mine_iter
        # break
        print("-------------SPLIT DONE----------------------\n")

    num_splits = 5
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
print(y_uniq)
input()

C_SVM = 1.

# convert multi-class to two classes
if len(y_uniq) != 2:
    print(y_uniq)
    for i in range(y_uniq.shape[0]):
        print()
        print("-----------------------------------")
        print("CLASS:", i, "\n")
        y2 = np.zeros(y.shape[0])
        for j in range(y.shape[0]):
            if y[j] == i:
                y2[j] = 0
            else:
                y2[j] = 1
        compare(X, y2, partial_=False, C=C_SVM)

else:
    compare(X, y, partial_=False, C=C_SVM)