import numpy as np
import pickle
import sys
import time

from sklearn import svm, datasets, linear_model
from sklearn.model_selection import KFold
from btp_python import SVM_Online

def compare(X, y, partial_=True):
    # print(y.reshape(-1, 1)[:5,:])
    big = np.append(X, y.reshape(-1,1), axis=1)

    print(X.shape)
    tot_pts = X.shape[0]

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

        pickle.dump((X_train, y_train), open("random_cancer.p", "wb"))
        n = len(X_train)

        if not partial_:
            clf = svm.SVC(kernel="linear", shrinking=False)
            t1 = time.time()
            # clf.fit(X, y)
            clf.fit(X_train, y_train)
            time_std += time.time()-t1
            print("TIme taken for svc: ", time.time()-t1)
        # mesh_plot(X_train, y_train, clf)

            print((clf.support_))
            print("Length: ")
            print(len(clf.support_))
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
        for i in range(pred.shape[0]):
            if pred[i] == y_test[i]:
                cor += 1
        print("Accuracy of svc: ", cor/X_test.shape[0])
        acc_std += cor/X_test.shape[0]

        # mysvm = SVM_Online(X=X, y=y*2-1,C_svm=1)

        mysvm = SVM_Online(X=X_train, y=y_train*2-1,C_svm=1)
        t1 = time.time()
        mysvm.train_all()
        time_mine += time.time() - t1
        # print("BDRY-------------")
        # mysvm.dec_bdry(12, 17)

        pos, neg = [],[]
        print()
        for x in sorted(mysvm.Margin_v+mysvm.Error_v):
            if mysvm.y_all[x] == 1:
                pos.append(x)
            else:
                neg.append(x)

        print(pos, neg)

        cor = 0
        mismatch = 0
        for i in range(X_test.shape[0]):
            if mysvm.predict(X_test[i]) == (y_test[i]*2-1):
                cor += 1
            if mysvm.predict(X_test[i]) != (pred[i]*2-1):
                mismatch += 1

        # Testing on training data
        # pred = clf.predict(X_train)
        # for i in range(X_train.shape[0]):
        #     if mysvm.predict(X_train[i]) != (pred[i]*2-1):
        #         mismatch += 1

        # print("Mismatch: ", mismatch)
        # print("Accuraccy: ", cor/(tot_pts-n))
        acc_mine += cor/X_test.shape[0]
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
print(y_uniq)
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
        compare(X, y2, partial_=False)

else:
    compare(X, y, partial_=False)