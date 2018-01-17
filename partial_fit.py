import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from sklearn import datasets, linear_model
from btp_2 import SVM_Online


def mesh_plot(X, y, svc):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max - x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
     np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()


X, y = (datasets.load_breast_cancer(True))
# print(y.reshape(-1, 1)[:5,:])
big = np.append(X, y.reshape(-1,1), axis=1)

print(X.shape)
tot_pts = X.shape[0]

np.random.shuffle(big)
pickle.dump(big, open("random_cancer.p", "wb"))

X = big[:, :-1]
y = big[:, -1]

n = round(X.shape[0]*0.7)
print(n)

X_train = X[:n]
y_train = y[:n]
X_test = X[n:]
y_test = y[n:]

# print(y_train)
# input()

# For best results using the default learning rate schedule,
# the data should have zero mean and unit variance.

clf = linear_model.SGDClassifier()
t1 = time.time()
# clf.fit(X, y)
# for jj in range(5):
for ii in range(0, len(X_train), 15):
    upper = min(ii+15, len(X_train))
    clf.partial_fit(X_train[ii : upper], y_train[ii : upper], np.array([0, 1]))
# clf.fit(X_train, y_train)
print("TIme taken for SGDClassifier: ", time.time()-t1)
# mesh_plot(X_train, y_train, clf)
# input("INP: ")
# print((clf.support_))
# print("Length: ")
# print(len(clf.support_))

cor = 0
pred = clf.predict(X_test)
for i in range(pred.shape[0]):
    if pred[i] == y_test[i]:
        cor += 1
print("Accuraccy:")
print(cor/(tot_pts-n))

# input("INP: ")
# mysvm = SVM_Online(X=X, y=y*2-1,C_svm=1)

mysvm = SVM_Online(X=X_train, y=y_train*2-1,C_svm=1)
mysvm.train_all()
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


print("Mismatch")
print(mismatch)
print("Accuraccy:")

print(cor/(tot_pts-n))
# print(y_test)