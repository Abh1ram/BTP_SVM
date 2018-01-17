import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from sklearn import linear_model, datasets
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


dataset_sz = 800

t1 = time.time()
# X, y = datasets.make_blobs(n_samples=dataset_sz, n_features=2, centers=2, cluster_std=100)
X, y = datasets.make_classification(n_samples=dataset_sz, n_features=2, n_informative=2,
 n_redundant=0,
    n_clusters_per_class=2, )
# shift=None, scale=None)
print("Time taken to generate: ", time.time() - t1)

# print(y[:3])
# input()
# pickle.dump(big, open("random_cancer.p", "wb"))

n = round(dataset_sz*0.8)
print(n)

# move the points to have non zero mean
# X = X+np.array([3, 9])
# X = X*np.array([12,20])
# import random
# X = X + np.array([random.randrange(1, 100) for i in range(3)])
# X = X * np.array([random.randrange(1, 100)/10 for i in range(3)])


X_train = X[:n]
y_train = y[:n]
X_test = X[n:]
y_test = y[n:]

clf = linear_model.SGDClassifier()
t1 = time.time()
# clf.fit(X, y)
# for jj in range(5):
for ii in range(0, len(X_train), 10):
    upper = min(ii+15, len(X_train))
    clf.partial_fit(X_train[ii : upper], y_train[ii : upper], np.array([0,1]))
print("Time taken for SGDClassifier: ", time.time()-t1)
# mesh_plot(X_train, y_train, clf)

# print((clf.support_))
# print("Length: ")
# print(len(clf.support_))

correct = 0
pred = clf.predict(X_test)
for i in range(pred.shape[0]):
    if pred[i] == y_test[i]:
        correct += 1
print("Accuraccy:")
print(correct/pred.shape[0])

# mysvm = SVM_Online(X=X, y=y*2-1,C_svm=1)

mysvm = SVM_Online(X=X_train, y=y_train*2-1,C_svm=2)
mysvm.train_all()
print(len(mysvm.Error_v + mysvm.Margin_v))
print("BDRY-------------")
# mysvm.dec_bdry(-300, 300)



correct = 0
mismatch = 0
for i in range(X_test.shape[0]):
    if mysvm.predict(X_test[i]) == (y_test[i]*2-1):
        correct += 1
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
print(correct/X_test.shape[0])
# print(y_test)