import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from classifierNN import *
from util import *
from dim_reduction import *
from sklearn.externals import joblib

folder = 'Data_Train'
filename = 'classifier.sav'
eta = 0.01
n_epoch = 10
n_hidden = 1
n_node = 4
batch_size = 1
activation = 'sigmoid'
figname = 'decBoundary_1.png'
verbose = False
plotOpt = False

print("##### Loading training data from {} #####".format(folder))
X, T = readAll(folder)
N, D = X.shape
DS = np.column_stack((X, T))
rng = np.random.RandomState(1)
DS = rng.permutation(DS)

print("##### Preparing training and testing dataset #####")
DS_train, DS_test = partition(DS, 2, 1)
normalizer = Normalizer()
DS_train[:,:D] = normalizer.fit_transform(DS_train[:,:D])

print("##### Performing dimension reduction #####")
pca = PcaModel(n_components=8, solver='eigen')
lda = LdaModel(n_components=2, solver='eigen')
X_train = pca.fit_transform(DS_train[:,:D])
X_train = lda.fit_transform(X_train, DS_train[:,D:])

print("##### Start Training #####")
classifier = ClassifierNN(eta=eta, n_epoch=n_epoch, n_hidden=n_hidden, n_node=n_node, batch_size=batch_size, activation=activation, verbose=verbose)
classifier.fit(X_train, DS_train[:,D:])

print("##### Start Testing #####")
X_test = normalizer.transform(DS_test[:,:D])
X_test = pca.transform(X_test)
X_test = lda.transform(X_test)
predictions = classifier.predict(X_test)
hitIndex = np.where(np.all(predictions==DS_test[:,D:], axis=1))[0]
print("# Number of correct predictions --> {} / {}".format(len(hitIndex), X_test.shape[0]))
recogRate = len(hitIndex) / float(X_test.shape[0])
print("# Recognition rate --> {}%".format(recogRate * 100.0))

print("##### Start Training Again with the Entire Dataset #####")
DS[:,:D] = normalizer.fit_transform(DS[:,:D])
X = pca.fit_transform(DS[:,:D])
X = lda.fit_transform(X, DS[:,D:])
classifier.fit(X, DS[:,D:])
print("# -> Saving trained models to *.sav files")
joblib.dump(normalizer, 'normalizer.sav')
joblib.dump(pca, 'pca.sav')
joblib.dump(lda, 'lda.sav')
joblib.dump(classifier, filename)

if plotOpt == 1:
    # Plot learning curve
    plt.plot(range(1, classifier.n_epoch * X.shape[0] / classifier.batch_size + 1), classifier.cost)
    plt.ylabel('cost')
    plt.xlabel('iteration')
    plt.tight_layout()
    plt.savefig('cost1_batch.png', dpi=300)
    plt.show()

    # Plot decision boundary
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_bar = np.dot(Z, np.array([[1,2,3]]).T)
    Z_bar = Z_bar.reshape(xx.shape)
    plt.contourf(xx, yy, Z_bar, cmap=plt.cm.Paired)

    # Plot also the training data points
    plt.scatter(X[:, 0], X[:, 1], c=DS[:,D:], cmap=plt.cm.Paired)
    plt.savefig(figname, dpi=300)
    plt.show()

