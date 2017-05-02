import numpy as np
from sklearn.externals import joblib
from util import *

normalizer_file = 'normalizer.sav'
pca_file = 'pca.sav'
lda_file = 'lda.sav'
model_file = 'classifier.sav'
target_file = 'DemoTarget.csv'
folder_name = 'Demo'

normalizer = joblib.load(normalizer_file)
pca = joblib.load(pca_file)
lda = joblib.load(lda_file)
classifier = joblib.load(model_file)
X_test = readImages(folder_name)
X_test = normalizer.transform(X_test)
X_test = pca.transform(X_test)
X_test = lda.transform(X_test)
predictions = classifier.predict(X_test)
np.savetxt(target_file, predictions, fmt='%d', delimiter=',')
print("Predictions written to file ---> {}".format(target_file))

"""
T = np.loadtxt('answer.csv', delimiter=',')
hitIndex = np.where(np.all(predictions==T, axis=1))[0]
print("# Number of correct predictions --> {} / {}".format(len(hitIndex), X_test.shape[0]))
recogRate = len(hitIndex) / float(X_test.shape[0])
print("# Recognition rate --> {}%".format(recogRate * 100.0))
"""

