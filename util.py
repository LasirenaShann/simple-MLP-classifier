import os
import numpy as np
from PIL import Image
from natsort import natsorted

class Normalizer(object):
    """
    Normalize to have zero-mean and unit-variance
    """
    def fit(self, X):
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)

    def fit_transform(self, X):
        self.fit(X)
        X_new = self.transform(X)
        return X_new

    def transform(self, X):
        X_new = (X - self.mu) / self.sigma
        return X_new

def gaussianMLE(X):
    """
    Maximum likelihood estimation for gaussian pdf
    """
    N = X.shape[0]
    mu = np.mean(X, axis=0)
    sigma = np.dot((X - mu).T, X - mu) / (N-1)
    return mu, sigma

def readAll(dir_path):
    """
    Read images from all subdirectories under the directory specified by dir_path
    Params
        dir_path: path to the directory containing subdirectores(classes) of images
    Return
        img: N-by-900 matrix
        labels: N-by-3 matrix, one-hot
    """
    subdir_list = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]
    img = []
    labels = []
    for i, subdir in enumerate(subdir_list):
        if not len(img):
            img, labels = readImages(dir_path + '/' + subdir, i, len(subdir_list))
        else:
            i, l = readImages(dir_path + '/' + subdir, i, len(subdir_list))
            img = np.concatenate((img, i), axis=0)
            labels = np.concatenate((labels, l), axis=0)
    return img, labels

def readImages(subdir_path, label=None, num_class=None):
    """
    Read images from the subdirectory(class)
    Params
        subdir_path: subdirectory containing images of the same class
        label: scalar, indicating the label of the class
        num_class: number of classes
    Return
        img: N-by-900 matrix
        labels: N-by-3 matrix, one-hot
    """
    img = []
    for f in natsorted(os.listdir(subdir_path)):
        if f.endswith(".bmp"):
            i = Image.open(subdir_path + '/' + f)
            # print f
            if not len(img):
                img = np.array(i.getdata())
            else:
                img = np.vstack((img, np.array(i.getdata())))
    if label is not None and num_class is not None:
        labels = np.zeros((img.shape[0], num_class))
        labels[:,label] = 1
        return img, labels
    return img

def partition(DS, n_fold, idx):
    """
    Partition the dataset into training set and validating set
    Params
        DS: input dataset
        n_fold: number of folds for cross-validation
        idx: the index of the fold to be used as the validating set
    Returns
        training_set, validating_set
    """
    chunks = np.array_split(DS, n_fold) # split the dataset into n_fold equally-sized chunks
    validating_set = chunks[idx] # assign chunks[idx]
    if idx == 0:
        training_set = np.concatenate(chunks[idx+1:], axis=0)
    elif idx == n_fold-1:
        training_set = np.concatenate(chunks[:idx], axis=0)
    else:
        a = np.concatenate(chunks[idx+1:], axis=0)
        b = np.concatenate(chunks[:idx], axis=0)
        training_set = np.append(a, b, axis=0)
    return training_set, validating_set

if __name__ == '__main__':
    img = readImages('Demo')
    print img.shape

