import numpy as np

class PcaModel(object):
    """
    A dimensionality reduction model by principal components analysis
    """
    def __init__(self, n_components, solver='eigen'):
        """
        Param
            n_components: number of components to keep
            solver: svd or eigen
        """
        self.n_components = n_components
        self.solver = solver

    def fit(self, X):
        """
        Fit the model with X
        Param
            X: N-by-n_features matrix, training data
        """
        N = X.shape[0]
        mu = np.mean(X, axis=0)
        # Find the covariance matrix
        C = np.dot(X.T, X) / (N-1)
        if self.solver == 'svd':
            # PCA by singular value decomposition
            u, s, v = np.linalg.svd(C)
            self.components = v[:,:self.n_components]
        elif self.solver == 'eigen':
            # PCA by eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
            self.components = eigvecs[:,:self.n_components]
        else:
            raise NotImplementedError()
        return self

    def fit_transform(self, X):
        """
        Fit the model with X and apply dimensionality reduction on it
        Param
            X: N-by-n_features matrix
        Return
            X_new: reduced N-by-n_components matrix
        """
        self.fit(X)
        X_new = self.transform(X)
        return X_new

    def transform(self, X):
        """
        Apply dimensionality reduction on X
        Param
            X: N-by-n_features matrix
        Return
            X_new: reduced N-by-n_components matrix
        """
        X_new = np.dot(X, self.components)
        return X_new

class LdaModel(object):
    """
    A dimensionality reduction model by linear discriminant analysis
    """
    def __init__(self, n_components, solver='svd'):
        self.n_components = n_components
        self.solver = solver

    def fit(self, X, t):
        """
        Fit the model with training data X and its label t
        Params
            X: training data
            t: labels
        """
        n_class = t.shape[1]
        n_features = X.shape[1]
        idx = range(n_class) # scalar index to each class
        labels = np.dot(t, idx) # transform each label to be a scalar
        mean = []
        n_elements = []
        # Calculate the within-class scatter matrix
        S_w = np.zeros((n_features,n_features))
        for i in range(n_class):
            idx = np.nonzero(labels==i)[0]
            cls = X[idx]
            n_elements.append(cls.shape[0]) # number of elements in each class
            mean.append(np.mean(cls, axis=0)) # mean of each class
            S_w += np.dot((cls-mean[i]).T, cls-mean[i])

        # Calculate the between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        S_b = np.zeros((n_features,n_features))
        for i in range(n_class):
            S_b += n_elements[i] * np.dot(np.asarray(mean[i] - overall_mean).reshape(n_features,1),
                                          np.asarray(mean[i] - overall_mean).reshape(n_features,1).T)

        if self.solver == 'svd':
            u, s, v = np.linalg.svd(np.dot(np.linalg.inv(S_w), S_b))
            self.components = v[:,:self.n_components]
        elif self.solver == 'eigen':
            eigvals, eigvecs = np.linalg.eigh(np.dot(np.linalg.inv(S_w), S_b))
            eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
            self.components = eigvecs[:,:self.n_components]
        else:
            raise NotImplementedError()
        return self

    def fit_transform(self, X, t):
        """
        Fit the model with X and apply dimensionality reduction on it
        Param
            X: N-by-n_features matrix
        Return
            X_new: reduced N-by-n_components matrix
        """
        self.fit(X, t)
        X_new = self.transform(X)
        return X_new

    def transform(self, X):
        """
        Apply dimensionality reduction on X
        Param
            X: N-by-n_features matrix
        Return
            X_new: reduced N-by-n_components matrix
        """
        X_new = np.dot(X, self.components)
        return X_new

