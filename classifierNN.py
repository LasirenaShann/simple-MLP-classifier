import numpy as np
from scipy.special import expit

class ClassifierNN(object):
    """ Feedforward Neural Network for Classification.

    Parameters
    ------------
    eta: learning rate
    n_epoch: number of iterations over ALL training data
    n_hidden: number of hidden layers
    n_node: number of nodes per hidden layer
    batch_size: how many data per gradient descent
    activation: choose between sigmoid and relu
    random_state: random seed, default to be 1
    verbose: if true, display training progress

    Attributes
    ------------
    cost: a list of cross-entropy loss after each epoch

    """
    def __init__(self, eta, n_epoch, n_hidden, n_node, batch_size=1,
                 activation='sigmoid', random_state=1, verbose=False):
        self.eta = eta
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.n_node = n_node
        self.batch_size = batch_size
        self.activation = activation
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, t):
        """ Learn weights from the training data

        Parameters
        -----------
        X: input data with shape (n_sample, n_feature)
        t: expected class labels with shape (n_sampe, n_output)
        """
        self.cost = []
        n_sample, n_feature = X.shape
        n_output = t.shape[1]
        self._init_weights(n_feature, n_output)
        for i in range(self.n_epoch):
            if self.verbose:
                print("Epoch {}...".format(i+1))
            idx = np.random.permutation(t.shape[0])
            X, t = X[idx], t[idx]
            # Split the dataset
            mini_batch = np.array_split(range(n_sample), np.ceil(n_sample / float(self.batch_size)))
            for idx in mini_batch:
                a = self._feedforward(X[idx]) # forward pass
                loss = self._compute_cost(a[-1], t[idx]) # compute loss
                self._backprop(a, t[idx]) # compute gradient and update weights via backpropagation
                self.cost.append(loss)

    def _compute_cost(self, y, t):
        """ Compute the cross-entropy cost

        Parameters
        -----------
        y: output of the network
        t: expected output

        Returns
        -----------
        cross-entropy loss
        """
        return -(t * np.log(y)).sum() / t.shape[1]
        #return -((t * np.log(y) + (1 - t) * np.log(1 - y) * (1 - y)).sum()) / t.shape[1]

    def _init_weights(self, n_feature, n_output):
        """ Initialize the network weights and bias randomly

        Parameters
        -----------
        n_feature: feature dimension
        n_output: number of classes
        """
        rng = np.random.RandomState(self.random_state)
        self.w_ = []
        self.bias_ = []
        w_i2h_ = rng.normal(loc=0.0, scale=0.01, size=(n_feature, self.n_node))
        bias_i2h_ = rng.normal(loc=0.0, scale=0.01, size=(self.n_node, 1))
        self.w_.append(w_i2h_)
        self.bias_.append(bias_i2h_)
        for i in range(self.n_hidden - 1):
            #print('hidden layer {} to {}'.format(i, i+1))
            w_h2h_ = rng.normal(loc=0.0, scale=0.01, size=(self.n_node, self.n_node))
            bias_h2h_ = rng.normal(loc=0.0, scale=0.01, size=(self.n_node, 1))
            self.w_.append(w_h2h_)
            self.bias_.append(bias_h2h_)
        w_h2o_ = rng.normal(loc=0.0, scale=0.01, size=(self.n_node, n_output))
        bias_h2o_ = rng.normal(loc=0.0, scale=0.01, size=(n_output,1))
        self.w_.append(w_h2o_)
        self.bias_.append(bias_h2o_)

    def _activate(self, Z, output=False):
        """ Apply activation functions

        Parameters
        -----------
        Z: net input from the previous layer
        output: indicates the output layer. If true, apply softmax function

        Returns
        -----------
        activations: an array
        """
        if output:
            return self._softmax(Z)
        elif self.activation == 'sigmoid':
            return self._sigmoid(Z)
        elif self.activation == 'relu':
            return self._relu(Z)
        else:
            raise NotImplementedError()

    def _feedforward(self, X):
        """ Compute feed-forward

        Parameters
        -----------
        X: input

        Returns
        -----------
        a: a list of activations (with each list element a np-array)
        """
        a = [None] * (self.n_hidden + 2)
        a[0] = X
        for i in range(self.n_hidden):
            # Compute net input Z
            Z = np.dot(a[i], self.w_[i]) + self.bias_[i].T
            a[i + 1] = self._activate(Z)
        Z = np.dot(a[-2], self.w_[-1]) + self.bias_[-1].T
        a[-1] = self._activate(Z, output=True)
        return a

    def _sigmoid_grad(self, Z):
        """ Compute the derivative of the logistic function
        """
        s = self._sigmoid(Z)
        return s * (1.0 - s)

    def _relu_grad(self, Z):
        """ Compute the derivative of the rectified linear unit
        """
        r = self._relu(Z)
        delta = np.ones(r.shape)
        delta[r == 0] = 0
        return delta

    def _backprop(self, a, t):
        """ Perform back-propagation to update weights and bias

        Parameters
        -----------
        a: a list of activations
        t: an array expected output
        """
        delta = [None] * (self.n_hidden + 1)
        grad = [None] * (self.n_hidden + 1)
        # For the output layer
        delta[-1] = (a[-1] - t).T
        grad[-1] = np.dot(delta[-1], a[-2]).T
        # For the hidden layers
        for i in reversed(range(self.n_hidden)):
            h = np.dot(a[i], self.w_[i])
            if self.activation == 'sigmoid':
                delta[i] = np.dot(self.w_[i+1], delta[i+1]) * self._sigmoid_grad(h.T + self.bias_[i])
            elif self.activation == 'relu':
                delta[i] = np.dot(self.w_[i+1], delta[i+1]) * self._relu_grad(h.T + self.bias_[i])
            else:
                raise NotImplementedError()
            grad[i] = np.dot(delta[i], a[i]).T

        # Update weights
        for i in range(self.n_hidden+1):
            self.w_[i] -= self.eta * grad[i]
            self.bias_[i] -= self.eta * np.reshape(delta[i].sum(axis=1), (delta[i].shape[0],1))

    def _sigmoid(self, X):
        """ Compute the logistic sigmoid function
        """
        # return 1.0 / (1 + np.exp(-X))
        # Use scipy expit to prevent overflow
        return expit(X)

    def _relu(self, X):
        """ Rectified linear unit
        """
        X_new = np.zeros(X.shape)
        np.maximum(X, 0, X_new)
        return X_new

    def _softmax(self, x):
        """ Softmax function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=1)[:,None]

    def _encode_one_hot(self, y):
        """ Encode the softmax output into one-hot vector
        """
        labels = np.identity(y.shape[1])
        one_hot = labels[np.argmax(y, axis=1)]
        return one_hot


    def predict(self, X):
        """ Predict class labels
        """
        a = self._feedforward(X)
        return self._encode_one_hot(a[-1])

if __name__ == '__main__':
    classifier = ClassifierNN(eta=0.01, n_epoch=5, n_hidden=1, n_node=10);
    #classifier._init_weights(n_feature=2, n_output=3)
    classifier.fit(np.ones((3,2)), np.identity(3))

