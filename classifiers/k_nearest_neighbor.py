from builtins import range
from builtins import object

import numpy as np


class KNearestNeighbor(object):
    """a kNN clf using l2 distance"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        This function trains the clf. It memorize the data.
        :param X: matrix in which each row is a image sample
        :param y: matrix which each row is a image label
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Predict the labels of the test data.
        :param X: matrix containing test samples as rows.
        :param k: number of nearest neighbors to use in the algorithm
        :return y_pred: matrix with labels as rows
        """
        # Calcular distancia entre X e self.X
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        num_pix = self.X_train.shape[1]
        dists = np.zeros((num_test, num_train, num_pix))
        # 1 loops way
        for i in range(num_test):
            dists[i][:][:] = np.square(X[i] - self.X_train)
        return dists
        #return self.predict_labels(dists, k=k)

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        num_train = dists.shape[1]
        num_pix = dists.shape[2]
        y_pred = np.zeros((num_test, num_pix))
        for i in range(num_test):
            index_lbl = np.argsort(dists[i], axis=0)[0:k,:]
            pixel_sum = [self.y_train.T[j][index_lbl.T[j]] for j in range(num_pix)]
            pixel_sum = np.sum(pixel_sum, axis=1)
            y_pred[i] = np.where(pixel_sum > k//2, 1, 0)
            if (i + 1) % 10 == 0:
                print('%i EPOCH' % (i+1))
        return y_pred

