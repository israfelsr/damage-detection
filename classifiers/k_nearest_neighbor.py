from builtins import range
from builtins import object

import numpy as np


class KNearestNeighbor(object):
    """a kNN clf using l2 distance"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        This function trains the clf. It simply memorize the data.
        :param X: matrix in which each row is a image sample
        :param y: matrix which each row is a image label
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Calculate the distance pixel-wise between the test data
        and the images saved in the training set.
        :param X: matrix containing test samples as rows.
        :param k: number of nearest neighbors to use in the algorithm
        :return dists: matrix of matrices with distance to each pixel.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        num_pix = self.X_train.shape[1]
        dists = np.zeros((num_test, num_train, num_pix))
        # L2 distance
        for i in range(num_test):
            dists[i][:][:] = np.square(X[i] - self.X_train)
        return dists

    def predict_labels(self, dists, k=1):
        """
        Using the dists matrix evaluate the kNN to choose a label for
        each pixel in a image.
        :param dists: Matrix with pixel-wise distance
        :param k: Number of neighbours voting for the label
        :return y_pred: matrix with a predicted image for each test input.
        """
        num_test = dists.shape[0]
        num_pix = dists.shape[2]
        y_pred = np.zeros((num_test, num_pix))
        for i in range(num_test):
            # Choose the k nearest pixels and search their GT value
            index_lbl = np.argsort(dists[i], axis=0)[0:k,:]
            pixel_sum = [self.y_train.T[j][index_lbl.T[j]] for j in range(num_pix)]
            # Sum the pixels. Each pixel in GT is 1 or 0.
            pixel_sum = np.sum(pixel_sum, axis=1)
            # Pixel in GT voting for the label. In this moment we're using 50+1.
            y_pred[i] = np.where(pixel_sum > k//2, 1, 0)
            if (i + 1) % 10 == 0:
                print('%i EPOCH' % (i+1))
        return y_pred

