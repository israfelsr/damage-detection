import scipy.io as sio
import numpy as np

from sklearn.model_selection import train_test_split
ROOT = 'project/'

def load_data(filename):
    datadic = sio.loadmat(filename)
    X = datadic['Deltas3']
    label_images = datadic['GT']
    X = image_extractor(X)
    label_images = image_extractor(label_images)
    y = match_labels(label_images)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

def image_extractor(data):
    index = 0
    matrix = np.zeros((data.shape[0] * data.shape[1], 51, 71))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            matrix[index] = data[i, j]
            index = index + 1
    return matrix

def match_labels(label_images):
    y = np.zeros([3000, 51, 71])
    for i in range(500):
        for j in range(6):
            num = i * 6
            y[num + j] = label_images[i]
    return y
