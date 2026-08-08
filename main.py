import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("data/train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
X_train = data_train[1:n]
Y_train = data_train[0]

def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return exp(Z) / np.sum(exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = reLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

def onehot(Y):
    onehot_Y = np.zeros((Y.size, Y.max()+1))
    onehot_Y[np.arrange(Y.size), Y] = 1
    return (onehot_Y.T)

def deriv_ReLU(Z):
    if Z > 0:
        return 1
    if Z <= 0:
        return 0

def back_prop(Z1, Z2, A1, A2, Y):
    one_hot_Y = onehot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2)