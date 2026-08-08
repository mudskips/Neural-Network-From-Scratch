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
X_dev = X_dev / 255.

data_train = data[1000:m].T
X_train = data_train[1:n]
X_train = X_train / 255.
Y_train = data_train[0]

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def onehot(Y):
    onehot_Y = np.zeros((Y.size, Y.max()+1))
    onehot_Y[np.arange(Y.size), Y] = 1
    return (onehot_Y.T)

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, Z2, A1, A2, W2, X, Y):
    one_hot_Y = onehot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def grad_descent(X, Y, iters, alpha):
    W1,b1, W2, b2 = init_params()
    for i in range(iters):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, Z2, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 50 == 0):
            print(f"Iteration {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
    return W1, b1, W2, b2

W1, b1, W2, b2 = grad_descent(X_train, Y_train, 3000, 0.1)
            
def make_predictions(X, W1, b1, W2, b2):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(f"Dev accuracy: {get_accuracy(dev_predictions, Y_dev)}")

