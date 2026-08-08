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

print(X_train[:, 0].shape)

def init_params():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

def RelU(z):
    

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relU(Z1)