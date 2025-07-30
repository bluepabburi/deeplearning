import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity(x):
    return x


x = np.array([1, 0.2])
w1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
b = np.array([0.2, 0.3, 0.4])
y = np.dot(x, w1) + b

z1 = sigmoid(y)

w2 = np.array([[0.2, 0.3],[0.4, 0.5],[0.3, 0.4]])

z2 = np.dot(z1, w2)

print(z2)

Y = identity(z2)

