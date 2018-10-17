import numpy as np
from perceptron import Perceptron as perc

w = np.array([[1, 2, 3]])
b = 0
x = np.array([[1, 2, 3], [0, -1, -1], [7, 8, 9]])
xi = np.array([[1, 2, 3]])
y = np.array([[0, 1, 0]])

a = perc(w.T,b)

# print(w.)

# print(y[0, 0])
# print(x[0:1].T)
# print(a.vectorized_forward_pass(x))
print(a.train_on_single_example(xi.T, y[0, 0]))
# print(w)
# print(list(x))