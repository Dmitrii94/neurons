import numpy as np
from neiron import Neuron as neuron

batch_size = 3
learning_rate = 0.01
eps = 1e-4
max_steps = 1

w = np.array([[0, 0.5, 0.1, 0.2]])
X = np.array([[1, 1, 2, 3], [1, 2, -1, -1], [1, 3, 8, 9], [1, 4, 4, 8], [1, 5, 8, 1], [1, 6, 5, 2]])
xi = np.array([[1, 2, 3]])
y = np.array([[0], [1], [0], [1], [0], [1]])

a = neuron(w.T)

print(a.SGD(X, y, batch_size, learning_rate, eps, max_steps))
print(a.w)
# print(a.vectorized_forward_pass(x))
# print(a.update_mini_batch(x, y.T, learning_rate, eps))

# print(w.size)
# rand = rnd.randint(0, x.shape[0]-batch_size)
# print(rand)
# print(x[rand:rand+batch_size])
# print(y.T[rand:rand+batch_size])
# print(x[3:6])