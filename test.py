import numpy as np

n = 4
x = np.zeros((n, 1))
A = np.ones((n, n))

res = np.dot(A, x)
print(res)