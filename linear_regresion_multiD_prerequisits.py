"""
Multiple Linear Regression:

* When there are multiple inputs that can affect the output
    * E.g. Housing price is predicted by neighborhood, size, number of rooms
    * X: is an NxD matrix, each row of X is one sample (feature vector)  of size 1xD
    * N: Number of Samples
    * D: number of inputs/features.

* in algebra: y_hat = w_T*x + b
    * We can always absorb b into w by appending a 1 to the feature vector
    * in python:  y_hat = w_0x_0 + w_1x_1 +... +w_Dx_D, x_0 = 1
    * This is equivalent to adding a column of 1s to the data matrix X  (originally of size NxD)
    * Rename b  to w_0 and append  x_0which is always 1
* in python (inner dimensions must match): Y_hat = X_(NxD) W_(Dx1)

* Given that a_T.b = sigma(a_i.b_i) for i in [1:N)

* Error: E = sigma(y_i -y_hat_i)^2 for i in [1:N) ; y_hat_i = w_Tx_i  ; i in [1:N);
    * X indices are (ixj) - i: identifies which simple - j identifies which feature in that sample
    * solve for derivative of tot w.r.t w and b; given that d(w_T)/d(w_j) = 1
    * dE/dw_j =  sigma[2(y_i - w^Tx_i)(- d(w^Tx_i))/dw_j)] for j [1:N)
              =  sigma[2(y_i - w^Tx_i)(-x_ij)]  for j [1:N)
              = sigma[2(y_i - w^Tx_i)(-x_ij)] = 0
              = sigma[y_i(-x_ij)] - sigma[w_Tx_i(-x_ij)] = 0
              = w_T sigma[(x_i x_ij)] = sigma(y_i x_ij)

    * w_T(X_T.X) = y_T.X -> [w_T(X_T.X)]_T = [y_T.X]_T -> (X_T.X)w = X_T.y  -> (X_T.X)w = X_T.y
    *  similar to Ax=b; x=np.linalg.solve(A,b)
    *  solve for (X_T.X)w = X_T.y ; w = np.linalg.solve(X_T.X,X_T.y)
"""


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# load the data
X = []
Y = []
# 2D-linear regression x1,x2
for line in open('Data/data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1]) # add the bias term
    Y.append(float(y))

# let's turn X and Y into numpy arrays since that will be useful later
X = np.array(X)
Y = np.array(Y)


# let's plot the data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()


# apply the equations we learned to calculate a and b
# numpy has a special method for solving Ax = b
# so we don't use x = inv(A)*b
# note: the * operator does element-by-element multiplication in numpy
#       np.dot() does what we expect for matrix multiplication
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y)) #(X_T.X)w = X_T.y ; w = np.linalg.solve(X_T.X,X_T.y)
Yhat = np.dot(X, w)


# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)