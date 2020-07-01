"""
* Gradient Descent solution you have a function you want to minimize J(w)=cost/error
  So you iteratively update w, in the direction of dJ(w)/d(w) in small steps
  By moving slowly in the direction of the gradient of a function we get closer to the minmum of that function
  Gradient descent to Linear Regression:
   -   J = cost/error = sum(y_n-y_hat_n)^2 = (Y - X.w)_T(Y - X.w)
   -   dJ/dw = - 2.X_T.Y + 2.X_T.X.w = 2.X_T.(Y_hat - Y) (can drop 2 as itas a constant)
   -   instead setting dJ/dw to 0 and solving it for w, we will take samll steps in the direction
   -   initial w is w_0
   -   w <- w - etha.dJ(w)/d(w)
   - Gradient descent for linear regression:
     w = draw a sample from N(0,1/D)
      for t = [1,T]:
        w = w - learning_rate * X_T.(Y_hat - Y)
     > can quit after a number of steps or when change in w is smaller tthan a predetermined threshold
     - If learning rate is too big: won't converge (bounce back and forth across the optima)
     - If learning rate is too small: gradient descent will be too slow
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3
X = np.zeros((N, D))
X[:,0] = 1 # bias term
X[:5,1] = 1
X[5:,2] = 1
Y = np.array([0]*5 + [1]*5)

# print X so you know what it looks like
print("X:", X)

# won't work!
# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
"""
* w = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) won't work due to `Multicollinearity`: when one column is a linear combination of other columns it is also called Multicollinearity.
IN GENERAL YOU CAN'T GUARANTEE THAT YOUR DATA ISN'T CORRELATED. Image data has strong correlations. 
Therefore Gradient descent is the preferred most general solution  as it is used throughout deep learning
* `Dummy Variable Trap`: It happens when we need to solve (X_T X)^ -1 when dealing with one-hot encoding for Categorical variables
    because (X_T X) is not invertible as X is a singular matrix because
    it has a column of 1s) and the addition of the one-hot-encoded columns sum to 1 
    meaning that one column is a linear combination of other columns.

"""


# let's try gradient descent
costs = [] # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.001
for t in range(1000):
  # update w
  Yhat = X.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*X.T.dot(delta)

  # find and store the cost
  mse = delta.dot(delta) / N
  costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

# plot prediction vs target
plt.plot(Yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()
plt.show()