"""
 * L2 regularization (Ridge Regression): it helps with reducing the model complexity and prevent us from over-fitting to outliers.
    Data may have outliers that pull the line away from the main trend in order to minimize the square error.
    Hence we don't want very large weights because that might lead to fitting to outliers to minimize the squared error.
    As a result we add a penalty for large weights
    * L2 regularization penalty: lambda|w|^2: to do this we add a lambda multiplied by squared norm of the weights:
     J= sum(y_n-y_hat_n)^2 + lambda|w|^2
     |w|^2 = w.dot(w) = w_T w = w_1 ^2 +  ... + w_D ^2
    * Plain squared error maximizes the likelihood, and the goal of P(data|w) is to find the best w for the data distribution
      but now that we changed the cost function it is no longer the case because there are 2 terms:
      * Negative log likelihood (new cost function): J= sum(y_n-y_hat_n)^2 + lambda|w|^2
      * exp(-J) = sum(exp-(y_n-y_hat_n)^2) + exp(-lambda|w|^2)
      * Likelihood: P(Y|X,w) = mult((1/sqrt(2.pi.sig^2)) exp(0.5 (y_i - w_T x_n)^2/sig^2)) for n in [1 , N)
      * Prior: P(w) = sqrt(lambda/2.pi) exp(- 0.5 lambda w_T w)
      * L2 regularization encourages the magnitude of weights to be small cause the prior is a Gaussian centered around 0

      * Bayes rule: P(w|Y,X) = P(Y|w,X)P(w) / P(Y|X)
      * This called MAP(Maximum A Posteriori) Means we minimize the posteriori P(w|data)
         Hence P(w|Y,X) ~ P(Y|X,w)P(w)
         take derivative of cost wrt to w, set to 0, solve for w
         J = (Y - X.w)_T.(Y - X.w) + lambda.w_T.w
           = (Y_T - w_T.X_T)(Y - X.w) + lambda.w_T.w
           = Y_T.Y - Y_T.X.w - w_T.X_T.Y + w_T.X_T.X.w + lambda.w_T.w
           = Y_T.Y - 2.Y_T.X.w + w_T.X_T.X.w + lambda.w_T.w
         dJ/dw = - 2.X_T.Y + 2.X_T.X.w + 2.lambda.w = 0
         w = (lambda.I + X_T.X)^-1 X_T.Y
"""

import numpy as np
import matplotlib.pyplot as plt

N = 50

# generate the data
X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)

# make outliers manually
Y[-1] += 30
Y[-2] += 30

# plot the data
plt.scatter(X, Y)
plt.show()

# add bias term
# vstack: stack arrays in sequence vertically (row wise).
# This is equivalent to concatenation along the first axis
# after 1-D arrays of shape (N,) have been reshaped to (1,N).
# Rebuilds arrays divided by vsplit.

# X_new.shape (50,2): X_old=np.linspace(0,10,N).shape :(50,) and np.ones(N).shape : (50,)

X = np.vstack([np.ones(N), X]).T

# plot the maximum likelihood solution(ml)
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml)
plt.show()

# plot the regularized solution (map)
# probably don't need an L2 regularization this high in many problems
# everything in this example is exaggerated for visualization purposes
l2 = 1000.0 #l2 penalty
# w = (lambda.I + X_T.X)^-1 X_T.Y
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1], Yhat_map, label='map')
plt.legend()
plt.show()