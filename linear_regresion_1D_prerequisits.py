"""
1-D Linear regression
find the best line that fits the data (X,Y)
    * solve the Y_hat = a X + b, calculate a,b
    * tot = sum(Y -  Y_hat)^2 and Y_hat = a*X + b
    * solve for derivative of tot w.r.t a and b
"""


import numpy as np
import matplotlib.pyplot as plt
#%%
# load data
X = []
Y = []
for line in open('Data/data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# convert X,Y into numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot the data
plt.scatter(X,Y)
plt.show()



denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# then calculate the predicted Y from, a, X, b
Yhat = a*X + b

# let's plot everything together to make sure it worked
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
"""
S_res = sum (y-y_hat)^2 : d1=(Y-Y_hat) and d1.dot(d1)
S_tot  = sum (y-average(y))^2 :  d2=(Y-Y_avg) and d2.dot(d2)
R^2 = 1 - S_res/S_tot:  1 - d1.dot(d1) / d2.dot(d2)
R^2 ~ 1 is best ( your model should predict Y_hat way better than returning Y_hat as Y_average)
"""
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)

#%%

import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# some numbers show up as 1,170,000,000 (commas)
# some numbers have references in square brackets after them
non_decimal = re.compile(r'[^\d]+')

for line in open('Data/moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)


X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# copied from lr_1d.py
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# let's calculate the predicted Y
Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("a:", a, "b:", b)
print("the r-squared is:", r2)

# how long does it take to double?
# log(transistorcount) = a*year + b
# transistorcount = exp(b) * exp(a*year)
# 2*transistorcount = 2 * exp(b) * exp(a*year) = exp(ln(2)) * exp(b) * exp(a * year) = exp(b) * exp(a * year + ln(2))
# a*year2 = a*year1 + ln2
# year2 = year1 + ln2/a
print("time to double:", np.log(2)/a, "years")


