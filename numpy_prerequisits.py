"""
* dot product (inner product of a and b both must have same shape):
    * `a.b `= `a^T b `= `sigma a_db_d (for d in [1,D])`

* Matrix multiplication
    * Generalized dot product of A which is mxn and B which is nxp
    * AB= sigma a_ik b_kj ( for k in [1,n])

* Elementwise  matrix product : not common in Algebra (common in ML):
    * both A and B must have the same shape and out put is also the same shape (eg. mxn)

* Use Numpy to solve:
    * Linear Systems: Ax=b
    * Inverse: A^-1
    * Determinant : |A|
    * Choosing random numbers (e.g. from Uniform, Gaussian distributions)

* Applications:
 * linear regression, Logistic regression, DNNs, K-means clustering, density estimation, PCA, matrix factorization, HMMs, MMs, SVMs, Game theory, Portfolio Optimization, etc


"""



#%%

# Size of a list can change but size of an array is fixed
# You need to instanciate a new array in order to incerease the size
# If you apply a function to numpy array, it often operates elementwise (broadcasting)


import numpy as np

L = [1, 2, 3]  # list
A = np.array([1, 2, 3])  # Array

# %% List operations: made for doing data structure operations

for i in L:
    print(i)

L.append(4)
L_concat = L + [5]
L_extend = L.extend(6)
L_concat = L + L  # repeats the list two times,
L_concat = 2 * L  # repeats the list two times, similar  to concatining two lists
element_wise_sum = [i + 3 for i in L]  # list comprehension
element_wise_square = [i ** 2 for i in L]
# %% Array operations : made for doing math
for i in A:
    print(i)

A_sum = A + np.array([4])  # "broadcasting: 4 is added to every element in the array A"
A_sum = A + np.array([4, 5, 6])  # adding array to array: regular array addition
A_mult = 2 * A  # multiplication
A_squared = A ** 2
A_radical = np.sqrt(A)  # elementwise square operation
A_log = np.log(A)
A_exp = np.exp(A)
A_tanh = np.tanh(A)

# %% Array dot product

"""
* dot product (inner product of a and b both must have same shape):
    * `a.b `= `a^T b `= `sigma a_db_d (for d in [1,D])` = |a||b|cos(theta_ab)
* angle between two vectors: cos(theta_ab)= A^Tb / (|a||b|)
"""

a = np.array([1, 2])
b = np.array([3, 4])

# method 1
dot = 0
for i, j in zip(a, b):
    dot += i * j

# method 2
dot = 0
for i in range(len(a)):
    dot += a[i] * b[i]

# method 3
dot = np.sum(a * b)

# method 4 (instance methods)
dot = (a * b).sum()

# method 5
dot = np.dot(a, b)

# method 6
dot = a.dot(b)

# method 7
dot = a @ b

# %% finding norm/magnitude of a vector ||a||

# method 1
a_magnitude = np.sqrt((a * a).sum())  # a_magnitude = sqrt(sum(a_d^2) for d in range(1,D))

# method 2 (using linear algebra)
a_magnitude = np.linalg.norm(a)

# %% angle between two vectors in gradients not degrees

cos_angle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

angle = np.arccos(cos_angle)

# %% speedtest

from datetime import datetime

T = 100000

a = np.array([1, 2])
b = np.array([3, 4])


def slow_dot_product(a, b):
    dot = 0
    for i, j in zip(a, b):
        dot += i * j
    return dot


t0 = datetime.now()
for t in range(T):
    slow_dot_product(a,b)
dt1 = datetime.now() - t0

t0 = datetime.now()
for t in range(T):
    a.dot(b)
dt2 = datetime.now() - t0

print(f"dt1/dt2:{dt1.total_seconds() / dt2.total_seconds()}")

#%% Matrices and Operations
"""
* In numpy the numpy.matrix object which is specialized 2D array
* whereas np.array object can have any dimensions
* so in most cases we prefer to us enp.array over nimpy.matrix unless we are dealing with sparse metrices in scipy 
"""

L = [[1,2],[3,4]] # list of lists
access_ij = L[0][1] # access element in row i=0 column j=1

A = np.array(L)+2 #elementwise broadcast operation
B = np.array(L)*2 #elementwise broadcast operation

access_ij = A[0][1] # access element in row i=0 column j=1
access_ij = A[0,1] # access element in row i=0 column j=1
access_col_j = A[:,0]  # access element in row column j=0
A_transpose = A.T

# you dont need to pass your list to numpy array before operations
# these two have same results
exp_A = np.exp(A)
exp_L = np.exp(L) # numpy ppretends as if the list is a numpy array

# Elementwise pultiplication
element_wise_mult = A*B

# Matrix multiplication
matrix_mult = A.dot(B)

# matrix determinant
det_A = np.linalg.det(A)
det_A = np.linalg.det(L)

# identity matrix A^-1 . A
identity_matrix = np.linalg.det.inv(A).dot(A)

# matrix trace
matrix_trace = np.trace(A)
matrix_trace = np.trace(L)

# diag: if you give a matrix input, you will get a vector. If you give a vector input you get a diagonal matrix
matrix_diag = np.diag(A)
matrix_diag = np.diag(L)
generate_diag_matrix_from_vector = np.diag([1,4]) # [[1,0],[0,4]]

"""
Compare whether matrices are equal np.allclose(A,B)
"""
# eigen values and eigen vectors of a matrix
# returns an array comprising an array of eigen values and an array of eigen matrix
Lam, V = np.linalg.eig(A)
print(V[:,0] * Lam[0] , A @ V[:,0])

# array([ True, False]): due to numerical precision despite the fact that both are the same we dont see True,True
V[:,0] * Lam[0] == A @ V[:,0] # [ 0.30697009 -0.21062466] == [ 0.30697009 -0.21062466]

# the right way to compare whether two arrays are equal is following
np.allclose(V[:,0] * Lam[0], A @ V[:,0])
np.allclose(V @ np.diag(Lam), A @ V)

#%%
"""
Solving linear systems Ax=b  using np.linalg.solve(A,b)
However we should not use (x = A^-1 b) approach in numpy cause numpy inverse is slow and not accurate
"""
L = [[1,1],[1.5,4]]
A  = np.array(L)
b = np.array([2200,5050])

x = np.linalg.inv(A).dot(b) # NO!
x = np.linalg.solve(A,b) # YES!

#%% Generating data

# array of all zeros
np.zeros((2,3))

# array of all ones
np.ones((2,3))

# array of all tens
10 * np.ones((2,3))

# identity matrix
np.eye(3) # 3x3 identity matrix


# select values that match a conditional criteria
A = np.arange(10)
A_even = A[A%2==0] #array([0, 2, 4, 6, 8])

L=[[1,2,3],[4,5,6],[7,8,9]]
A=np.array(L)
A_even = A[A%2==0] #array([2, 4, 6, 8])

# generate arrays with random numbers in it. Numbers are between 0 and 1
constant_rand_num = np.random.random()
matrix_rand_num = np.random.random((2,3))

# generate arrays with random numbers generated from a normal distribution (gaussian)
# unlike zeros, ones and random commands,
# randn doesn't accept input tuples (i,j) and
# it only accepts i,j single numbers as input

np.random.randn(2,3)

R = np.random.randn(1000)

R_mean = R.mean()
R_mean = np.mean(R)

R_variance = R.var()
R_variance = np.var(R)

R_std = R.std() # standard_deviation = sqrt(variance)
R_std = np.std(R)

RR = np.random.randn(1000,3) # each row is a sample/observation, and each col is an specific measurement
mean_of_each_col = RR.mean(axis=0)
RR.mean(axis=0).shape #(3,)

mean_of_each_row = RR.mean(axis=1)
RR.mean(axis=1).shape # (1000,)

#Find covariance of a matrix:
# cov function treats each column as a vector observation instead of each row
np.cov(RR) #
np.cov(RR).shape # (1000,1000)
#We have to transpose R, to make the cov function treat each row as an observation
np.cov(R.T)
np.cov(RR.T).shape # (3,3)
#Another option is to use rowvar
np.cov(RR, rowvar=False)

#%%
# Generating numbers
np.random.randint(low=0, high=10, size=(3,3))
np.random.choice(a=[1,2,3],size=(2,2,3),replace=True)
#Cannot take a larger sample than population when 'replace=False'
np.random.choice(a=[1,2,3,4,5,6,7,8],size=(2,2),replace=False)
#gives you an arrany of (m,n) with values in [0,8) range
np.random.choice(a=8,size=(3,3))
