Before you start to use deeplearning libraries you need to have a prior knowledge of the following:
* Numpy stack in python: Numpy is a library for Linear Algebra and probability with Numpy Array as its central object

* `Scipy`: Adds functionality for statistics, signal processing, and computer vision: e.g. PDF, CDF, Standard normal, convolution ... 

* `Matplotlib` : linechart, scatterplot, histogram, plotting images, etc 

* `Pandas`: useful for data that is structured  like a table, csv, excel, etc

* `Linear Regression`: (line of best ft): solving real life problems to get ready for deep learning (theory and code)
    * Loss function, minimize the loss function, hyperparameters, generalization, over-fitting, regularization, etc

* addressing `over-fitting` using `L1 (LASSO Regression)` and `L2 Regularization (RIDGE Regression)` and `Elasticnet (L1 and L2)` : 
              these methods help prevent over-fitting, by not fitting to noise and outliers
              
      J = sum(y_n-y_hat_n)^2
      J_LASSO = J + lambda.|w|
      J_RIDGE = J + lambda.|w|^2
      J_ELASTIC = J + lambda.|w| + lambda.|w|^2
    
      - L1: accomplishes this by choosing the most important features. 
            It encourages a sparse solution (few non-zero w) 
      - L2: accomplishes this by making the assertion that none of the weights are extremely large
            It encourages small weights ( all w values close to 0 not exactly 0) 
      - L2 penalty is `quadratic` and L1 is an `absolute` function
        -In Quadratic as w->0, derivative -> 0: if w is already small, further gradient descent won't change much
        -In Absolute as derivative is always +/-1 (0 at w=0); so it doesn't matter where w is, it will fall at a
                       constant rate, when it reaches 0 it stays there forever
      - It's possible to include both L1 and L2 simultaneously (`Elasticnet`)

* `weight initialization (why divide w by sqrt(D))`: 
            * Poor weight initialization can lead to poor convergence of your loss per iteration, 
              your loss might even explode to infinity as a result of weights and their variance getting too big
              we might be able to avoid the exploding weights by making the variance smaller by
              initializing weights w to have mean 0 and variance 1/D as: `w = np.random.randn(D) / np.sqrt(D)`
              since we know `y = var(w_1)var(x_1)+ ...+var(w_D)var(x_D) = D.var(w)` as `var(x)=1`
              therefore if we want the `var(y)=1`, we must have `var(w) = 1/D` and hence `sd(w)=1/sqrt(D)`
              
             * This isn't the only way to initialize the weights, but same general theme applies
                - e.g. He Normal, Glorot Normal, or multiply by a small number like 0.01
                          
                                                     
    - Bellow `D` is input dimensionality and `randn()` 
      and draws samples from the standard normal N(0,1) distribution                                            
    - w = np.random.randn(D) / np.sqrt(D) 
    - so it means we want  w to have mean 0 and variance 1/D
      because randn()*C == sample from N(0,C^2) 
      if we dont devide by  np.sqrt(D) loss explodes because weights are  too large

* `Sample standardization (or Normalization)`:
make all inputs 0 mean, 1 standard deviation
    - To standardize z = (x - mu)/sigma  `here z has mean 0, sd 1`
    - Inverse transform x = z.sigma + mu `here x has mean mu, sd sigma`

* `cross-validation`: this is another method to measure your generalization error
   we split the entire dataset into k parts and for k iterations 
   we treat the ith part as test set where i is [1,k] and the rest is the training set
   we can then find mean and standard deviation of k different errors as a measure of    
   how accurate our model is and how confident we can be in that measure
   
   
* `ML broad categories`
    * supervised: output is given (X->Y)
        * Classification: trying to predict a label/category.  
            * Categorical variables: (gender, degree, city, etc)
                * one-hot-encoding leads to good interpretability because each weight tells us how much that category 
                   value affects the output
                   if a categorical variable has K different values, X will have K columns
                   one-hot-encoding of Degree : bachelors=[1,0,0] masters=[0,1,0] PhD=[0,0,1]
                   one-hot-encoding of Salary: y = 50000 - 5000x_1 + 5000x_2 (female: x_1 = 1, male x_2 = 1) 
                                           or y = 45000 + 10000 (male:x=1, female  x=0)
                * K-1 encoding: we can save 1 column out of K columns by letting all 0s represent one category value
                   this is undesirable because the effect of one category will be absorbed into the bias term
                * Predicted value E(y|female)=45000 E(y|male)=55000
        * Regression: predicting a real-valued  number or vector. 
            * Continues values: (age, years of experience, GPA)
    * unsupervised: no output - just trying to learn the structure of the data (X)
 
* `Dummy Variable Trap`: It happens when we need to solve (X_T X)^ -1 when dealing with one-hot encoding for Categorical variables
    because (X_T X) is not invertible as X is a singular matrix because
    it has a column of 1s) and the addition of the one-hot-encoded columns sum to 1 
    meaning that one column is a linear combination of other columns.

* `Multicollinearity`: when one column is a linear combination of other columns it is also called Multicollinearity.
   IN GENERAL YOU CAN'T GUARANTEE THAT YOUR DATA ISN'T CORRELATED. Image data has strong correlations. 
   Therefore Gradient descent is the preferred most general solution  as it is used throughout deep learning
   

* Ways to deal with the `Dummy Variable Trap`:
    * Statisticians suggest using `K-1 approach` instead (not interpretable as one categories will be included in the bias)
    * `L2 regularization`: X_T X is singular but (lambda. I + X_T X ) is not and we can inverse it!
            * inverting a singular matrix is the matrix equivalent of dividing by 0 hence
               adding lambda.I is the equivalent of adding a small number lambda to the denominator
    * `Remove` the column of all 1s, so there is `no bias`
    * `Gradient descent`: (the most general method, used through out deep learning since linear regression 
    is the only setting where we can find a closed form solution for the weights)
    
You can find a high level review of the used cases in this directory.

* Notations:
    * X is an NxD input matrix
    * X(i,j) is X_ij refers to i_th row and j_th columns
    * X(n,d) is X_nd refers to n_th row and d_th columns
    * N is number of samples
    * D is number of features
    * Y is an N-length Target/Output vector
    * Y_hat is an N-length prediction vector
    * Linear regression: y = ax + b 
    * Polynomial regression: add polynomial terms to linear regression 
    * Multiple Linear Regression y = wx + b
        * w: every time x increases by one y increases by w
        * b: when x is 0, y is b
        * x_i,w_i: every time x_i increases by one, and all other x's don't change, then y increases by w_i
    * Multiple Linear Regression in implementation: y = w_Tx (where w_T: transpose of matrix w)
    * J is the objective function
    * E is the Error or Cost function. The goal is to minimize the errors.
        * Squared Error: E= sigma( y - y_hat)^2 for i in [1 , N). 
        * linear regression is the maximum likelihood solution to line of best fit
        * find mu, where x_i are from the same Gaussian distribution X ~ N(mu,sig^2) : where mean = mu ; sig^2:variance
        * prob of any single point x_i: p(x_i)=pdf of the gaussian = (1/sqrt(2.pi.sig^2)) exp(0.5 (x_i - mu)^2/sig^2)
        * we can write joint likelihood of all x_i's. Multiply them cause they are IID
        * IID: Independent and Identically distributed = p(x_1,x_2,...x_N)= p(x_1)p(x_2)...p(x_N)
        * Maximum likelihood  p(X|mu)=p(x_1,...x_N), we want to find a mu so that the likelihood is maximized
        * we solve it by maximizing the log-likelihood  = d(log(p))/d(log(mu)) = 0
        * for i in [1, N) : l = - sum(x_i - mu)^2  is equivalent of minimization in linear regression E = sum(yi-yi_hat)^2
        * y ~ N(w_Tx,sig^2)  equivalent y = w_Tx + epsilon ~ N(0, sig^2)
   * Minimizing squared error is the same as maximizing log-likelihood and also maximizing the likelihood
    * l is log-likelihood
    * L is likelihood



* `L1 regularization  (LASSO Regression)`: In general we want the X matrix to be skinny meaning D<<N (# features << # samples) 
    * Goal: select a small number of important features that predict the trend and remove the features that are noise
    * We use `L1 regularization` mainly when we have a fat data matrix X and 
      want to create `sparsity` so most of the weights become zero
      * This also puts a prior on w, so it's also a `MAP estimation` of w
          
    - `L1 (LASSO) regularization` uses L1 norm for penalty term    
        - J_LASSO = sum(y_n-y_hat_n)^2 + lambda.|w|
        - J = (Y - X.w)_T.(Y - X.w) + lambda.|w|
        - J = Y_T.Y - 2.Y_T.X.w + w_T.X_T.X.w + lambda.|w|
        - dJ/dw = - 2.X_T.Y + 2.X_T.X.w + lambda.sign(w) = 0 [sign(1)=1 if x> 0; sign(x)=-1 if x< 0; sign(1)=0 if x=0
        
           * We have a Laplace distribution meaning exp of negative of absolute value.
             unlike in L2 in L1 we don't have a gaussian prior because we dont have exp of negative square anymore 
           * Prior (Laplace distribution) : p(w) = (lambda/2)exp(-lambda.|w|)
                   
    - `L2 (RIDGE) regularization` used L2 norm for penalty term
        - J_RIDGE = sum(y_n-y_hat_n)^2 + lambda|w|^2
        - J = (Y - X.w)_T.(Y - X.w) + lambda.w_T.w
        - J = Y_T.Y - 2.Y_T.X.w + w_T.X_T.X.w + lambda.w_T.w
        - dJ/dw = - 2.X_T.Y + 2.X_T.X.w + 2.lambda.w = 0
                
          * Both Likelihood and Prior are Gausian because they contain exp of negative of a square 
          * Likelihood (Gaussian distribution) : P(Y|X,w) = mult((1/sqrt(2.pi.sig^2)) exp(0.5 (y_i - w_T x_n)^2/sig^2)) for n in [1 , N)
          * Prior (Gaussian distribution) : p(w) = sqrt(lambda/2.pi) exp(- 0.5 lambda w_T w)
        
     
* `Maximum likelihood (ML)` solution: `Minimizing squared error` is the same as `maximizing log-likelihood` 
    - J = sum(y_n-y_hat_n)^2
    * dJ/dw = - 2.X_T.Y + 2.X_T.X.w 
    * w_ml = (X.T.dot(X))^-1 + X_T.Y
    In Python:
    * w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    * Yhat_ml = X.dot(w_ml)
    
* `Maximum A Posteriori (MAP)` solution through `L2 regularization (Ridge Regression)`.
    * `L2 regularization (Ridge Regression)`: it helps with reducing the model complexity and prevent us from over-fitting to outliers. 
        Data may have outliers that pull the line away from the main trend in order to minimize the square error. 
        Hence we don't want very large weights because that might lead to fitting to outliers to minimize the squared error. 
        As a result we add a penalty for large weights
        * L2 regularization penalty: lambda|w|^2: to do this we add a lambda multiplied by squared norm of the weights:
         - J_RIDGE = sum(y_n-y_hat_n)^2 + lambda|w|^2
         - J = Y_T.Y - 2.Y_T.X.w + w_T.X_T.X.w + lambda.w_T.w
         - dJ/dw = - 2.X_T.Y + 2.X_T.X.w + 2.lambda.w = 0
        * w_MAP = (lambda.I + X_T.X)^-1 + X_T.Y
        In Python:
        * w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
        * Yhat_map = X.dot(w_map)
         
* `Gradient Descent` solution you have a function you want to minimize J(w)=cost/error
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
       - for t = [1,T]:                                                                                
       -  w = w - learning_rate * X_T.(Y_hat - Y)
      python:
       -  Yhat = X.dot(w)
       -  delta = Yhat - Y
       -  w = w - learning_rate*X.T.dot(delta)
       -  mse = delta.dot(delta) / N
       -  costs.append(mse)
      
     > can quit after a number of steps or when change in w is smaller tthan a predetermined threshold
     - If learning rate is too big: won't converge (bounce back and forth across the optima)
     - If learning rate is too small: gradient descent will be too slow 
     
     
     
 * Matrix vertical concat: 

                        - np.vstack([[0]*5 , [1]*5])
                        - shape:  (2, 5)                       
                        - array([[0, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1]])           
                        
                        - np.vstack([[[1,22,33],[1,2,3]] , [[10,220,330],[10,20,30]]]).shape
                        - shape:   (4,3)
                        - array([[  1,  22,  33],
                               [  1,   2,   3],
                               [ 10, 220, 330],
                               [ 10,  20,  30]]) 

 * Matrix concat:  

                        - np.array([0]*5 + [1]*5)
                        - shape:   (10,) 
                        - array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
                        
                        - np.array([[1,22,33],[1,2,3]] + [[10,220,330],[10,20,30]])
                        - shape:   (4,3) 
                        - array([[  1,  22,  33],
                               [  1,   2,   3],
                               [ 10, 220, 330],
                               [ 10,  20,  30]]) 
                               
                               
 ## Free Data
 
 * Governemnt data: 
 https://www.data.gov
 https://data.gov.in/
 http://open.canada.ca