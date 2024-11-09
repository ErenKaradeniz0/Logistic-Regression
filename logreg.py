# Test script for training a logistic regressiom model
#
# Author: Eric Eaton
#
# This file should run successfully without changes if your implementation is correct
#
import numpy as np
class LogisticRegression:
 
    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        
        first requirement
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
            
        fourth requirement
        '''
        m = len(y)
        h = self.sigmoid(X @ theta)
        cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
        reg_term = (regLambda / (2 * m)) * np.sum(np.square(theta[1:]))  # theta_0'ı düzenlemeye dahil etmemek için
        return cost + reg_term

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
            
            fifth requirement
        '''
        m = len(y)
        h = self.sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        gradient[1:] += (regLambda / m) * theta[1:]  # theta_0 için düzenleme yok
        return gradient
    


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        ** the d here is different from above! (due to augmentation) **
        
        second requirement
        '''
        
        X = np.c_[np.ones((X.shape[0], 1)), X]  # X matrisini genişletip başına birler ekliyoruz
        self.theta = np.zeros(X.shape[1])
        
        for _ in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            new_theta = self.theta - self.alpha * gradient
            if np.linalg.norm(new_theta - self.theta, ord=2) <= self.epsilon:
                break
            self.theta = new_theta


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions, the output should be binary (use h_theta > .5)
            
            third requirement
        '''
        X = np.c_[np.ones((X.shape[0], 1)), X]  # tahminlerde de X'i genişletiyoruz
        return (self.sigmoid(X @ self.theta) >= 0.5).astype(int)


    def sigmoid(self, Z):
        '''
        Applies the sigmoid function on every element of Z
        Arguments:
            Z can be a (n,) vector or (n , m) matrix
        Returns:
            A vector/matrix, same shape with Z, that has the sigmoid function applied elementwise
        
        Six requirement
        '''
        return 1 / (1 + np.exp(-Z))
