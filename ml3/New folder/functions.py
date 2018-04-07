import numpy as np
import pandas as pd
from scipy.optimize import minimize

def sigmoid(z):
    return 1/(1+np.exp(-z));

def cost(theta,X,y,lrate):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    m=X.shape
    sig=sigmoid(X*theta)
    cost=(-1/m[0])*(np.dot(y.T,np.log(sig))+ np.dot((1-y).T,np.log(1-sig)))+((lrate/(2*m[0]))*(np.sum(np.multiply(theta,theata))-(int(theta.item(0,0))*int(theta.item(0,0)))))
    grad=(1/m[0])*(np.dot(X.T,np.subtract(y-sig)))+((lrate/m[0])*theta)
    return cost ,grad
def onevsall(X,y,label,lrate):
    rows = X.shape[0]
    params = X.shape[1]
    all_theta = np.zeros((num_labels, params + 1))
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x
    return all_theta
                                                                              
