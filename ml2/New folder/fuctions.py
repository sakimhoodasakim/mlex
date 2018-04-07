import numpy as np
import pandas as pd


def sigmoid(theta,X):
    X=np.matrix(X)
    theta=np.matrix(theta)
    print(X.shape,theta.shape)
    return 1/(1+np.exp(-1*(X*theta.T)))

def cost(theta,X,y):
    X=np.matrix(X)
    theta=np.matrix(theta)
    y=np.matrix(y)
    m=X.shape
    
    sig=sigmoid(theta,X)
    cost=(-1/m[0])*(y.T*np.log(sig) + (1-y.T)*np.log(1-sig))
    grad=(1/m[0])*(X.T*np.subtract(sig,y))
    return cost,grad
def grad(theta,X,y):
    m=X.shape
    sig=sigmoid(theta,X)
    grad=(1/m[0])*(np.dot(X.T,np.subtract(sig,y)))
    return grad
def predict(theta, X):  
    probability = sigmoid(theta,X)
    return [1 if x >= 0.3 else 0 for x in probability]


def costreg(theta,X,y,lambval):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    m=X.shape
    
    sig=sigmoid(theta,X)
    cost=((-1/m[0])*(y.T*np.log(sig)+(1-y.T)*np.log(1-sig) ))+(((lambval/(2*m[0]))*(np.sum(np.multiply(theta,theta))))-(int(theta.item(0,0))*int(theta.item(0,0))) )
    
    grad=((1/m[0])*(X.T*np.subtract(sig,y)))
    grad+=((lambval/m[0])*theta.T)
    
    print (cost,grad.shape)
    return cost ,grad   

