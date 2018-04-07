import pandas as pd
import numpy as np
def costfunc(X,y,theta):
    m=X.shape
    return (1/(2*m[0]))*np.sum(np.square(np.dot(X,theta)-y))[0]
def gradient(X,y,theta,alpha,iterat):
    m=X.shape
    for _ in range(iterat):
        theta=theta-(alpha/m[0])*np.dot(X.T,(np.dot(X,theta)-y))
    return theta
        
