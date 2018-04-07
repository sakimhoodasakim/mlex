import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as func
from scipy.io import loadmat



data=loadmat('ex3data1.mat')
print(data,data['X'].shape,data['y'].shape)


rows = data['X'].shape[0]  
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])  
y_0 = np.reshape(y_0, (rows, 1))

print(X.shape, y_0.shape, theta.shape, all_theta.shape)
