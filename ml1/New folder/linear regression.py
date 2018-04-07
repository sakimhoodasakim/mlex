import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import costgradfunc as func
print("sAKIM")
data=pd.read_csv("ex1data1.txt")
data.columns=['x','y']
theta=np.zeros((2,1),dtype='int')
m=data.shape
X=pd.DataFrame(data['x'])
X.insert(0,column="b",value=np.ones((m[0],1)))
y=pd.DataFrame(data['y'])
a=func.costfunc(data,y,theta)
print(data.iloc[1,1])
print(a)
theta=func.gradient(X,y,theta,0.01,1500)
print(theta)
print(data.columns)

plt.scatter(X.iloc[:,1],y,marker="+",alpha=1)
plt.scatter(X.iloc[:,1],np.dot(X,theta),marker=".",alpha=1)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()
       

print("hooda")
