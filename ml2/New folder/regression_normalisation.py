import numpy as np
import pandas as pd
import scipy.optimize as opt
import fuctions as func
import matplotlib.pyplot as plt

data=pd.read_csv("ex2data2.txt")
print(data.head())
data.columns=["x","y","z"];
print(data.head())
m=data.shape
X=pd.DataFrame(data.loc[:,"x":"y"])
y=pd.DataFrame(data.loc[:,"z"])
X.insert(0,"ones",value=np.ones((m[0],1)))
print(X.head())
print(y.head())

##ploting graph

pos=pd.DataFrame(X.iloc[np.where(data["z"]==1)])
neg=pd.DataFrame(X.iloc[np.where(data["z"]==0)])

plt.scatter(pos.iloc[:,1],pos.iloc[:,2],marker="+",label="admitted")
plt.scatter(neg.iloc[:,1],neg.iloc[:,2],label="not admitted")
plt.legend()
plt.xlabel("score in 1 test")
plt.ylabel("score in 2 test")
#plt.show()
## features initlisation
degree = 6
x1 = X['x']  
x2 = X['y']

for i in range(1, degree):  
    for j in range(0, i):
        X['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

X.drop('x', axis=1, inplace=True)  
X.drop('y', axis=1, inplace=True)

print(X.head())


## theata intilisation
m=X.shape
theta= pd.DataFrame(np.zeros((m[1],1) ))


## optimising theta
result=opt.fmin_tnc(func.costreg,x0=theta,args=(X,y,0.0124))
print(result)
pri=func.predict(result[0],X)
sum1=0
l=len(pri)
print(l)
for i in range(l):
    if(int(y.iloc[i])==int(pri[i])):
        sum1+=1
    
    
 
accuracy = (sum1) % len(pri)

print (sum1,'accuracy = {0}%'.format(accuracy) )




