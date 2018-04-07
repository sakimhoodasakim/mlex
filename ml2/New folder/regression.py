import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import fuctions as func

data=pd.read_csv("ex2data1.txt")
data.columns=["x","y","z"]
X=pd.DataFrame(data.loc[:,"x":"y"])

y=pd.DataFrame(data["z"])
size=X.shape

X.insert(0,column="ones",value=np.ones((size[0],1)))
## ploting the data
pos=pd.DataFrame(X.iloc[np.where(data["z"]==1)])
neg=pd.DataFrame(X.iloc[np.where(data["z"]==0)])
plt.scatter(pos.iloc[:,1],pos.iloc[:,2],marker="+",label="passed")

plt.scatter(neg.iloc[:,1],neg.iloc[:,2],label="failed")

plt.xlabel("score in ex1")
plt.ylabel("score in ex2")
plt.legend()
plt.show()
theta=pd.DataFrame(np.zeros((size[1]+1,1))   )

print(X.shape,theta.shape,"in main")
result=opt.fmin_tnc(func=func.cost, x0=theta, args=(X, y))  
cost4=func.cost( result[0],X,y)
print(cost4,result[0])
pri=func.predict(result[0],X)
sum1=0
print(len(pri))
for i in range(99):
    if(int(y.iloc[i])==int(pri[i])):
        sum1+=1
    
    
 
accuracy = (sum1) % len(pri)

print (sum1,'accuracy = {0}%'.format(accuracy) )

## testing model on new data set without normalisation on ex2data2

newdata=pd.read_csv("ex2data2.txt")
newdata.columns=["x","y","z"]

X=pd.DataFrame(newdata.loc[:,"x":"y"])
y=pd.DataFrame(newdata["z"])
m=X.shape
X.insert(0,column="ones",value=np.zeros((m[0],1)))
pri=func.predict(result[0],X)
sum1=0
l=len(pri)
print(l)

for i in range(l):
    if(int(y.iloc[i])==int(pri[i])):
        sum1+=1
    
    
 
accuracy = (sum1) % len(pri)

print (sum1,'accuracy = {0}%'.format(accuracy) )



