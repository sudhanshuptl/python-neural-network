## How To Extract Iris Dataset
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,]
Y = iris.target
inp=[]
for i in range(0,len(X)):
    inp.append([list(X[i])])
    if Y[i]==0:
        y=[1,0,0]
    elif Y[i]==1:
        y=[0,1,0]
    elif Y[i]==2:
        y=[0,0,1]
    inp[i].append(y)
   
class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in out training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets = np.array(target)
training_one=[]
for i in range(len(inp)):
    training_one.append(Instance(inp[i][0],inp[i][1]))
