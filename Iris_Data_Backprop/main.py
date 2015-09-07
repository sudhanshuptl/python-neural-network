__auther__='Jorgen Grimnes, Sudhanshu Patel'


from sklearn import datasets
import numpy as np
from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, symmetric_elliot_function, elliot_function
from neuralnet import NeuralNet



class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in out training set.
    def __init__(self, features, target):
        self.features = np.array(features)
        self.targets = np.array(target)
#endclass Instance


#-------------------------------------------------------------------------------
##Importing Iris data Set
iris = datasets.load_iris()
X = iris.data[:,]
Y = iris.target
inp=[]
for i in range(0,len(X)): # preprocessing Iris data, in 4 input and 3 output format 
    inp.append([list(X[i])])
    if Y[i]==0:
        y=[1,0,0]
    elif Y[i]==1:
        y=[0,1,0]
    elif Y[i]==2:
        y=[0,0,1]
    inp[i].append(y)
#training sets
training_one =[]
for i in range(len(inp)):
    training_one.append(Instance(inp[i][0],inp[i][1])) #Encapsulation of a `input signal : output signal
#------------------------------------------------------------------------------

n_inputs = 4            # Number of  input feature 
n_outputs = 3           # Number of neuron output
n_hiddens = 8           # Number of neuron at each hidden layer
n_hidden_layers = 2     # number of hidden layer
# here 2 Hidden layer with 8 node each and 1 output layer with 3 node 

#------------------------DEclaration of activation or Transfer function at each layer --------------------------------------#
# specify activation functions per layer eg: [ hidden_layer_1, hidden_layer_2, output_layer ]
activation_functions = [symmetric_elliot_function,]*n_hidden_layers + [ sigmoid_function ]

# initialize the neural network
network = NeuralNet(n_inputs, n_outputs, n_hiddens, n_hidden_layers, activation_functions)
# network is Instance of class Neuralnet

# start training on test set one
network.backpropagation(training_one, ERROR_LIMIT=.05, learning_rate=0.2, momentum_factor=0.2  )

# save the trained network
network.save_to_file( "trained_configuration.pkl" )

# load a stored network configuration
# network = NeuralNet.load_from_file( "trained_configuration.pkl" )

# print out the result
for instance in training_one:
    print instance.features, network.forwordProp( np.array([instance.features]) ), "\ttarget:", instance.targets

