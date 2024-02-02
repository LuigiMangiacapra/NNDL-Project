#Here all the activation functions
import numpy as np

#- x input matrix   
#- der is a flag, der different from 0 the function returns 
#  the derivatives as second value too.
def identity(x,der=0):
    if der==0:
        return x
    # if we want the derivative of the function during the learning phase
    else:
        return x, 1
    

#- x input matrix   
#- der is a flag, der different from 0 the function returns 
#  the derivatives as second value too.
def tanh(x,der=0):
    y = np.tanh(x)
    if der==0:
        return y
    # if we want the derivative of the function during the learning phase
    else:
        return y, 1-y*y

    
#- x input matrix   
#- der is a flag, der different from 0 the function returns 
#  the derivatives as second value too.
def relu(x,der=0):
    if der==0:
        return np.maximum(0, x)
    # if we want the derivative of the function during the learning phase
    else:
        return np.maximum(0, x), np.where(x > 0, 1, 0)
    
#- x input matrix   
#- der is a flag, der different from 0 the function returns 
#- alpha is a hyperparameter that controls the slope of the function for negative inputs
#  the derivatives as second value too.
def leaky_relu(x,der=0,alpha=0.01):
    if der==0:
        return np.where(x > 0, x, alpha*x)
    # if we want the derivative of the function during the learning phase
    else:
        return np.where(x > 0, x, alpha*x), np.where(x > 0, 1, alpha)