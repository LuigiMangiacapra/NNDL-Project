#Here all the error functions
import numpy as np

# - y are the model responses organizied as a matrix cxN, c is the
#   number of output values, N is the number of samples
# - t are targets organizied as a matrix cxN, c is the
#   number of target values (i.e, it corresponds to the numebr
#   of output neurons), N is the number of samples
# - der is a flag, if der>0, the function returns the derivatives of
#   the error function with respect the output.
# - the output is the standard sum-of-squares (if der==0)
def sumOfSquares(y,t,der=0):
    #x.shape
    #t.shape
    z= y-t
    if der==0:
        return (1/2)*np.sum(np.power(z,2))
    else:
        return z

#soft-max    
def softMax(y):
    #soft max is computing considering overflow
    y_exp=np.exp(y-y.max(0))
    z= y_exp/sum(y_exp,0) #here is soft-max
    return z

#Cross-Entropy with soft-max
# - y are the model responses organizied as a matrix cxN, c is the
#   number of output values, N is the number of samples
# - t are targets organizied as a matrix cxN, c is the
#   number of target values (i.e, it corresponds to the numebr
#  of output neurons, N is the number of samples
# - der is a flag, if der>0, the function returns the derivatives of
#   the error function with respect the output.
# - the output is the multiple-class cross-entroy function error with soft-max
#   CE_err = - Sum_j Sum_i t_{i,j} log (z_{i,j})    

def crossEntropyMCSoftMax(y,t,der=0, epsilon=1e-15):
    #soft max is computing considering overflow
    z=softMax(y)
    
    # Add epsilon to avoid division by zero and invalid multiplication
    z = np.clip(z, epsilon, 1 - epsilon)
    
    if der==0:
        #here the cross-entropy with soft-max is computed
        return -(t*np.log(z)).sum()
    else:
        return z-t
    
    