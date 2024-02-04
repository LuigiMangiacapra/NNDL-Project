#my network library
import numpy as np
import activationFunctions as af
import errorFunctions as myerr
import copy as cp

from copy import deepcopy
# ************************************
# Defined Functions:
# - new_net(in_size,hidden_size,out_size)
# - set_weights(net,n_layer,weight_m, bias=0)
# - get_weights(net,i=0)
# - get_net_structure(net,show=0)
# - get_biases(net,i=0)
# - forward_prop(net,x)
# - train_forward_prop(net,x)
# - back_prop(net,x,t,err_fun)
#**************************************

def new_network(input_size, hidden_sizes, output_size):
    """
    Create a new neural network with random weights and biases.

    Parameters:
    - input_size: Number of input neurons.
    - hidden_sizes: List containing the number of neurons for each hidden layer.
    - output_size: Number of output neurons.

    Returns:
    - neural_network: Dictionary containing network parameters (weights, biases, activation functions, depth).
    """

    sigma = 0.001
    weights = []
    biases = []
    activation_functions = []
    prev_layer_size = input_size

    # Create weights, biases, and activation functions for hidden layers
    for hidden_size in hidden_sizes:
        biases.append(sigma * np.random.normal(size=[hidden_size, 1]))
        weights.append(sigma * np.random.normal(size=[hidden_size, prev_layer_size]))
        activation_functions.append(af.tanh)  # Using tanh as activation function for hidden layers
        prev_layer_size = hidden_size

    # Create weights, biases, and activation function for the output layer
    weights.append(sigma * np.random.normal(size=[output_size, prev_layer_size]))
    biases.append(sigma * np.random.normal(size=[output_size, 1]))
    activation_functions.append(af.identity)  # Using identity as activation function for the output layer

    # Create the neural network dictionary
    neural_network = {'W': weights, 'B': biases, 'ActFun': activation_functions, 'Depth': len(weights)}

    return neural_network


def set_weights(network, layer_indices, weight_matrices, bias=0):
    """
    Set weights or biases for specific layers in a neural network.

    Parameters:
    - network: Dictionary containing network parameters (weights, biases).
    - layer_indices: Single layer index or list of layer indices to set weights or biases.
    - weight_matrices: Single weight matrix or list of weight matrices to set.
    - bias: Flag indicating whether to set weights (0) or biases (1).

    Returns:
    - Updated neural network dictionary.
    """

    # Check if layer_indices is a scalar or a list
    if np.isscalar(layer_indices):
        if bias == 0:
            network['W'][layer_indices - 1] = weight_matrices.copy()
        else:
            network['B'][layer_indices - 1] = weight_matrices.copy()
    else:
        count = 0
        # Set weights or biases for multiple layers
        if bias == 0:
            for i in layer_indices:
                network['W'][i - 1] = weight_matrices[count].copy()
                count += 1
        else:
            for i in layer_indices:
                network['B'][i - 1] = weight_matrices[count].copy()
                count += 1

    return network


def set_activation_function(network, layer_indices=[], activation_function=af.tanh, layer_type=0):
    """
    Set activation function for specific layers or all layers in a neural network.

    Parameters:
    - network: Dictionary containing network parameters (activation functions).
    - layer_indices: Single layer index or list of layer indices to set activation function.
    - activation_function: Activation function to set.
    - layer_type: Flag indicating the type of layers to set activation function (0 for specific layers, 1 for all hidden layers, 2 for output layer).

    Returns:
    - Updated neural network dictionary.
    """

    if layer_type == 0:  # Set activation function for specific layer(s)
        if np.isscalar(layer_indices):
            network['ActFun'][layer_indices - 1] = activation_function
        else:
            count = 0
            for i in layer_indices:
                network['ActFun'][i - 1] = activation_function[count]
                count += 1
    elif layer_type == 1:  # Set activation function for all hidden layers
        for i in range(network['Depth'] - 1):  # Exclude the output layer
            network['ActFun'][i] = activation_function
    else:  # Set activation function for the output layer
        network['ActFun'][network['Depth'] - 1] = activation_function

    return network
        
        
def copy_params_in_network(destination_network, source_network):
    """
    Copy parameters (weights, biases, activation functions) from one network to another.

    Parameters:
    - destination_network: Dictionary containing destination network parameters.
    - source_network: Dictionary containing source network parameters.

    Returns:
    - None
    """
    for l in range(len(source_network['W'])):
        destination_network['W'][l] = source_network['W'][l].copy()
        destination_network['B'][l] = source_network['B'][l].copy()
    destination_network['ActFun'] = source_network['ActFun']


def get_net_structure(network, show=0):
    """
    Return the structure of the neural network.

    Parameters:
    - network: Dictionary containing network parameters.
    - show: Flag indicating whether to print the network structure.

    Returns:
    - None
    """
    num_hidden_layers = len(network['W']) - 1
    input_size = network['W'][0].shape[1]
    output_size = network['W'][num_hidden_layers].shape[0]
    num_neurons_hidden_layers = [network['W'][i].shape[0] for i in range(num_hidden_layers)]
    activation_functions = [network['ActFun'][i].__name__ for i in range(num_hidden_layers)] + [network['ActFun'][num_hidden_layers].__name__]

    if show > 0:
        print('num_hidden_layers: ', num_hidden_layers)
        print('input_size: ', input_size)
        print('output_size: ', output_size)
        print('neurons in hidden layers:')
        for neurons in num_neurons_hidden_layers:
            print(neurons)
        print('activation functions:')
        for act_fun in activation_functions:
            print(act_fun)

    return


# This function creates a new instance of the network
def duplicateNetwork(net):
    n_layer=len(net['W'])
    W=[]
    B=[]
    for l in range(n_layer):
        W.append(net['W'][l].copy())
        B.append(net['B'][l].copy())
    newNet={'W':W,'B':B,'ActFun':net['ActFun'],'Depth':n_layer}
    return newNet

def get_weights(net,i=0):
    W=net['W']
    if (i>0):
        return W[i-1]
    else:
        return W

def get_biases(net,i=0):
    B=net['B']
    if (i>0):
        return B[i-1]
    else:
        return B

def get_act_fun(net,i=0):
    AF=net['ActFun']
    if (i>0):
        return AF[i-1]
    else:
        return AF


def forward_propagation(network, x):
    """
    Forward propagation through the neural network.

    Parameters:
    - network: Dictionary containing network parameters (weights, biases, activation functions).
    - x: Input data organized as a matrix dxN, where d is the number of features, and N is the number of samples.

    Returns:
    - Output of the last layer.
    """
    weights = get_weights(network)
    biases = get_biases(network)
    activation_functions = get_act_fun(network)
    num_layers = network['Depth']

    z = x
    for l in range(num_layers):
        a = np.matmul(weights[l], z) + biases[l]
        z = activation_functions[l](a)

    return z


def gradient_descent(network, x):
    """
    Compute the forward pass and derivatives of activation functions for each layer.

    Parameters:
    - network: Dictionary containing network parameters (weights, biases, activation functions).
    - x: Input data organized as a matrix dxN, where d is the number of features, and N is the number of samples.

    Returns:
    - List of layer outputs (including input and last layer).
    - List of derivatives of activation functions computed at the input of each layer.
    """
    weights = get_weights(network)
    biases = get_biases(network)
    activation_functions = get_act_fun(network)
    num_layers = network['Depth']

    a = []
    layer_outputs = []
    activation_derivatives = []
    layer_outputs.append(x)

    for l in range(num_layers):
        a.append(np.matmul(weights[l], layer_outputs[l]) + biases[l])
        z, d_act = activation_functions[l](a[l], 1)
        activation_derivatives.append(d_act)
        layer_outputs.append(z)

    return layer_outputs, activation_derivatives



def back_propagation(network, input_activations, layer_outputs, target, error_function):
    """
    Backward propagation algorithm to update weights and biases in a neural network.

    Parameters:
    - network: Dictionary containing network parameters (weights, biases, depth).
    - input_activations: List of input activations for each layer.
    - layer_outputs: List of output activations for each layer.
    - target: Target output for the given input.
    - error_function: Error function used to calculate the derivative of the error.

    Returns:
    - weight_gradients: List of weight gradients for each layer.
    - bias_gradients: List of bias gradients for each layer.
    """

    # Extracting network parameters
    weights = network['W']
    biases = network['B']
    depth = network['Depth']

    # Calculate delta for the last layer
    output_error_derivative = error_function(layer_outputs[-1], target, 1)
    delta = [input_activations[-1] * output_error_derivative]

    # Calculate delta for the previous layers
    for l in range(depth - 1, 0, -1):
        error_derivative = input_activations[l - 1] * np.matmul(weights[l].transpose(), delta[0])
        delta.insert(0, error_derivative)

    # Calculate weight and bias gradients
    weight_gradients = []
    bias_gradients = []

    for l in range(depth):
        weight_gradient = np.matmul(delta[l], layer_outputs[l].transpose())
        weight_gradients.append(weight_gradient)

        bias_gradient = np.sum(delta[l], axis=1, keepdims=True)
        bias_gradients.append(bias_gradient)

    return weight_gradients, bias_gradients

    

def rprop_training_phase(network, derW, derB, deltaW, deltaB, oldDerW, oldDerB, posEta=1.2, negEta=0.5, stepSizesPlus=50, stepSizesMinus=0.00001):
    """
    Update weights and biases using the Rprop training algorithm for one training phase.

    Parameters:
    - network: Dictionary containing network parameters (weights, biases).
    - derW: List of weight derivatives.
    - derB: List of bias derivatives.
    - deltaW: List of weight step sizes.
    - deltaB: List of bias step sizes.
    - oldDerW: List of previous weight derivatives.
    - oldDerB: List of previous bias derivatives.
    - posEta: Positive learning rate.
    - negEta: Negative learning rate.
    - stepSizesPlus: Maximum step size for weight increase.
    - stepSizesMinus: Minimum step size for weight decrease.

    Returns:
    - Updated neural network dictionary.
    """
    for l in range(len(network['W'])):
        for k in range(len(derW[l])):
            for m in range(len(derW[l][k])):
                # If the derivative has the same sign, increase delta, else decrease it
                if oldDerW[l][k][m] * derW[l][k][m] > 0: 
                    deltaW[l][k][m] = min(deltaW[l][k][m] * posEta, stepSizesPlus)
                    
                elif oldDerW[l][k][m] * derW[l][k][m] < 0:
                    deltaW[l][k][m] = max(deltaW[l][k][m] * negEta, stepSizesMinus)
                oldDerW[l][k][m] = derW[l][k][m]

        # Update weights using the sign of derivatives and step sizes
        network['W'][l] -= np.sign(derW[l]) * deltaW[l]
            
    for l in range(len(network['B'])):
        for k in range(len(derB[l])):
            # If the derivative has the same sign, increase delta, else decrease it
            if oldDerB[l][k][0] * derB[l][k][0] > 0:
                deltaB[l][k][0] = min(deltaB[l][k][0] * posEta, stepSizesPlus)
                
            elif oldDerB[l][k][0] * derB[l][k][0] < 0:
                deltaB[l][k][0] = max(deltaB[l][k][0] * negEta, stepSizesMinus)
            oldDerB[l][k][0] =  derB[l][k][0]

        # Update biases using the sign of derivatives and step sizes
        network['B'][l] -= np.sign(derB[l]) * deltaB[l]

    return network



def trainingPhase(net,XTrain,YTrain,XVal=[],YVal=[], maxNumEpoches=100, 
                errFun=myerr.crossEntropyMCSoftMax, eta=0.1):
    ######## Initialization of the learning process#####
    errTot=[]
    errValTot=[]
    dep=net['Depth']

    # Pesi dell'epoca precedente
    deltaW, deltaB, oldDerW, oldDerB = None, None, None, None
        
    #Epoca 0
    #Fase di Training
    Ynet=forward_propagation(net,XTrain)
    err=errFun(Ynet,YTrain) # la funzione di errore prende in input Ynet e utilizza YTrain come target da confrontare ad Ynet
    errTot.append(err)
    
    #Fase di Validation su dati non visti durante il training
    if len(XVal) > 0:
        Ynet_val=forward_propagation(net,XVal)
        err_val=errFun(Ynet_val,YVal)
        errValTot.append(err_val)
        
        minErrVal=err_val
        bestNet=duplicateNetwork(net)
        print('Epoch:',0,
                      'Train Err:',err,
                      'Train Accuracy:',compute_accuracy(Ynet,YTrain),
                      'Val Err:',err_val,
                      'Val Accuracy:',compute_accuracy(Ynet_val,YVal)
                     )
    else:
        print('Epoch:',0,'Train Err:',err,
              'Train Accuracy:',compute_accuracy(Ynet,YTrain))
        
    #Inizio fase di Training
    for epoch in range(maxNumEpoches):
        
        l_z,l_da=gradient_descent(net,XTrain)
            
        derW,derB=back_propagation(net,l_da,l_z,YTrain,errFun)
        
        if(epoch == 0):
            for l in range(dep):
                net['W'][l]=net['W'][l]-eta*derW[l]
                net['B'][l]=net['B'][l]-eta*derB[l]
                
            deltaW = [[[0.1 for _ in row] for row in sub_list] for sub_list in derW]

            deltaB = [[[0.1 for _ in row] for row in sub_list] for sub_list in derB]

            oldDerW = deepcopy(derW)
            oldDerB = deepcopy(derB)
        else:
            #print('\ndeltaW1: ',deltaW)
            net=rprop_training_phase(net, derW, derB, deltaW, deltaB, oldDerW, oldDerB)
            #print('\ndeltaW2: ',deltaW)
        ##############################################################################################
        Ynet=forward_propagation(net,XTrain)
        err=errFun(Ynet,YTrain)
        errTot.append(err)
        
        #Fase di validation
        Ynet_val=forward_propagation(net,XVal)
        err_val=errFun(Ynet_val,YVal)
        errValTot.append(err_val)
            
        #Cerca l'errore minimo e la rete migliore
        if err_val < minErrVal:
            minErrVal=err_val
            bestNet=duplicateNetwork(net)
                
        print('Epoch:',epoch+1,
                'Train Err:',err,
                'Train Accuracy:',compute_accuracy(Ynet,YTrain),
                'Val Err:',err_val,
                'Val Accuracy:',compute_accuracy(Ynet_val,YVal),end=''
            )
        print('\r', end='') 
            
    if len(XVal) > 0:
        copy_params_in_network(net,bestNet)
    return errTot,errValTot
 
    
def compute_accuracy(predictions, targets):
    """
    Calcola l'accuratezza di un modello.

    Args:
    - predictions: Array delle previsioni del modello (output della rete neurale)
    - targets: Array degli obiettivi desiderati

    Returns:
    - accuracy: Percentuale di accuratezza del modello
    """
    num_samples = targets.shape[1]

    # Applica la funzione softmax alle previsioni della rete
    softmax_predictions = myerr.softMax(predictions)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne
    predicted_classes = np.argmax(softmax_predictions, axis=0)

    # Trova l'indice dell'elemento di valore massimo lungo l'asse delle colonne negli obiettivi desiderati
    target_classes = np.argmax(targets, axis=0)

    # Confronta gli indici predetti con gli indici degli obiettivi desiderati e calcola l'accuratezza
    correct_predictions = np.sum(predicted_classes == target_classes)
    accuracy = correct_predictions / num_samples

    return accuracy
                  
def netAccuracy(net,X,target):
    y_net=forward_propagation(net,X)
    return compute_accuracy(y_net,target)

