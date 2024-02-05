#my network library
import numpy as np
import activationFunctions as af
import errorFunctions as errfun
import matplotlib.pyplot as plt
from neuralNetwork import NeuralNetwork

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
        
        
def copy_params_in_network(destination_network, source_network):
    
    for l in range(len(source_network.layers_weights)):
        destination_network.layers_weights[l] = source_network.layers_weights[l].copy()
        destination_network.layers_bias[l] = source_network.layers_bias[l].copy()
    destination_network.hidden_activation_functions = source_network.hidden_activation_functions


def get_net_structure(network, show=0):
    
    num_hidden_layers = network.number_of_hidden_layers
    input_size = network.layers_weights[0].shape[1]
    output_size = network.layers_weights[num_hidden_layers].shape[0]
    num_neurons_hidden_layers = [network.layers_weights[i].shape[0] for i in range(num_hidden_layers)]
    activation_functions = [network.hidden_activation_functions[i].__name__ for i in range(num_hidden_layers)] + [network.hidden_activation_functions[num_hidden_layers].__name__]
    error_function = network.error_function.__name__

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
        print('error_function:', error_function)
    return


# This function creates a new instance of the network
def duplicate_network(net):
    """ n_layer=net.number_of_hidden_layers
    W=[]
    B=[]
    for l in range(n_layer):
        W.append(net.layers_weights[l].copy())
        B.append(net.layers_bias[l].copy()) """
    newNet=deepcopy(net)
    return newNet



def forward_propagation(network, x):
    
    weights = network.layers_weights
    biases = network.layers_bias
    activation_functions = network.hidden_activation_functions
    num_layers = len(network.layers_weights)
    #output_function = network.output_activation_function
    
    z = x
    for l in range(num_layers):
        a = np.matmul(weights[l], z) + biases[l]
        z = activation_functions[l](a)
    """ a = np.matmul(weights[num_layers], z) + biases[num_layers]
    z = output_function(a) """

    return z


def gradient_descent(network, x):
    
    weights = network.layers_weights
    biases = network.layers_bias
    activation_functions = network.hidden_activation_functions
    num_layers = len(network.layers_weights)

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

    # Extracting network parameters
    weights = network.layers_weights
    biases = network.layers_bias
    activation_functions = network.hidden_activation_functions
    depth = len(network.layers_weights)
    
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
    
    for l in range(len(network.layers_weights)):
        for k in range(len(derW[l])):
            for m in range(len(derW[l][k])):
                # If the derivative has the same sign, increase delta, else decrease it
                if oldDerW[l][k][m] * derW[l][k][m] > 0: 
                    deltaW[l][k][m] = min(deltaW[l][k][m] * posEta, stepSizesPlus)
                    
                elif oldDerW[l][k][m] * derW[l][k][m] < 0:
                    deltaW[l][k][m] = max(deltaW[l][k][m] * negEta, stepSizesMinus)
                oldDerW[l][k][m] = derW[l][k][m]

        # Update weights using the sign of derivatives and step sizes
        network.layers_weights[l] -= np.sign(derW[l]) * deltaW[l]
            
    for l in range(len(network.layers_bias)):
        for k in range(len(derB[l])):
            # If the derivative has the same sign, increase delta, else decrease it
            if oldDerB[l][k][0] * derB[l][k][0] > 0:
                deltaB[l][k][0] = min(deltaB[l][k][0] * posEta, stepSizesPlus)
                
            elif oldDerB[l][k][0] * derB[l][k][0] < 0:
                deltaB[l][k][0] = max(deltaB[l][k][0] * negEta, stepSizesMinus)
            oldDerB[l][k][0] =  derB[l][k][0]

        # Update biases using the sign of derivatives and step sizes
        network.layers_bias[l] -= np.sign(derB[l]) * deltaB[l]

    return network



def train_neural_network(net, X_train, Y_train, X_val=[], Y_val=[], max_epochs=100, learning_rate=0.1):
    # Initialization of the learning process
    training_errors = []
    validation_errors = []
    num_hidden_layers = net.number_of_hidden_layers
    error_function = net.error_function

    # Previous epoch weights and biases
    delta_weights, delta_biases, old_derivative_weights, old_derivative_biases = None, None, None, None

    # Epoch 0
    # Training phase
    Y_net_train = forward_propagation(net, X_train)
    train_error = error_function(Y_net_train, Y_train)
    training_errors.append(train_error)

    # Validation phase on unseen data
    if len(X_val) > 0:
        Y_net_val = forward_propagation(net, X_val)
        val_error = error_function(Y_net_val, Y_val)
        validation_errors.append(val_error)

        min_val_error = val_error
        best_net = duplicate_network(net)
        
        print(f'Epoch: 0, Train Error: {train_error}, Train Accuracy: {compute_accuracy(Y_net_train, Y_train)}, '
              f'Val Error: {val_error}, Val Accuracy: {compute_accuracy(Y_net_val, Y_val)}')
    else:
        print(f'Epoch: 0, Train Error: {train_error}, Train Accuracy: {compute_accuracy(Y_net_train, Y_train)}')

    # Start the training phase
    for epoch in range(max_epochs):
        # Gradient descent
        layer_z, layer_da = gradient_descent(net, X_train)
        derivative_weights, derivative_biases = back_propagation(net, layer_da, layer_z, Y_train, error_function)

        # Initialize weights and biases for the first epoch
        if epoch == 0:
            delta_weights = [[[0.1 for _ in row] for row in sub_list] for sub_list in derivative_weights]
            delta_biases = [[[0.1 for _ in row] for row in sub_list] for sub_list in derivative_biases]
            old_derivative_weights = [[[0 for _ in row] for row in sub_list] for sub_list in derivative_weights]
            old_derivative_biases = [[[0 for _ in row] for row in sub_list] for sub_list in derivative_biases]

        # Update the network using Rprop training phase
        net = rprop_training_phase(net, derivative_weights, derivative_biases, delta_weights, delta_biases,
                                   old_derivative_weights, old_derivative_biases)

        # Forward propagation for training set
        Y_net_train = forward_propagation(net, X_train)
        train_error = error_function(Y_net_train, Y_train)
        training_errors.append(train_error)

        # Validation phase
        Y_net_val = forward_propagation(net, X_val)
        val_error = error_function(Y_net_val, Y_val)
        validation_errors.append(val_error)

        # Find minimum error and the best network
        if val_error < min_val_error:
            min_val_error = val_error
            best_net = duplicate_network(net)

        print(f'Epoch: {epoch + 1}, Train Error: {train_error}, Train Accuracy: {compute_accuracy(Y_net_train, Y_train)}, '
              f'Val Error: {val_error}, Val Accuracy: {compute_accuracy(Y_net_val, Y_val)}', end='\r')

    if len(X_val) > 0:
        copy_params_in_network(net, best_net)

    return training_errors, validation_errors

 
    
def compute_accuracy(predictions, targets):
   
    num_samples = targets.shape[1]

    # Applica la funzione softmax alle previsioni della rete
    softmax_predictions = errfun.softmax(predictions)

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


def test_prediction(network, train_mia_net, x, Xtest):
    ix = np.reshape(Xtest[:,x],(28,28))
    plt.figure()
    plt.imshow(ix, 'gray')
    y_net_trained=forward_propagation(train_mia_net, Xtest[:,x:x+1])
    y_net=forward_propagation(network, Xtest[:,x:x+1])
    y_net=errfun.softmax(y_net)
    y_net_trained=errfun.softmax(y_net_trained)
    print('y_net:', y_net)
    print('y_net_trained:', y_net_trained)
