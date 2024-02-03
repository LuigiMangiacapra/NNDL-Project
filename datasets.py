import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np



def get_mnist_validation(data, n):
    data_val = data[0:11999].T
    Y_val = data_val[0]  # Etichette di dev
    Y_val=get_mnist_labels(Y_val) #numero di etichette ridotto a 10
    X_val = data_val[1:n]  # Dati di input di dev
    X_val = X_val / 255.  # Normalizzazione dei dati divisi per 255
    return X_val, Y_val



def get_mnist_training(data, n, m):
    data_train = data[12000:m].T
    Y_train = data_train[0]  # Etichette di training
    Y_train=get_mnist_labels(Y_train) #numero di etichette ridotto a 10
    X_train = data_train[1:n]  # Dati di input di training
    X_train = X_train / 255.  # Normalizzazione dei dati divisi per 255
    return X_train, Y_train


def get_mnist_testing(data, n, m):
    data_test = data[0:m].T
    Y_test = data_test[0]  # Etichette di testing
    Y_test=get_mnist_labels(Y_test) #numero di etichette ridotto a 10
    X_test = data_test[1:n]  # Dati di input di testing
    return X_test, Y_test

def get_mnist_labels(labels):
    labels = np.array(labels)
    one_hot_labels = np.zeros((10, labels.shape[0]), dtype=int)

    for n in range(labels.shape[0]):
        label = labels[n]
        one_hot_labels[label][n] = 1

    return one_hot_labels

def showMnistImage(x):
    xx=x.reshape((28,28))
    plt.imshow(xx,'gray')
    plt.show()

