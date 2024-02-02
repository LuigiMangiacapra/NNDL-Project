import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np


<<<<<<< HEAD
'''def loadMnist(data_path=None):
    if data_path is None:
        # Ottiene il percorso assoluto del modulo corrente
        current_path = os.path.dirname(os.path.abspath(__file__))
        # Crea il percorso relativo alla cartella "Data" nello stesso livello del file
        data_path = os.path.join(current_path, 'Data')

    train_file_path = os.path.join(data_path, "mnist_train.csv")
    test_file_path = os.path.join(data_path, "mnist_test.csv")

    no_of_different_labels = 10 
    train_data = np.loadtxt(train_file_path, delimiter=",")
    test_data = np.loadtxt(test_file_path, delimiter=",")

    coeff = 1 / 255
    train_imgs = np.array(train_data[:, 1:]) * coeff
    test_imgs = np.array(test_data[:, 1:]) * coeff
    train_labels = np.array(train_data[:, :1])
    test_labels = np.array(test_data[:, :1])

    lr = np.arange(no_of_different_labels)
    train_labels_one_hot = (lr == train_labels).astype(np.int_)
    test_labels_one_hot = (lr == test_labels).astype(np.float_)

    return (
        train_imgs.transpose(),
        train_labels_one_hot.transpose(),
        test_imgs.transpose(),
        test_labels_one_hot.transpose(),
    )'''


def get_mnist_validation(data, n):
    data_val = data[0:10000].T
=======
def get_mnist_validation(data, n):
    data_val = data[0:11999].T
>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
    Y_val = data_val[0]  # Etichette di dev
    Y_val=get_mnist_labels(Y_val) #numero di etichette ridotto a 10
    X_val = data_val[1:n]  # Dati di input di dev
    X_val = X_val / 255.  # Normalizzazione dei dati divisi per 255
    return X_val, Y_val


<<<<<<< HEAD
def get_mnist_training(data, n, m=20000):
    data_train = data[10000:m].T
=======
def get_mnist_training(data, n, m):
    data_train = data[12000:m].T
>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
    Y_train = data_train[0]  # Etichette di training
    Y_train=get_mnist_labels(Y_train) #numero di etichette ridotto a 10
    X_train = data_train[1:n]  # Dati di input di training
    X_train = X_train / 255.  # Normalizzazione dei dati divisi per 255
    return X_train, Y_train

<<<<<<< HEAD
def get_mnist_testing(data, n, m=2500):
=======
def get_mnist_testing(data, n, m):
>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
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

