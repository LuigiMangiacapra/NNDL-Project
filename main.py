import numpy as np
import learningFunctions as mylib
import errorFunctions as myerr
import activationFunctions as myact
import datasets as ds
import matplotlib.pyplot as plt
import pandas as pd
import os

#Costruzione path
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'Data')
train_file_path = os.path.join(data_path, "mnist_train.csv")
test_file_path = os.path.join(data_path, "mnist_test.csv")

#Lettura csv
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

#Ottenimento array per il training e il testing
train_array = np.array(train_data)
test_array = np.array(test_data)

#Estrapoliamo il numero di righe e di colonne di train e test
m, n = train_array.shape 
mtest, ntest = test_array.shape  

np.random.shuffle(train_array)  # Mescola casualmente i dati prima di suddividerli in set di sviluppo e training

Xval, Yval = ds.get_mnist_validation(train_array, n)
Xtrain, Ytrain = ds.get_mnist_training(train_array, n, m)
Xtest, Ytest = ds.get_mnist_testing(test_array, ntest, mtest)

print(Xtrain.shape, Ytrain.shape, Xval.shape, Yval.shape, Xtest.shape, Ytest.shape)


#A neural network with just one hidden layer is created
NUM_HIDDEN_NEURONS=[2, 2]
mia_net=mylib.new_net(Xtrain.shape[0],NUM_HIDDEN_NEURONS,Ytrain.shape[0])
mylib.set_actfun(mia_net, act_fun=myact.tanh, layer_type=1)
#mylib.set_actfun(mia_net, act_fun=myact.relu, layer_type=1)
#mylib.set_actfun(mia_net, act_fun=myact.leaky_relu, layer_type=1)
mylib.get_net_structure(mia_net,show=1)

#A copy of the network is made, so one can restore the original neural network,
# if one wants

train_mia_net=mylib.duplicateNetwork(mia_net)
mylib.get_net_structure(train_mia_net,show=1)

#batch training
err,errV=mylib.trainingPhase(train_mia_net,Xtrain,Ytrain,Xval,Yval,maxNumEpoches=100,errFun= myerr.crossEntropyMCSoftMax,eta=0.00001)


plt.figure()
plt.plot(errV, 'r', label='Validation Error')
plt.plot(err, 'b', label='Training Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

#Accuracy on both the training and test set

acc=mylib.netAccuracy(train_mia_net,Xtest,Ytest)
print('test accuracy: ',acc)
acc=mylib.netAccuracy(train_mia_net,Xtrain,Ytrain)
print('train accuracy: ',acc)