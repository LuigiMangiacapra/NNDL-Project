#my network library
import numpy as np
import activationFunctions as af
import errorFunctions as myerr
import copy as cp
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

# Function to store the main structures for a multi-layer feed-forward neural network
#-in_size is the input size, 
#-hidden_size is an array (list) holding the number of neurons for each hidden layer
#-out_size is the output size
def new_net(in_size,hidden_size,out_size):
    sigma=0.001
    weights=[]
    biases=[]
    act_fun=[]
    j=in_size
    #Note that weights of the l-th layer are memorized 
    # as a matrix m_l x m_{l-1} (where l-1 is the previous
    # neuron layer or the input layer, and the biases as a matrix
    # m_l x 1 
    for i in hidden_size:  # for example: hidden_size=[5,3] hidden_size è la lista di nodi interni per ogni livello, poiché ha un solo elemento allora abbiamo un solo strato interno
        #Creiamo una matrice di bias e di pesi per ogni strato (lunghezza della lista hidden_size)
        biases.append(sigma*np.random.normal(size=[i,1])) 
        weights.append(sigma*np.random.normal(size=[i,j])) 
        j=i
        act_fun.append(af.tanh)
        
    weights.append(sigma*np.random.normal(size=[out_size,j]))
    biases.append(sigma*np.random.normal(size=[out_size,1]))
    act_fun.append(af.identity)
    n_net={'W':weights,'B':biases,'ActFun':act_fun,'Depth':len(weights)}
    return n_net

#- net is the network. Remember that it is an in-out parameter 
#- n_layer is either a list composed of integer, the integer i corresponds
#  to the i-th layer starting from the first hidden layer, or
#  n_layer is just one integer i corresponding
#  to the i-th layer, starting from the first hidden layer
#- weight_m is either a list composed of matrices or
#  a single matrix.
#- bias is a flag. When bias is equal to 0 (default), the function is used
#  to set the weights, if it is equal to 1, the function is used to set the biases.
def set_weights(net,n_layer,weight_m, bias=0):
    if np.isscalar(n_layer):
        if bias==0:
            net['W'][n_layer-1]=weight_m.copy()
        else:
            net['B'][n_layer-1]=weight_m.copy()

    else:
        count=0
        if bias==0:
            for i in n_layer:
                net['W'][i-1]=weight_m[count].copy()
                count=count+1
        else:
            for i in n_layer:
                net['B'][i-1]=weight_m[count].copy()
                count=count+1

#- net is the network. Remember that it is an in-out parameter 
#- n_layer is either a list composed of integer, the integer i corresponds
#  to the i-th layer starting from the first hidden layer, or
#  n_layer is just one integer i corresponding
#  to the i-th layer, starting from the first hidden layer
#- act_f is either a list composed of activation function names or
#  a single activation function name. The activation function must be already implemented.
# - layer_type is a flag. If it is equal to 0 we have the previous behavior, if it is equal to
#.  1, then all the hidden layers are set to the passed activation function. Finally, if it is 
#.  equal to 2, then the activation function of the output layer is only considered.
# Here, some examples of usage:
# x=np.random.normal(size=[5,4])
# mia_net=mylib.new_net(5,[2,4],3)
# mylib.get_net_structure(mia_net,1)
# mylib.set_actfun(mia_net,[1,3],[myact.sigm,myact.identity])
# mylib.get_net_structure(mia_net,1)
# mylib.set_actfun(mia_net,2,myact.relu)
# mylib.get_net_structure(mia_net,1)
# mylib.set_actfun(mia_net,[],myact.sigm,1)
# mylib.get_net_structure(mia_net,1)
# mylib.set_actfun(mia_net,[],myact.tanh,2)
# mylib.get_net_structure(mia_net,1)
# z=mylib.forward_prop(mia_net,x)
def set_actfun(net,n_layer=[],act_fun=af.tanh,layer_type=0):
    if layer_type==0: # set just one layer or a selected number of layers
        if np.isscalar(n_layer):
            net['ActFun'][n_layer-1]=act_fun
        else:
            count=0
            for i in n_layer:
                net['ActFun'][i-1]=act_fun[count]
                count=count+1
    elif layer_type==1: # setting the activation function of all layers whith just one activation function only
        len=net['Depth']
        for i in range(len-1):
            net['ActFun'][i]=act_fun
    else: # setting the activation function of the output layer only
        len=net['Depth']
        net['ActFun'][len-1]=act_fun
        
# This function copies weights, biases and activation function
# from net2 to net1
def copyParamInNetwork(net1,net2):
    for l in range(len(net2['W'])):
        net1['W'][l]=net2['W'][l].copy()
        net1['B'][l]=net2['B'][l].copy()
    net1['ActFun']=net2['ActFun']
    
# This function returns the network structure
def get_net_structure(net,show=0):
    l=len(net['W'])
    num_hidden_layers= l-1
    input_size= net['W'][0].shape[1]
    output_size= net['W'][num_hidden_layers].shape[0]
    num_neurons_hlayers=[]
    act_funs=[]
    for i in range(num_hidden_layers):
        num_neurons_hlayers.append(net['W'][i].shape[0])
        act_funs.append(net['ActFun'][i].__name__)
    act_funs.append(net['ActFun'][num_hidden_layers].__name__)
    if show>0:
        print('num_hidden_layers: ',num_hidden_layers)
        print('input_size: ', input_size)
        print('output_size: ', output_size)
        print('neurons into hidden layers:')
        for i in range(len(num_neurons_hlayers)):
            print(num_neurons_hlayers[i]) 
        print('activation functions:')
        for i in range(len(num_neurons_hlayers)+1):
            print(act_funs[i])
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


# - net is the network
# - x is data organizied as a matrix dxN, d is the
#   number of features, N is the number of samples
# - It returns the output of the last layer
def forward_prop(net,x):
    W=get_weights(net)
    B=get_biases(net)
    AF=get_act_fun(net)
    ll=net['Depth']
    z=x
    for l in range(ll):
        a=np.matmul(W[l],z)+B[l] # moltiplica la matrice di pesi e z (matrici di input) e gli somma la matrice di bias costituita da una colonna
        z=AF[l](a) # assegna la funzione di applicazione all'output appena calcolato
    #Example of usage:
    #x=np.random.normal(size=[5,4])
    #mia_net=mylib.new_net(5,[2,4],3)
    #z=mylib.forward_prop(mia_net,x)
    return z

# - net is the network
# - x are data organizied as a matrix dxN, d is the
#   number of features, N is the number of samples
# - It returns two lists, the first list is composed of
#   the output of each layer including the data input
#   the first element is the data input, the last element
#   of the list is the last layer. 
#   The second list is composed of the derivatives of 
#   activation funcions computed into the input of each 
#   layer
<<<<<<< HEAD
def train_forward_prop(net,x):
=======
def gradient_descent(net,x):
>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
    W=get_weights(net)
    B=get_biases(net)
    AF=get_act_fun(net)
    ll=net['Depth']
    a=[]
    z=[]
    d_act=[]
    z.append(x)
    for l in range(ll):
        a.append(np.matmul(W[l],z[l])+B[l])
        z_c,d_act_c=AF[l](a[l],1)
        d_act.append(d_act_c)
        z.append(z_c)
   # net['Out']=z
   # net['Dact']=d_act
    return z,d_act

# - net is the network
# - x are data organizied as a matrix dxN, d is the
#   number of features, N is the number of samples
# - t are targets organizied as a matrix cxN, c is the
#   number of target values (i.e, it corresponds to the numebr
#.  of output neurons, N is the number of samples
# - It returns the list of derivatives, which is composed of
#   the derivatives of each weight layer
<<<<<<< HEAD
def back_prop(net,x,t,err_fun):
    W=net['W']
    B=net['B']
    dep=net['Depth']
    l_z,l_da=train_forward_prop(net,x)
=======
def back_prop(net,l_da,l_z,t,err_fun):
    W=net['W']
    B=net['B']
    dep=net['Depth']
>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
    #-1 refers to the last element of the list
    d_err=err_fun(l_z[-1],t,1)
    #here delta are computed
    #**************************
    delta=[]
    delta.insert(0,l_da[-1]*d_err)
    for l in range(dep-1,0,-1):
        #print('l:',l)
        # transpose method does not affect the source matrix
        #print('W[l]:',W[l].shape)
        #print('delta[-1]:',delta[-1].shape)
        #print('l_da[l-1]:',l_da[l-1].shape)
        d_c= l_da[l-1]*np.matmul(W[l].transpose(),delta[0])
        delta.insert(0,d_c)
    #**************************
    #here derivatives are computed
    #**************************
    l_der=[]
    b_der=[]
    for l in range(0,dep):
        der_c=np.matmul(delta[l],l_z[l].transpose())
        l_der.append(der_c)
        #FINIRE CALCOLO DERIVATE BIAS, ATTENZIONE FORMATO!!!!
        b_der.append(np.sum(delta[l],1,keepdims=True))
    return l_der,b_der
    

<<<<<<< HEAD
def rpropTrainingPhase(net,XTrain,YTrain,XVal=[],YVal=[], maxNumEpoches=100, errFun=myerr.crossEntropyMCSoftMax):
    # Initialize update values
    deltaW = [0.1]*len(net['W'])
    deltaB = [0.1]*len(net['B'])
    
    oldDerW, oldDerB = [0]*len(net['W']), [0]*len(net['B'])
    
    for epoch in range(maxNumEpoches):
        derW, derB = back_prop(net, XTrain, YTrain, errFun)
        
        for l in range(len(net['W'])):
            # If the derivative has the same sign, increase delta, else decrease it
            deltaW[l] = deltaW[l]*1.2 if oldDerW[l]*derW[l] > 0 else deltaW[l]*0.5
            deltaB[l] = deltaB[l]*1.2 if oldDerB[l]*derB[l] > 0 else deltaB[l]*0.5
            
            # If the derivative has the same sign, update the weight/bias
            if oldDerW[l]*derW[l] > 0:
                net['W'][l] -= np.sign(derW[l])*deltaW[l]
            if oldDerB[l]*derB[l] > 0:
                net['B'][l] -= np.sign(derB[l])*deltaB[l]
            
            # Update the old derivatives
            oldDerW[l], oldDerB[l] = derW[l], derB[l]
    
    return net


    
=======
def rpropTrainingPhase(net, derW, derB, deltaW, deltaB, oldDerW, oldDerB, posEta=1.2, negEta=0.5):
    
    for l in range(len(net['W'])):
        for k in range(len(net['W'][l])):
            for m in range(len(net['W'][l][k])):
                #print('\nderW: ',derW[l][k][m], '\noldDerW: ',oldDerW[l])
                # If the derivative has the same sign, increase delta, else decrease it
                if oldDerW[l] * derW[l][k][m] > 0:
                    deltaW[l] = deltaW[l] * posEta
                    #print('\ndeltaW1: ',deltaW[l])
                else:
                    deltaW[l] = deltaW[l] * negEta
                    #print('\ndeltaW2: ',deltaW[l])
                    
                if oldDerB[l] * derB[l][k][0] > 0:
                    deltaB[l] = deltaB[l] * posEta
                    #print('\ndeltaB1: ',deltaB[l])
                else:
                    deltaB[l] = deltaB[l] * negEta
                    #print('\ndeltaB2: ',deltaB[l])
                    
                
                # If the derivative has the same sign, update the weight/bias
                if oldDerW[l] * derW[l][k][m] > 0:
                    net['W'][l][k][m] -= np.clip(np.sign(derW[l][k][m]) * deltaW[l], -1e10, 1e10)
                    #print('\nnetW: ',net['W'][l][k][m])
                if oldDerB[l] * derB[l][k][0] > 0:
                    # Use a more stable update for the bias
                    net['B'][l][k][0] -= np.clip(np.sign(derB[l][k][0]) * deltaB[l], -1e10, 1e10)
                    #print('\nnetB: ',net['B'][l][k][0])

                # Update the old derivatives
                oldDerW[l], oldDerB[l] = derW[l][k][m], derB[l][k][0]

    return net

'''
#Versione con np.all
def rpropTrainingPhase(net, derW, derB, deltaW, deltaB, oldDerW, oldDerB, posEta=1.2, negEta=0.5):
    
    for l in range(len(net['W'])):
        
        #print('\nderW: ',derW[l][k][m], '\noldDerW: ',oldDerW[l])
        # If the derivative has the same sign, increase delta, else decrease it
        if np.all(oldDerW[l] * derW[l] > 0):
            deltaW[l] = deltaW[l] * posEta
            #print('\ndeltaW1: ',deltaW[l])
        else:
            deltaW[l] = deltaW[l] * negEta
            #print('\ndeltaW2: ',deltaW[l])
                    
        if np.all(oldDerB[l] * derB[l] > 0):
            deltaB[l] = deltaB[l] * posEta
            #print('\ndeltaB1: ',deltaB[l])
        else:
            deltaB[l] = deltaB[l] * negEta
            #print('\ndeltaB2: ',deltaB[l])
                    
                
        # If the derivative has the same sign, update the weight/bias
        if np.all(oldDerW[l] * derW[l]> 0):
            net['W'][l] -= np.clip(np.sign(derW[l]) * deltaW[l], -1e10, 1e10)
            #print('\nnetW: ',net['W'][l][k][m])
        if np.all(oldDerB[l] * derB[l] > 0):
            # Use a more stable update for the bias
            net['B'][l]-= np.clip(np.sign(derB[l]) * deltaB[l], -1e10, 1e10)
            #print('\nnetB: ',net['B'][l][k][0])

            # Update the old derivatives
        oldDerW[l], oldDerB[l] = derW[l], derB[l]

    return net
'''


>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
def trainingPhase(net,XTrain,YTrain,XVal=[],YVal=[], maxNumEpoches=100, 
                errFun=myerr.crossEntropyMCSoftMax, eta=0.1):
    ######## Initialization of the learning process#####
    errTot=[]
    errValTot=[]
    dep=net['Depth']
<<<<<<< HEAD
    #bInd=splitKFolds(YTrain)
=======
    # Initialize update values
    deltaW = [0.1] * len(net['W'])
    deltaB = [0.1] * len(net['B'])
    # Pesi dell'epoca precedente
    oldDerW = [0] * len(net['W'])
    oldDerB = [0] * len(net['B'])
>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
        
    #Epoca 0
    #Fase di Training
    Ynet=forward_prop(net,XTrain)
    err=errFun(Ynet,YTrain) # la funzione di errore prende in input Ynet e utilizza YTrain come target da confrontare ad Ynet
    errTot.append(err)
    
    #Fase di Validation su dati non visti durante il training
    if len(XVal) > 0:
        Ynet_val=forward_prop(net,XVal)
        err_val=errFun(Ynet_val,YVal)
        errValTot.append(err_val)
        
        minErrVal=err_val
        bestNet=duplicateNetwork(net)
        print('Epoch:',0,
                      'Train Err:',err,
                      'Train Accuracy:',computeAccuracy(Ynet,YTrain),
                      'Val Err:',err_val,
                      'Val Accuracy:',computeAccuracy(Ynet_val,YVal)
                     )
    else:
        print('Epoch:',0,'Train Err:',err,
              'Train Accuracy:',computeAccuracy(Ynet,YTrain))
        
    #Inizio fase di Training
    for epoch in range(maxNumEpoches):
        
<<<<<<< HEAD
        #Fase di Training
        derW,derB=back_prop(net,XTrain,YTrain,errFun)
        
        #Aggiornament dei pesi
        for l in range(dep):
            net['W'][l]=net['W'][l]-eta*derW[l]
            net['B'][l]=net['B'][l]-eta*derB[l]
        
        Ynet=forward_prop(net,XTrain)
        
=======
        l_z,l_da=gradient_descent(net,XTrain)
        #Fase di Training
        derW,derB=back_prop(net,l_da,l_z,YTrain,errFun)
        
        #Aggiornament dei pesi
        #for l in range(dep):
            #net['W'][l]=net['W'][l]-eta*derW[l]
            #net['B'][l]=net['B'][l]-eta*derB[l]
        
        net=rpropTrainingPhase(net, derW, derB, deltaW, deltaB, oldDerW, oldDerB)
        
        Ynet=forward_prop(net,XTrain)
>>>>>>> parent of d68d946 (Revert "Pushed Neural Network")
        err=errFun(Ynet,YTrain)
        errTot.append(err)
        
        #Fase di validation
        Ynet_val=forward_prop(net,XVal)
        err_val=errFun(Ynet_val,YVal)
        errValTot.append(err_val)
            
        #Cerca l'errore minimo e la rete migliore
        if err_val < minErrVal:
            minErrVal=err_val
            bestNet=duplicateNetwork(net)
                
        print('Epoch:',epoch+1,
                'Train Err:',err,
                'Train Accuracy:',computeAccuracy(Ynet,YTrain),
                'Val Err:',err_val,
                'Val Accuracy:',computeAccuracy(Ynet_val,YVal),end=''
            )
        print('\r', end='') 
            
    if len(XVal) > 0:
        copyParamInNetwork(net,bestNet)
    return errTot,errValTot
 
    
def computeAccuracy(y_net,target):
    N=target.shape[1]
    z_net=myerr.softMax(y_net)
    return ((z_net.argmax(0)==target.argmax(0)).sum())/N
                  
def netAccuracy(net,X,target):
    y_net=forward_prop(net,X)
    return computeAccuracy(y_net,target)
