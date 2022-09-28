## -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:37:37 2021

@author: Lucky Yerimah
Deep Learning Programming Assignment 1

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf
import pickle
from tqdm import tqdm


def get_train_data(file_dir):
    # image features
    #file_dir = "train_data/"
    path = os.listdir(file_dir['data'])
    dir_name = file_dir['data']
    
    image_data = []
    for file in path:
        filename = dir_name+file
        image = mpimg.imread(filename)
        image = np.reshape(image,28*28)/255.0
        image_data.append(image)
    
    image_data = np.array(image_data)
    
    # include ones for the bias term into X
    N,_ = np.shape(image_data)
    image_data = np.insert(image_data,[0],np.ones([N,1]),axis=1)
    
    
    # image labels
    labels = np.loadtxt(file_dir['label'])
    labels = np.reshape(labels,[labels.shape[0],1])
    
    return image_data, labels


def get_test_data(k):
    # image features
    file_dir = "test_data/"
    path = os.listdir(file_dir)

    image_data = []
    for file in path:
        filename = file_dir+file
        image = mpimg.imread(filename)
        image = np.reshape(image,28*28)/255.0
        image_data.append(image)
    
    image_data = np.array(image_data)
    
    # include ones for the bias term into X
    N,_ = np.shape(image_data)
    image_data = np.insert(image_data,[0],np.ones([N,1]),axis=1)
    
    
    # image labels
    labels = np.loadtxt("labels/test_label.txt")
    labels = np.reshape(labels,[labels.shape[0],1])
    
    # separate and return data for the digit specified
    x = []
    y = []
    
    
    for i in range(labels.shape[0]):
        if labels[i,:] == k:
            x.append(image_data[i,:])
            y.append(labels[i,:])
    
    return np.array(x),np.array(y)


# generate tm vectors of the classes
def get_tm_label(labels,k):
    Tm = []
    for i in np.nditer(labels):
        tm = np.zeros(k)
        tm[int(i-1)] = 1.0
        Tm.append(tm)
        
    Tm = np.array(Tm,dtype='float32')
    return Tm

     
def classifier(k,M,Y,X,W,eta):
    
    # initialize tensorflow parameters
    
    W = tf.Variable(W,dtype=("float32"))
    X = tf.constant(X,dtype="float32")
    Y = tf.constant(Y,dtype="float32")
    
    # softmax function
    exponential_fun = tf.exp(tf.matmul(X,W))
    summation_fun = tf.tile(tf.reshape(tf.reduce_sum(exponential_fun,1),[M,1]),[1,k])
    softmax_fun = tf.divide(exponential_fun,summation_fun)


    # gradient of loss function
    gradient = (-1/M)*tf.matmul(X,tf.subtract(Y,softmax_fun),transpose_a=True)
    
    # weight update
    New_weight = tf.subtract(W,tf.multiply(eta,gradient))
    W.assign(New_weight)
    
    return W.numpy()

"""
def test_model(W,X,Y,truetm,k):
    M,N = X.shape
    W = tf.Variable(W,dtype=("float32"))
    X = tf.constant(X,dtype="float32")
    
    exponential_fun = tf.exp(tf.matmul(X,W))
    summation_fun = tf.tile(tf.reshape(tf.reduce_sum(exponential_fun,1),[M,1]),[1,k])
    softmax_fun = tf.divide(exponential_fun,summation_fun)
    loss = -tf.reduce_sum(tf.multiply(truetm,tf.math.log(softmax_fun)))
    # cross entropy loss function
    #Loss_fun = - tf.reduce_sum(tf.multiply(truetm,tf.math.log(softmax_fun)))
    
    accuracy = 0
    for i in range(M):
        if np.int(Y[i,:]) == (tf.argmax(softmax_fun[i,:])+1):
            accuracy += 1
            
    Accuracy = (accuracy/M)*100
    return Accuracy,-1*loss


"""

# decided to use numpy for test model. Tensorflow is 5 times slower
def test_model(W,X,Y,truetm,k):
    M,N = X.shape
    
    exponential_fun = np.exp(np.matmul(X,W))
    summation_fun = np.tile(np.reshape(np.sum(exponential_fun,1),[M,1]),[1,k])
    softmax_fun = np.divide(exponential_fun,summation_fun)
    loss = -np.sum(np.multiply(truetm,np.log(softmax_fun)))
    # cross entropy loss function
    #Loss_fun = - tf.reduce_sum(tf.multiply(truetm,tf.math.log(softmax_fun)))
    
    accuracy = 0
    for i in range(M):
        if np.int(Y[i,:]) == (np.argmax(softmax_fun[i,:])+1):
            accuracy += 1
            
    Accuracy = (accuracy/M)*100
    return Accuracy,-1*loss


def classification_error(W,X,Y,k):
    M,N = X.shape
    
    exponential_fun = np.exp(np.matmul(X,W))
    summation_fun = np.tile(np.reshape(np.sum(exponential_fun,1),[M,1]),[1,k])
    softmax_fun = np.divide(exponential_fun,summation_fun)
    
    # cross entropy loss function
    #Loss_fun = - tf.reduce_sum(tf.multiply(truetm,tf.math.log(softmax_fun)))
    
    error = 0
    for i in range(M):
        if np.int(Y[i,:]) != (np.argmax(softmax_fun[i,:])+1):
            error += 1
            
    Error = error/M
    return Error

    
def train_model(Xtrain,Ytrain,tmtrain,Xtest,Ytest,tmtest,iteration,W,k,eta):
    Train_loss = []
    Test_loss = []
    
    Train_accuracy = []
    Test_accuracy = []
    
    M,_ = Xtrain.shape
    
    for i in tqdm(range(iteration)):
        W = classifier(k,M,tmtrain,Xtrain,W,eta)
        
        # calculate the loss and accuracy of train and testing data after every 10 iteration
        train_accuracy,train_loss = test_model(W,Xtrain,Ytrain,tmtrain,k)
        test_accuracy,test_loss = test_model(W,Xtest,Ytest,tmtest,k)
        
        Train_accuracy.append(train_accuracy)
        Test_accuracy.append(test_accuracy)
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
        
        
    weights = W    
    return weights, np.array(Train_accuracy),np.array(Train_loss),\
        np.array(Test_accuracy),np.array(Test_loss)
    
    

# get training data    
train_file_dir = {"data": "train_data/",
                  "label":"labels/train_label.txt"
                  }
Xtrain,Ytrain = get_train_data(train_file_dir)
tmtrain = get_tm_label(Ytrain,5)

# get testing data
test_file_dir = {"data":"test_data/",
                  "label":"labels/test_label.txt"
                  }
Xtest,Ytest = get_train_data(test_file_dir)
tmtest = get_tm_label(Ytest,5)

k = 5 # of classes
M,N = Xtrain.shape
eta = 0.75
W = 0.01*np.random.rand(N,k)

    
#%%
# Training
iteration = 2000

Weight,train_accuracy,train_loss,test_accuracy,test_loss = \
    train_model(Xtrain,Ytrain,tmtrain,Xtest,Ytest,tmtest,iteration,W,k,eta)

# save weights
filehandler = open("multiclass_parameters.txt","wb")
pickle.dump(Weight,filehandler)
filehandler.close()

#%%
 
# separate the different digits and calculate classification errors

#digit = 1
x1,y1 = get_test_data(1)
Error1 = classification_error(Weight,x1,y1,k)

#digit = 2
x2,y2 = get_test_data(2)
Error2 = classification_error(Weight,x2,y2,k)

#digit = 3
x3,y3 = get_test_data(3)
Error3 = classification_error(Weight,x3,y3,k)

#digit = 4
x4,y4 = get_test_data(4)
Error4 = classification_error(Weight,x4,y4,k)

#digit = 5
x5,y5 = get_test_data(5)
Error5 = classification_error(Weight,x5,y5,k)

# get overall error 
Training_error = classification_error(Weight,Xtrain,Ytrain,k)
Testing_error = classification_error(Weight,Xtest,Ytest,k)    

#%%
# Plot weight matrices and errors

# digit 1
W = Weight[1:,0]
W = np.reshape(W,[28,28])
plt.imshow(W)
plt.colorbar()
plt.show()


# digit 2
W = Weight[1:,1]
W = np.reshape(W,[28,28])
plt.imshow(W)
plt.colorbar()
plt.show()

# digit 3
W = Weight[1:,2]
W = np.reshape(W,[28,28])
plt.imshow(W)
plt.colorbar()
plt.show()

# digit 4
W = Weight[1:,3]
W = np.reshape(W,[28,28])
plt.imshow(W)
plt.colorbar()
plt.show()

# digit 5
W = Weight[1:,4]
W = np.reshape(W,[28,28])
plt.imshow(W)
plt.colorbar()
plt.show()

# Training and testing  Error

plt.semilogy(-train_loss,scalex=True,label="Training Error")
plt.semilogy(-test_loss,scalex=True,label="Testing Error")
plt.legend()
plt.show()

# Training and testing  Accuracy
plt.plot(train_accuracy,scalex=True,label="Training Accuracy")
plt.plot(test_accuracy,scalex=True,label="Testing Accuracy")
plt.legend()
plt.show()