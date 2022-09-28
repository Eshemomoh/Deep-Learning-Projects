# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 00:29:58 2021

@author: Lucky Yerimah
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf
import pickle
from tqdm import trange
np.random.seed(10)



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
        tm = np.zeros(10)
        tm[int(i-1)] = 1.0
        Tm.append(tm)
        
    Tm = np.array(Tm,dtype='float32')
    return Tm


class Model:
    def __init__(self,InputDim,nodes,K):
        self.activation = tf.nn.relu
        self.output_activation = tf.nn.softmax
        self.w1 = tf.Variable(0.01*np.random.rand(InputDim,nodes))
        self.w10 = tf.Variable(np.zeros([1,nodes])) 

        self.w2 = tf.Variable(0.01*np.random.rand(nodes,nodes))
        self.w20 = tf.Variable(np.zeros([1,nodes])) 
        
        self.w3 = tf.Variable(0.01*np.random.rand(nodes,K))
        self.w30 = tf.Variable(np.zeros([1,K])) 
        
        
    def forwardprop(self,x,y):
        
        
        z1 = tf.add(tf.matmul(x,self.w1),self.w10)
        H1 = self.activation(z1)
        
        z2 = tf.add(tf.matmul(H1,self.w2),self.w20)
        H2 = self.activation(z2)
        
        z3 = tf.add(tf.matmul(H2,self.w3),self.w30)
        yhat = self.output_activation(z3)
        dy = -tf.divide(y,yhat)
        
        return dy.numpy(),yhat.numpy(),H2.numpy(),H1.numpy(),z1.numpy(),z2.numpy()
        
    
    def forwardproptest(self,x,y):
        
        z1 = tf.add(tf.matmul(x,self.w1),self.w10)
        H1 = self.activation(z1)
        
        z2 = tf.add(tf.matmul(H1,self.w2),self.w20)
        H2 = self.activation(z2)
        
        z3 = tf.add(tf.matmul(H2,self.w3),self.w30)
        
        return self.output_activation(z3).numpy()
    
    
    def backprop(self,X,Y,K,eta):
        nodes = self.w1.shape[1]
        M,N = X.shape
        dy,yhat,H2,H1,z1,z2 = self.forwardprop(X,Y)
        
        
        #output layer
        dzdz = np.zeros([M,K,K])
        for k in range(K):
            for i in range(K):
                if k == i:
                    dzdz[:,k,i] = yhat[:,i]*(1-yhat[:,i])
                else:
                    dzdz[:,k,i] = -yhat[:,k]*yhat[:,i]
        
        dy = np.expand_dims(dy,1)
        dw30 = np.matmul(dy,dzdz)
        dw3 = np.matmul(np.expand_dims(H2,2),dw30)
        w3 = self.w3.numpy()
        dH2 = np.matmul(dw30,np.transpose(np.reshape(np.tile(w3,[M,1]),[M,nodes,K]),axes=[0,2,1]))
        
        
        #second hidden layer
        dzdz = np.zeros([M,nodes,nodes])
        for m in range(M):
            for k in range(nodes):
                if z2[m,k] > 0:
                    dzdz[m,k,k] = 1.0 #relu
        
        dw20 = np.matmul(dH2,dzdz)
        dw2 = np.matmul(np.expand_dims(H1,2),dw20)
        w2 = self.w2.numpy()
        dH1 = np.matmul(dw20,np.transpose(np.reshape(np.tile(w2,[M,1]),[M,nodes,nodes]),[0,2,1]))
        
        
        #first hidden layer
        dzdz = np.zeros([M,nodes,nodes])
        for m in range(M):
            for k in range(nodes):
                if z1[m,k] > 0:
                    dzdz[m,k,k] = 1.0 #relu
        
        dw10 = np.matmul(dH1,dzdz)
        dw1 = np.matmul(np.expand_dims(X,2),dw10)
        
        
        #compute average gradients
        dW1 = (1/M)*np.sum(dw1,0)
        dW10 = (1/M)*np.sum(dw10,0)
        
        dW2 = (1/M)*np.sum(dw2,0)
        dW20 = (1/M)*np.sum(dw20,0)
        
        dW3 = (1/M)*np.sum(dw3,0)
        dW30 = (1/M)*np.sum(dw30,0)
    
        #weight updates
        self.w1.assign_sub(eta*dW1)
        self.w10.assign_sub(eta*dW10)
        
        self.w2.assign_sub(eta*dW2)
        self.w20.assign_sub(eta*dW20)
        
        self.w3.assign_sub(eta*dW3)
        self.w30.assign_sub(eta*dW30)
        
        
    def test_model(self,Y,yhat):
        
        #classification error
        equals = tf.cast(tf.equal(tf.argmax(Y,1),tf.argmax(yhat,1)),tf.float32)
        Class_error = tf.subtract(1.0,tf.reduce_mean(equals))
        Accuracy = tf.reduce_mean(equals)*100
        
        #loss
        loss = - np.sum(np.sum((Y*np.log(yhat)),1))/Y.shape[0]
        
        return Class_error.numpy(),Accuracy.numpy(),loss
        
    def train_model(self,X,Y,Xtest,Ytest,epoch,batch_size,eta):
        
        indices = np.arange(X.shape[0])
        Class_error,Accuracy,Loss = [],[],[]
        Test_error,Test_accuracy = [],[]
        
        counter = 0
        for m in trange(epoch):
            
            np.random.shuffle(indices)
            counter += 1
            index = indices[:batch_size] 
            xdata = X[index,:]
            ydata = Y[index,:]
            
                
            self.backprop(xdata,ydata,Y.shape[1],eta)
            
            
            
            yhat = self.forwardproptest(xdata,ydata)
            class_error,accuracy,loss = self.test_model(ydata,yhat)
            testhat = self.forwardproptest(Xtest,Ytest)
            test_error,test_accuracy,_ = self.test_model(Ytest,testhat)
            
            Class_error.append(class_error)
            Accuracy.append(accuracy)
            Loss.append(loss)
        
            Test_error.append(test_error)
            Test_accuracy.append(test_accuracy)
            
            
        Weights = [self.w1.numpy(),self.w10.numpy(),self.w2.numpy(),self.w20.numpy(),self.w3.numpy(),self.w30.numpy()]
            
            
        return Class_error,Accuracy,Loss,Test_error,Test_accuracy,eta,Weights
                
            
     
        
#%%
# get training data    
train_file_dir = {"data": "train_data/",
                  "label":"labels/train_label.txt"
                  }
Xtrain,Ytrain = get_train_data(train_file_dir)
tmtrain = get_tm_label(Ytrain,10)

# get testing data
test_file_dir = {"data":"test_data/",
                  "label":"labels/test_label.txt"
                  }
Xtest,Ytest = get_train_data(test_file_dir)
tmtest = get_tm_label(Ytest,10)                   

#%%

batch_size = 50
eta = 0.1

#initialize model class
model = Model(784,100,10)

#training
epoch = 10000
Class_error,Accuracy,Loss,Test_error,Test_accuracy,eta,Weights = model.train_model(Xtrain,tmtrain,Xtest,tmtest,epoch,batch_size,eta)                      

#%%
# save weights
Theta = Weights
filehandler = open("nn_parameters.txt","wb") 
pickle.dump(Theta, filehandler, protocol=2) 
filehandler.close() 

#individual digit errors
for i in range(10):
    x1,y1 = get_test_data(i)
    tm = get_tm_label(y1,i)
    yhat = model.forwardproptest(x1,tm)
    test_error,test_accuracy,_ = model.test_model(tm,yhat)
    print("digit, Error, Accuracy")
    print([i, test_error, test_accuracy,])
    
#over digit error
yhat = model.forwardproptest(Xtest,tmtest)
test_error,test_accuracy,_ = model.test_model(tmtest,yhat)
print("Overall Error, Overall Accuracy")
print([test_error, test_accuracy,])

#loss
plt.plot(Loss,scalex=True,label="Loss")
plt.legend()
plt.show()

# testing error
plt.plot(Test_error,scalex=True,label="Testing error")
plt.legend()
plt.show()

# Training error
plt.plot(Class_error,scalex=True,label="Training error")
plt.legend()
plt.show()