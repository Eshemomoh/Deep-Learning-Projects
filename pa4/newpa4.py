# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:16:06 2021

@author: Lucky Yerimah
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import MaxPool3D,LSTM,Dense,Conv2D,Reshape
from tqdm import trange
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def load_data(k):
    if k == 1:
        train_data = np.load('xtrain1.npy').astype(np.float32)
        train_label = np.load('ytrain1.npy').astype(np.float32)
    elif k == 2:
        train_data = np.load('xtrain2.npy').astype(np.float32)
        train_label = np.load('ytrain2.npy').astype(np.float32)
    elif k ==3:
        train_data = np.load('xtrain3.npy').astype(np.float32)
        train_label = np.load('ytrain3.npy').astype(np.float32)
    elif k ==4:
        train_data = np.load('xtrain4.npy').astype(np.float32)
        train_label = np.load('ytrain4.npy').astype(np.float32)
    elif k ==5:
        train_data = np.load('xtrain5.npy').astype(np.float32)
        train_label = np.load('ytrain5.npy').astype(np.float32)
    elif k==6:
        train_data = np.load('xtrain6.npy').astype(np.float32)
        train_label = np.load('ytrain6.npy').astype(np.float32)
    elif k==7:
        train_data = np.load('xtrain7.npy').astype(np.float32)
        train_label = np.load('ytrain7.npy').astype(np.float32)
    elif k ==8:
        train_data = np.load('xtrain8.npy').astype(np.float32)
        train_label = np.load('ytrain8.npy').astype(np.float32)
    elif k ==9:
        train_data = np.load('xtrain9.npy').astype(np.float32)
        train_label = np.load('ytrain9.npy').astype(np.float32)
    else:
        train_data = np.load('xtrain10.npy').astype(np.float32)
        train_label = np.load('ytrain10.npy').astype(np.float32)
        
    
    train_data = train_data/255.0
        
    
    return train_data,train_label
        
        
def load_test_data():    
    test_data = np.load('videoframes_clips_valid.npy').astype(np.float32)
    test_label = np.load('joint_3d_clips_valid.npy').astype(np.float32)
    
    
        
    for i in range(len(test_data)):
        test_data[i] = test_data[i]/255.0
    
    
    return test_data,test_label



def genBatch(X,Y,batch_size,shuffle):
    
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        # index = indices[:100] #memory unable to process all data at once
        xdata = Dataset.from_tensor_slices(X[indices]).batch(batch_size)
        ydata = Dataset.from_tensor_slices(Y[indices]).batch(batch_size)
    
    else:
        xdata = Dataset.from_tensor_slices(X)
        ydata = Dataset.from_tensor_slices(Y)
        
     
    return Dataset.zip((xdata,ydata))


def Model(input_size):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32,5,activation='relu',input_shape=input_size))
    model.add(MaxPool3D(pool_size=(1,2,2),strides=(1,2,2)))
    model.add(Reshape((8,-1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dense(51,activation="tanh"))
    model.add(Reshape((8,17,3)))
    
    return model
    
    
#%%

input_size = (8,224,224,3)

model = Model(input_size)
#Hyperparameters
epoch = 30
batch_size = 5
rate = 0.0001
loss_fun = tf.keras.losses.mean_squared_error
optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
Loss = []
Total_loss = []
Val_loss = []
Training_MPJPE = np.zeros((1,17))
#%%

xtest,ytest = load_test_data()
Test_data = genBatch(xtest,ytest,batch_size,shuffle=True)
del xtest,ytest

# obtain first validation losses
Test_loss = []
for Xtest,Ytest in Test_data:
    test_pred = model(Xtest)
    test_loss = loss_fun(Ytest,test_pred)
    Test_loss.append(tf.reduce_mean(test_loss).numpy())
    
    Test_MPJPE = tf.linalg.norm((tf.subtract(Ytest,test_pred)),axis=3)
    Test_MPJPE = tf.reduce_mean(Test_MPJPE,axis=1).numpy()

Val_MPJPE = np.expand_dims(np.mean(Test_MPJPE,axis=0)*1000,0)
Val_loss.append(np.mean(np.array(Test_loss)))

#%%
# train model


for i in trange(epoch):
    
    for k in range(10):
        # generate data
        xtrain,ytrain = load_data(k)
        
        
        Train_data = genBatch(xtrain,ytrain,batch_size,shuffle=True)
        for X,Y in Train_data:
            with tf.GradientTape() as tape:
                ypred = model(X,training=True)
                loss = loss_fun(Y,ypred)
                Train_mpjpe = tf.linalg.norm((tf.subtract(Y,ypred)),axis=3)
                Train_mpjpe = tf.reduce_mean(Train_mpjpe,axis=1).numpy()
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            Loss.append(tf.reduce_mean(loss).numpy())
            Training_MPJPE = np.concatenate((Training_MPJPE,np.expand_dims(np.mean(Train_mpjpe,axis=0),axis=0)),axis=0)
    
        del xtrain,ytrain
    Total_loss.append(np.array(Loss))
    Test_loss = []   
    for Xtest,Ytest in Test_data:
        test_pred = model(Xtest)
        test_loss = loss_fun(Ytest,test_pred)
        
        Test_loss.append(tf.reduce_mean(test_loss).numpy())
        
        Test_MPJPE = tf.linalg.norm((tf.subtract(Ytest,test_pred)),axis=3)
        Test_MPJPE = tf.reduce_mean(Test_MPJPE,axis=1).numpy()
        
        
    Val_MPJPE = np.concatenate((Val_MPJPE,np.expand_dims(np.mean(Test_MPJPE,axis=0)*1000,0)),axis=0)
    Val_loss.append(np.mean(np.array(Test_loss)))
    
# model.save("pa4moodel")        
#%%
# obtain the MPJPE
# load model
model = tf.keras.models.load_model("pa4moodel")
MPJPE = np.zeros((1,17))
for Xtest,Ytest in Test_data:
    test_pred = model(Xtest).numpy()
    #Test_prediction = np.concatenate((Test_prediction,test_pred),axis=0)
    Ytest = np.stack(Ytest)
    mpjpe = np.zeros((len(test_pred),17))
    for i in range(len(test_pred)):
        for j in range(17):
            for k in range(8):
                mpjpe[i,j] = mpjpe[i,j] + \
                np.linalg.norm(Ytest[i,k,j,:]-test_pred[i,k,j,:])/8
                
    MPJPE = np.concatenate((MPJPE,mpjpe),axis=0)
    
MPJPE = np.mean(MPJPE,axis=0)*1000 # converting to mm
print(MPJPE)
#%%  

# plots
# plt.plot(Total_loss1,label="Training loss")
plt.plot(Val_loss,label="Validation loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()  

plt.figure()
for i in range(17):
    plt.plot(Val_MPJPE[:,i],linewidth=1.0,label="Joint "+ str(i+1))
plt.title("Plots of Validation MPJPE")
plt.xlabel("Epoch")
plt.ylabel("MPJPE (mm)")
plt.show()
  
#%%
# TO EVALUATE THE CODE!!!

# to load model, uncomment the line below

#model = tf.keras.models.load_model("pa4moodel")

#to obtain model prediction using xtest_data, uncomment the line below
# prediction = model(xtest_data)

    
    
    
    
    
    
    
    
    
    