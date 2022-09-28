# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:35:39 2021

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
    
    data_filename = "train" + str(k+1) + ".npy"
    label_filename = "labels" + str(k+1) + ".npy"
    
    data = np.load(data_filename)
    labels = np.load(label_filename)
    # if k == 1:
    #     data = np.load("train1.npy")
    #     labels = np.load("label1.npy")
    # elif k == 2:
    #     data = np.load("train2.npy")
    #     labels = np.load("label2.npy")
    # elif k == 3:
    #     data = np.load("train3.npy")
    #     labels = np.load("label3.npy")
    # else:
    #     data = np.load("train4.npy")
    #     labels = np.load("label4.npy")
        
    
    trainlabel = tf.one_hot(labels,11).numpy()
    
    return data,trainlabel
    
    # batch generator
def genBatch(X,Y,batch_size,shuffle):
    
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        xdata = Dataset.from_tensor_slices(X[indices]).batch(batch_size)
        ydata = Dataset.from_tensor_slices(Y[indices]).batch(batch_size)
    
    else:
        xdata = Dataset.from_tensor_slices(X).batch(batch_size)
        ydata = Dataset.from_tensor_slices(Y).batch(batch_size)
        
     
    return Dataset.zip((xdata,ydata))

# model
def Model(input_size):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32,5,activation='relu',input_shape=input_size))
    model.add(MaxPool3D(pool_size=(1,2,2),strides=(1,2,2)))
    model.add(Conv2D(64,3,activation='relu',input_shape=input_size))
    model.add(MaxPool3D(pool_size=(1,2,2),strides=(1,2,2)))
    model.add(Reshape((30,-1)))
    model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(25,return_sequences=False))
    model.add(Dense(11,activation="softmax"))
    
    return model

def test_model(Y,yhat):
        
    #classification accuracy
    equals = tf.cast(tf.equal(tf.argmax(Y,1),tf.argmax(yhat,1)),tf.float32)
    Accuracy = tf.reduce_mean(equals)*100
    
    
    return Accuracy.numpy()


def call_train_classes(k):
    
    data_filename = "c" + str(k+1) + "data.npy"
    label_filename = "c" + str(k+1) + "label.npy"
    data = np.load(data_filename)
    label = np.load(label_filename)
    
    label = tf.one_hot(label,11).numpy()
    
    data = Dataset.from_tensor_slices(data).batch(20)
    label = Dataset.from_tensor_slices(label).batch(20)
    return Dataset.zip((data,label))

                       
def call_val_classes(k):
    
    data_filename  = "vc" + str(k+1) + "data.npy"
    label_filename = "vc" + str(k+1) + "label.npy"
    data = np.load( data_filename )
    label = np.load(label_filename)
    
    label = tf.one_hot(label,11).numpy()
    
    data = Dataset.from_tensor_slices(data).batch(20)
    label = Dataset.from_tensor_slices(label).batch(20)
    
    return Dataset.zip((data,label))
    
        
#%%
# load validation data
xval = np.load("valdata.npy")
yval = np.load("vallabel.npy")
vallabel = tf.one_hot(yval,11).numpy()
Val_data = genBatch(xval,vallabel,20,shuffle=False)
del xval,vallabel

#%%
# input size dimension
input_size = (30,64,64,3)
# call model
model = Model(input_size)

#Hyperparameters
epoch = 30
batch_size = 10
rate = 0.001
loss_fun = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(learning_rate=rate)

Total_loss = []
Val_loss = []

Otrain_acc = [] # overall training accuracy
Classtrain_acc = np.zeros((epoch,11)) # class wise training accuracy

Oval_acc = [] # overall validation accuracy
Classval_acc = np.zeros((epoch,11)) #class wise validation accuracy
#%%

for i in trange(epoch):
    Loss = []
    for k in range(4):
        
        # load data
        xtrain,ytrain = load_data(k)
        
        # generate batches
        Train_data = genBatch(xtrain,ytrain,batch_size,shuffle=True)
        train_acc = []
        for X,Y in Train_data:
            with tf.GradientTape() as tape:
                ypred = model(X,training=True)
                loss = loss_fun(Y,ypred)
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            Loss.append(tf.reduce_mean(loss).numpy())
            
            train_acc.append(test_model(Y,ypred))
    
            
        del xtrain,ytrain,Train_data,X,Y,ypred
    Otrain_acc.append(np.mean(np.array(train_acc)))
    
    #training class wise accuracy
    for k in range(11):
        class_acc = []
        train_class = call_train_classes(k)
        for x,y in train_class:
            yhat = model(x)
            class_acc.append(test_model(y,yhat))
        Classtrain_acc[i,k] = np.mean(np.array(class_acc))
        del train_class   
        
    #compute validation losses
    Total_loss.append(np.mean(np.array(Loss)))
    Test_loss = []
    val_acc = []
    for Xtest,Ytest in Val_data:
        test_pred = model(Xtest)
        test_loss = loss_fun(Ytest,test_pred)
        
        Test_loss.append(tf.reduce_mean(test_loss).numpy())
        val_acc.append(test_model(Ytest,test_pred))
        
    Val_loss.append(np.mean(np.array(Test_loss)))
    Oval_acc.append(np.mean(np.array(val_acc)))

    #validation class wise accuracy
    for k in range(11):
        class_acc = []
        val_class = call_val_classes(k)
        for x,y in val_class:
            yhat = model(x)
            class_acc.append(test_model(y,yhat))
        Classval_acc[i,k] = np.mean(np.array(class_acc))
        del val_class
#%%

# plots
# losses
plt.plot(Val_loss,label="Val_loss")
plt.plot(Total_loss,label="Train_loss")
plt.title("Losses")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# training accuracies
plt.figure()
plt.plot(Otrain_acc,label="Overall")
for i in range(11):
    plt.plot(Classtrain_acc[:,i],linewidth=1.0,label="Class "+ str(i+1))
plt.title("Training Accuracies")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# validation accuracies
plt.figure()
plt.plot(Oval_acc,label="Overall")
for i in range(11):
    plt.plot(Classval_acc[:,i],linewidth=1.0,label="Class "+ str(i+1))
plt.title("Validation Accuracies")
plt.xlabel("Epoch")
plt.legend()
plt.show()

model.save("projectmodel")









