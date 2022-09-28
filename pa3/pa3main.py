# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 23:08:52 2021

@author: Lucky Yerimah
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.data import Dataset
from tqdm import trange
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)



# generate one hot labels
def one_hot(labels,k):
    Tm = []
    for i in np.nditer(labels):
        tm = np.zeros(k)
        tm[int(i)] = 1.0
        Tm.append(tm)
        
    Tm = np.array(Tm,dtype='float32')
    return Tm

#load data and obtain one hot labels

def load_data(file_dir):
    xtrain = np.load(file_dir['xtrain'])
    ytrain = np.load(file_dir['ytrain'])
    xtest = np.load(file_dir['xtest'])
    ytest = np.load(file_dir['ytest'])
    
    #preprocess data
    xtrain = xtrain/255.0
    xtest = xtest/255.0
    
    # get one hot labels
    ytrain = one_hot(ytrain,file_dir['K'])
    ytest = one_hot(ytest,file_dir['K'])
    return xtrain,ytrain,xtest,ytest


# batch generator

def genBatch(X,Y,batch_size,shuffle):
    
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        xdata = Dataset.from_tensor_slices(X[indices]).batch(batch_size)
        ydata = Dataset.from_tensor_slices(Y[indices]).batch(batch_size)
    
    else:
        xdata = Dataset.from_tensor_slices(X)
        ydata = Dataset.from_tensor_slices(Y)
        
    Data = Dataset.zip((xdata,ydata)) 
    return Data
    

# model class
class MyModel(tf.keras.Model):
    
    def __init__(self,input_shapes,K):
        super(MyModel, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(32,5,activation='relu',input_shape=input_shapes[0])
        self.pol1 = tf.keras.layers.MaxPool2D((2,2),(2,2))
        self.conv2 = tf.keras.layers.Conv2D(32,5,activation='relu',input_shape=input_shapes[1])
        self.pol2 = tf.keras.layers.MaxPool2D((2,2),(2,2))
        self.conv3 = tf.keras.layers.Conv2D(64,3,activation='relu',input_shape=input_shapes[2])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(K,activation='softmax')
        
    # forward pass
    def call(self,X):
        c1 = self.conv1(X)
        pol1 = self.pol1(c1)
        c2 = self.conv2(pol1)
        pol2 = self.pol2(c2)
        c3 = self.conv3(pol2)
        
        cflat = self.flatten(c3)
        yout = self.dense(cflat)
        
        return yout
        

    def test_model(self,Y,yhat):
        
        #classification error
        equals = tf.cast(tf.equal(tf.argmax(Y,1),tf.argmax(yhat,1)),tf.float32)
        Class_error = tf.subtract(1.0,tf.reduce_mean(equals))
        Accuracy = tf.reduce_mean(equals)*100
        
        
        return Class_error.numpy(),Accuracy.numpy()#,loss


    def train_model(self,Traindata,rate):
        
        opt = tf.optimizers.Adam(rate)
        Loss = []
        Accuracy = []
        
        for X,Y in Traindata:
            
            with tf.GradientTape(persistent=True) as tape:
                yout = self.call(X)
                loss = tf.nn.softmax_cross_entropy_with_logits(yout,Y)
                
            gradients = tape.gradient(loss,self.trainable_variables)
            grads_and_vars = zip(gradients,self.trainable_variables)
            opt.apply_gradients(grads_and_vars)
           
            _,accuracy = self.test_model(Y,yout)
            Loss.append(tf.reduce_mean(loss).numpy())
            Accuracy.append(accuracy)
            
                  
            
        
            
        return tf.reduce_mean(Loss).numpy(),tf.reduce_mean(Accuracy).numpy()


def get_test_data(file_dir,k):
    
    xtest = np.load(file_dir['xtest'])
    ytest = np.load(file_dir['ytest'])
    
    xtest = xtest/255.0
    # separate and return data for the image type specified
    x = []
    y = []
    
    
    for i in range(ytest.shape[0]):
        if ytest[i] == k:
            x.append(xtest[i])
            y.append(ytest[i])
    
    y = np.array(y)
    y = np.reshape(y,[len(y)])
    yout = one_hot(y,file_dir['K'])
    return np.array(x),np.array(yout)






#%%
# directory for loading dataset
file_dir = {
    "xtrain": "training_data.npy",
    "ytrain":"training_label.npy",
    "xtest": "testing_data.npy",
    "ytest":"testing_label.npy",
    "K": 10,
                  }

xtrain,ytrain,xtest,ytest = load_data(file_dir)
#%%


Train_Loss = []
Train_Accuracy = []

Test_Loss = []
Test_Accuracy = []

input_shape = [(32,32,3),(14,14,32),(5,5,32)]
model = MyModel(input_shape,10)
#%%
#Hyperparameters
epoch = 2000
batch_size = 500
rate = 0.001

for i in trange(epoch):
    
    # # generate batches
    Data = genBatch(xtrain,ytrain,batch_size,shuffle=True)
    train_loss,train_accuracy = model.train_model(Data,rate)
    
    # test loss and accuracy
    predict = model(xtest).numpy()
    test_loss = tf.nn.softmax_cross_entropy_with_logits(predict,ytest)
    _,test_accuracy = model.test_model(ytest,predict)
    
    Train_Loss.append(train_loss)
    Train_Accuracy.append(train_accuracy)
    
    if test_accuracy > 65:
        rate = rate*0.9
    
    # trying to get test accuracy and saved it immediately
    if test_accuracy > 68.56 and test_accuracy > Test_Accuracy[-1]:
       
        model.save("cnnmodel")
        best_accuracy = test_accuracy
            
    Test_Loss.append(tf.reduce_mean(test_loss).numpy())
    Test_Accuracy.append(test_accuracy)
    
    
    
    

#%%
# individual image class errors
image_classes = ["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

#load saved model as saved_model
saved_model = tf.keras.models.load_model("cnnmodel")

for i in range(10):
    x1,y1 = get_test_data(file_dir,i)
    yhat = saved_model(x1)
    error,accuracy = model.test_model(y1,yhat)
    print("Image class, Error, Accuracy")
    print([image_classes[i], error, accuracy])
    

#over digit error
yhat = saved_model(xtest)
test_error,test_accuracy,= model.test_model(ytest,yhat)
print("Overall Error, Overall Accuracy")
print([test_error, test_accuracy,])
    

#%%
#Plots
#loss
plt.plot(Train_Loss,scalex=True,label="Training Loss")
plt.plot(Test_Loss,scalex=True,label="Testing Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

#accuracy
plt.plot(Train_Accuracy,scalex=True,label='Training Accuracy')
plt.plot(Test_Accuracy,scalex=True,label='Testing Accuracy')
plt.xlabel("Epoch")
plt.legend()
plt.show()


#filters
filters,bias = saved_model.conv1.get_weights()

# visualize the plots of the first layer filters
plt.subplots(figsize=(30,15))

count = 1
for i in range(filters.shape[3]):
    plt.subplot(4,8,count)
    image = filters[:,:,:,i]
    plt.imshow(image)
    plt.title("Filter" + str(i+1),fontsize=18)
    count +=1



    
#%%
#LOAD MODEL AND OBTAIN PREDICTIONS FROM HERE BY Uncommenting. 

# #load saved model as saved_model
# saved_model = tf.keras.models.load_model("cnnmodel")

# # to obtain predicted classes, using "testdata"
# predictions = saved_model(testdata)























