# -*- coding: utf-8 -*-
"""
Created on Fri May  7 02:22:23 2021

@author: Lucky Yerimah
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from tensorflow.data import Dataset
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# compute accuracy
def test_model(Y,yhat):
        
    #classification accuracy
    equals = tf.cast(tf.equal(Y,tf.argmax(yhat,1)),tf.float32)
    Accuracy = tf.reduce_mean(equals)*100
    
    
    return Accuracy.numpy()


# load validation data
valdata = np.load("valdata.npy")
vallabel = np.load("vallabel.npy")
vallabel = np.array(vallabel,dtype=np.int64)

# load model
model = tf.keras.models.load_model("projectmodel")



"""
Two types of using this model is provided. My computer is unable to forward propagate the entire data 
at once hence i break the data into batches. If you can forward propagate the entire data, then use the second approach.
Uncomment anyone to use.
"""

#%%
"""FIRST APPROACH: When Computer is Unable to Process Entire Dataset Through The Model """

# generate batches
batch_size = 20
xdata = Dataset.from_tensor_slices(valdata).batch(batch_size)
ydata = Dataset.from_tensor_slices(vallabel).batch(batch_size)

Data = Dataset.zip((xdata,ydata))
Accuracy = []
Yhat = []
for X,Y in Data:
    Ypred = model(X)
    accuracy = test_model(Y,Ypred)
    Accuracy.append(accuracy)
    Yhat.append(np.argmax(Ypred.numpy(),axis=1))
    
Final_Accuracy = np.mean(np.array(Accuracy))

#reshape Yhat into a vector
Prediction = np.reshape(np.array(Yhat[:-1]),-1)
# last batch is not upto 20 hence is added manually 
Prediction = np.concatenate((Prediction,np.array(Yhat[-1])),axis=0)

metric = confusion_matrix(vallabel,Prediction)

#%%
# """SECOND APPOACH: When Computer is Able to Process Entire Dataset Through The Model"""
# Ypred = model(valdata)
# Final_Accuracy = test_model(vallabel,Ypred)
# Ypred = np.argmax(Ypred.numpy(),axis = 1)
# metric = confusion_matrix(vallabel,Yhat)

#%%
#display confusion matrix and final accuracy
disp = ConfusionMatrixDisplay(confusion_matrix = metric)
disp.plot()
plt.show()
print("Final Accuracy = " + str(Final_Accuracy))