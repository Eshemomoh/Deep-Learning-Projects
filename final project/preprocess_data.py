# -*- coding: utf-8 -*-
"""
Created on Sun May  9 02:45:13 2021

@author: Lucky Yerimah
"""

import numpy as np
import pickle



# data_file = open('youtube_action_train_data_part1.pkl', 'rb')
# train_data1, train_labels1 = pickle.load(data_file)
# data_file.close()

# data_file = open('youtube_action_train_data_part2.pkl', 'rb')
# train_data2, train_labels2 = pickle.load(data_file)
# data_file.close()

#%%
data = np.concatenate((train_data1,train_data2),axis=0)
label = np.concatenate((train_labels1,train_labels2),axis=0)

# clear memory 
del train_data1,train_data2,train_labels2,train_labels1 

"""
I observe that the data is arranged from labels 0 to 10, hence there's need to shuffle
"""

index = np.arange(label.shape[0])
np.random.shuffle(index)

# split files into validation and training sets

valdata = data[index[:1272]]/255.0
vallabel = label[index[:1272]]

train1 = data[index[1272:2772]]/255.0
labels1 = label[index[1272:2772]]

train2 = data[index[2772:4272]]/255.0
labels2 = label[index[2772:4272]]

train3 = data[index[4272:5772]]/255.0
labels3 = label[index[4272:5772]]

train4 = data[index[5772:]]/255.0
labels4 = label[index[5772:]]

#%%
# save files 

np.save("valdata",np.array(valdata,dtype=np.float32))
np.save("vallabel",vallabel)

np.save("train1",np.array(train1,dtype=np.float32))
np.save("labels1",labels1)

np.save("train2",np.array(train2,dtype=np.float32))
np.save("labels2",labels2)

np.save("train3",np.array(train3,dtype=np.float32))
np.save("labels3",labels3)

np.save("train4",np.array(train4,dtype=np.float32))
np.save("labels4",labels4)

#%%
# split data into different classes

# generate different classes of data

c1data = []
c1label = []
c2data = []
c2label = []
c3data = []
c3label = []
c4data = []
c4label = []
c5data = []
c5label = []
c6data = []
c6label = []
c7data = []
c7label = []
c8data = []
c8label = []
c9data = []
c9label = []
c10data = []
c10label = []
c11data = []
c11label = []

# for separating training classes datasets
# data1 = np.load("train1.npy")
# labels1 = np.load("labels1.npy")
# data2 = np.load("train2.npy")
# labels2 = np.load("labels2.npy")
# data3 = np.load("train3.npy")
# labels3 = np.load("labels3.npy")
# data4 = np.load("train4.npy")
# labels4 = np.load("labels4.npy")

# traindata = np.concatenate((data1,data2,data3,data4),axis=0)
# trainlabels = np.concatenate((labels1,labels2,labels3,labels4),axis=0)

# for separating validation classes
traindata = np.load("valdata.npy")
trainlabels = np.load("vallabel.npy") 


for i in range(trainlabels.shape[0]):
    if trainlabels[i] == 0:
        c1data.append(traindata[i])
        c1label.append(trainlabels[i])
    elif trainlabels[i] == 1:
        c2data.append(traindata[i])
        c2label.append(trainlabels[i])
    elif trainlabels[i] == 2:
        c3data.append(traindata[i])
        c3label.append(trainlabels[i])
    elif trainlabels[i] == 3:
        c4data.append(traindata[i])
        c4label.append(trainlabels[i])
    elif trainlabels[i] == 4:
        c5data.append(traindata[i])
        c5label.append(trainlabels[i])
    elif trainlabels[i] == 5:
        c6data.append(traindata[i])
        c6label.append(trainlabels[i])
    elif trainlabels[i] == 6:
        c7data.append(traindata[i])
        c7label.append(trainlabels[i])
    elif trainlabels[i] == 7:
        c8data.append(traindata[i])
        c8label.append(trainlabels[i])
    elif trainlabels[i] == 8:
        c9data.append(traindata[i])
        c9label.append(trainlabels[i])
    elif trainlabels[i] == 9:
        c10data.append(traindata[i])
        c10label.append(trainlabels[i])
    elif trainlabels[i] == 10:
        c11data.append(traindata[i])
        c11label.append(trainlabels[i]) 


# # save training classes
# np.save("c1data",np.array(c1data))
# np.save("c1label",np.array(c1label))

# np.save("c2data",np.array(c2data))
# np.save("c2label",np.array(c2label))

# np.save("c3data",np.array(c3data))
# np.save("c3label",np.array(c3label))

# np.save("c4data",np.array(c4data))
# np.save("c4label",np.array(c4label))

# np.save("c5data",np.array(c5data))
# np.save("c5label",np.array(c5label))

# np.save("c6data",np.array(c6data))
# np.save("c6label",np.array(c6label))

# np.save("c7data",np.array(c7data))
# np.save("c7label",np.array(c7label))

# np.save("c8data",np.array(c8data))
# np.save("c8label",np.array(c8label))

# np.save("c9data",np.array(c9data))
# np.save("c9label",np.array(c9label))

# np.save("c10data",np.array(c10data))
# np.save("c10label",np.array(c10label))

# np.save("c11data",np.array(c11data))
# np.save("c11label",np.array(c11label))


# save vilidation classes
np.save("vc1data",np.array(c1data))
np.save("vc1label",np.array(c1label))

np.save("vc2data",np.array(c2data))
np.save("vc2label",np.array(c2label))

np.save("vc3data",np.array(c3data))
np.save("vc3label",np.array(c3label))

np.save("vc4data",np.array(c4data))
np.save("vc4label",np.array(c4label))

np.save("vc5data",np.array(c5data))
np.save("vc5label",np.array(c5label))

np.save("vc6data",np.array(c6data))
np.save("vc6label",np.array(c6label))

np.save("vc7data",np.array(c7data))
np.save("vc7label",np.array(c7label))

np.save("vc8data",np.array(c8data))
np.save("vc8label",np.array(c8label))

np.save("vc9data",np.array(c9data))
np.save("vc9label",np.array(c9label))

np.save("vc10data",np.array(c10data))
np.save("vc10label",np.array(c10label))

np.save("vc11data",np.array(c11data))
np.save("vc11label",np.array(c11label))






















