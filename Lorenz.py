#!/usr/bin/env python
# coding: utf-8

# In[17]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


import math
import time 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.distributions import Gumbel,Bernoulli,Normal
from torchvision import datasets, transforms

#from torchviz import make_dot, make_dot_from_trace
#from tensorboardX import SummaryWriter

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
import copy

from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau


# # Model

# In[19]:


import DataProcessing
from DataProcessing import *

import DSSSMCode
from DSSSMCode import *


# ### Reload Model as needed

# In[20]:


import importlib
importlib.reload(DataProcessing)
from DataProcessing import *

import importlib
importlib.reload(DSSSMCode)
from DSSSMCode import *


# # Dataset Preprocessing
# 

# In[21]:


restore = True


# In[22]:


bidirection = False
dataname = 'Lorenz'


# In[23]:


import json
f = open("./data/lorenz.json")
data_st_all = json.load(f)
D = len(data_st_all['data'][0][0])
factors_true = torch.FloatTensor(data_st_all['factors'])
z_true = torch.FloatTensor(data_st_all['latents'])
z_true = z_true[:,2000:5000]
data_st = np.array(data_st_all['data'])
data_st = data_st[:,2000:5000]
data_st= data_st+data_st[0].std(axis=0)*0.001*np.random.randn(data_st.shape[1],10) #added noise
data_st = (data_st-data_st[0].mean(axis=0))/data_st[0].std(axis=0) #added
states = np.zeros(z_true.numpy().shape[0:2])
states[z_true.numpy()[:,:,0]>0] = 1
states = torch.LongTensor(states)


# In[24]:


data_st.shape


# In[25]:


RawDataOriginal = data_st.transpose(1,0,2)
moments = normalize_moments(RawDataOriginal)


# In[26]:


RawDataOriginal.shape


# In[27]:


freq = 1
test_len = 1000 * freq
timestep = 5
predict_dim = 10
look_back =  timestep


# In[28]:


RawData = RawDataOriginal - 0
RawData = RawData.reshape(-1,RawData.shape[2])
data = RawData
data.shape

# Split into train and text data, train data 0.6 vs validation 0.2 test data 0.2
length = len(data) - test_len
train_len = 1000+ timestep#int(length * 0.66)
valid_len = length - train_len#int(length * 0.33)

train_data = data[:train_len]
valid_data = data[(train_len):(train_len+valid_len)]
test_data  = data[(-test_len-timestep-1):-1]

print("train size (days):",train_data.shape[0],
      "valid size(days):",valid_data.shape[0],
      "test size(days):",test_data.shape[0])

# Normalize the dataset
moments = normalize_moments(train_data)
train_data = normalize_fit(train_data,moments)
valid_data = normalize_fit(valid_data,moments)
test_data = normalize_fit(test_data,moments)
print('std:',train_data.std())

# Create training and test dataset
trainX, trainY = create_dataset2(train_data,look_back)
validX, validY = create_dataset2(valid_data,look_back)
testX, testY = create_dataset2(test_data,look_back)
print("2D size(X):", trainX.shape,validX.shape,testX.shape)
print("2D size(Y):", trainY.shape,validY.shape,testY.shape)

trainX = np.transpose(trainX, (1, 0, 2))
validX = np.transpose(validX, (1, 0, 2))
testX = np.transpose(testX, (1, 0, 2))
print("3D size(X):",trainX.shape,validX.shape,testX.shape)

trainY = np.transpose(trainY, (1, 0, 2))
validY = np.transpose(validY, (1, 0, 2))
testY = np.transpose(testY, (1, 0, 2))
print("3D size(Y):",trainY.shape,validY.shape,testY.shape)

print("Numpy into Tensor, done!")
trainX = torch.from_numpy(trainX).float()
validX = torch.from_numpy(validX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
validY = torch.from_numpy(validY).float()
testY = torch.from_numpy(testY).float()


# In[ ]:





# # Training

# In[29]:


## Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("Device:",device)
trainX = trainX.to(device)
validX = validX.to(device)
testX = testX.to(device)
trainY = trainY.to(device)
validY = validY.to(device)
testY = testY.to(device)


# In[30]:


#writer = SummaryWriter()
## hyperparameters
x_dim = predict_dim # Dimension of x 
y_dim = predict_dim # Dimension of y #equal to predict_len
h_dim = 20 # Dimension of the hidden states in RNN
z_dim = 3 # Dimension of the latent variable z
d_dim = 2 # Dimension of the latent variable d
n_layers =  1 # Number of the layers of the RNN
clip = 10 # Gradient clips
learning_rate = 1e-3 # Learning rate
batch_size = 64 # Batch size


# In[31]:


#print_every = 100
save_every = 1
save_best = True

##manual seed
#seed = 128
#torch.manual_seed(seed)

## Init model + optimizer
model = DSSSM(x_dim, y_dim, h_dim, z_dim, d_dim, n_layers, device,bidirection).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Training
start = time.time()
loss_train_list = []
loss_valid_list = []
loss_test_list = []
best_validation = 1e5


# In[32]:


directory = os.path.join("Checkpoint/Lorenz")
directoryBest = directory
figdirectory = os.path.join(directory,"figures")
print(directory)
print(directoryBest)
print(figdirectory)


# In[33]:


if not os.path.exists(directoryBest):
    os.makedirs(directoryBest)
if not os.path.exists(figdirectory):
    os.makedirs(figdirectory)
figdirectory = figdirectory+'/'+dataname +'_'


# In[34]:


#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min',factor = 0.1,patience = 10)
# initialize the early_stopping object
early_stopping = EarlyStopping(20, verbose=True)


# In[35]:


restore


# In[36]:


n_epochs = 1000 # Number of epochs for traning 
if restore == False:
    for epoch in range(1, n_epochs + 1):

        ### Training
        all_d_t_sampled_train,all_z_t_sampled_train,loss_train,all_d_posterior_train,all_z_posterior_mean_train = train(model,optimizer,trainX,trainY,epoch,batch_size,n_epochs)

        ### Validation
        all_d_t_sampled_valid,all_z_t_sampled_valid,loss_valid,all_d_posterior_valid,all_z_posterior_mean_valid = test(model,validX,validY,epoch,"valid")
        #loss_valid =  loss_train
        ### Testing
        all_d_t_sampled_test,all_z_t_sampled_test,loss_test,all_d_posterior_test,all_z_posterior_mean_test = test(model,testX,testY,epoch,"test")
        loss_train_list.append(loss_train)
        loss_valid_list.append(loss_valid)
        loss_test_list.append(loss_test)

    #     ### Save the results to tensorboard
    #     #writer.add_scalar("scalar/train_loss",loss_train,epoch)
    #     #writer.add_scalar("scalar/valid_loss",loss_valid,epoch)



        ### Save checkpoint
    #     if (epoch % save_every == 0):
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #         torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict':optimizer.state_dict(),
    #                 'loss': loss_train,
    #             }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))


        if save_best:        
            if not os.path.exists(directoryBest):
                os.makedirs(directoryBest)
            if (loss_valid < best_validation):
                best_validation = copy.deepcopy(loss_valid)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss': loss_train,
                },os.path.join(directoryBest,'best.tar'))

        scheduler.step(loss_valid)
        print("Learning rate:",optimizer.param_groups[0]['lr'])
        
        loss_valid_average = np.average(loss_valid_list)           
        early_stopping(loss_valid_average, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Running Time:", time.time()-start)

    #writer.close()


# In[37]:


if restore == False:
    plt.plot(np.array(loss_train_list),label="train")
    plt.plot(np.array(loss_valid_list),label="validation")
    plt.plot(np.array(loss_test_list),label="test")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show();


# In[38]:


next(model.parameters())


# # Validation 
# Chose the best model with goodest validation results

# In[39]:


import importlib
importlib.reload(DataProcessing)
from DataProcessing import *

import importlib
importlib.reload(DSSSMCode)
from DSSSMCode import *


# In[44]:


# Reload the parameters
print(directoryBest,os.listdir(directoryBest))
PATH = os.path.join(directoryBest,'checkpoint.tar')


# In[45]:


PATH


# In[46]:


model = DSSSM(x_dim,y_dim, h_dim, z_dim, d_dim,n_layers,device,bidirection).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
checkpoint = torch.load(PATH,map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


# In[47]:


## The Parameters
print("Epoch:",epoch)
total_params = sum(p.numel() for p in model.parameters())
#total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("The total number of parameters:",total_params)

## Print Model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
## Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])


# In[48]:


model.TransitionMatrix()


# # Inference

# In[49]:


# d_original = pd.read_csv('./DataSimulation/190603_simulation_data_MoreStocha_d.csv',header=None)[-(testX.size(1)+timestep):].values.reshape(-1)
# z_original = pd.read_csv('./DataSimulation/190603_simulation_data_MoreStocha_z.csv',header=None)[-(testX.size(1)+timestep):].values.reshape(-1)
# y_original2 = pd.read_csv('./DataSimulation/190603_simulation_data_MoreStocha_y.csv',header=None)[-(testX.size(1)+timestep-1):].values.reshape(-1)


# In[50]:


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(colors[:3])


# In[51]:


import seaborn as sns


# In[52]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def classification_scores(testy,yhat_classes):
    accuracy = accuracy_score(testy, yhat_classes)
    if accuracy < 0.5:
        yhat_classes = 1-yhat_classes
    accuracy = accuracy_score(testy, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(testy, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(testy, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1)


# In[53]:


trainX.shape


# In[54]:


test_len


# In[55]:


print("3D size(Y):",trainY.shape,validY.shape,testY.shape)
trainX2 = torch.from_numpy(RawDataOriginal[:1000,:,:]).float().to(device)
trainY2 = torch.from_numpy(RawDataOriginal[1:1001,:,:]).float().to(device)
testX2 = torch.from_numpy(RawDataOriginal[(-test_len-1):-1,:,:]).float().to(device)
testY2 = torch.from_numpy(RawDataOriginal[-test_len:,:,:]).float().to(device)
print("3D size(Y):",trainY2.shape,testY2.shape)
print("3D size(X):",trainX2.shape,testX2.shape)


# In[56]:


#sns.set()
all_d_t_sampled_plot_test,all_z_t_sampled_test,loss_test,all_d_posterior_test,all_z_posterior_mean_test = test(model,trainX2,trainY2,0,"test")
latents = z_true[0,1:1001,:]
categories = all_d_t_sampled_plot_test[:,1:,:].reshape(-1)
accuracy = sum(states.numpy().reshape(-1)[1:(trainX.shape[1]+1)] == categories)/len(categories)
classification_scores(states.numpy().reshape(-1)[1:(trainX.shape[1]+1)],categories)
colormap = np.array(['r', 'b','g','y'])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
plt.show();

latents = all_z_posterior_mean_test[0,1:,:]
colormap = np.array(['r', 'b','g','y'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
plt.show();


# In[57]:


######
all_d_t_sampled_plot_test,all_z_t_sampled_test,loss_test,all_d_posterior_test,all_z_posterior_mean_test = test(model,testX2,testY2,0,"test")
latents = z_true[0,-testX2.shape[0]:,:]

categories = all_d_t_sampled_plot_test[:,1:,:].reshape(-1)
accuracy = sum(states.numpy().reshape(-1)[-testX2.shape[0]:] == categories)/len(categories)
classification_scores(states.numpy().reshape(-1)[-testX.shape[1]:],categories)
colormap = np.array(['r', 'b','g','y'])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
plt.show();

latents = all_z_posterior_mean_test[0,1:,:]
colormap = np.array(['r', 'b','g','y'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
plt.show();


# In[58]:


# #sns.set()
# all_d_t_sampled_plot_test,all_z_t_sampled_test,loss_test,all_d_posterior_test,all_z_posterior_mean_test = test(model,trainX,trainY,0,"test")
# latents = z_true[0,timestep:(trainX.shape[1]+timestep),:]
# categories = all_d_t_sampled_plot_test[:,-1,:].reshape(-1)
# accuracy = sum(states.numpy().reshape(-1)[1:(trainX.shape[1]+1)] == categories)/len(categories)
# classification_scores(states.numpy().reshape(-1)[1:(trainX.shape[1]+1)],categories)

# colormap = np.array(['r', 'b','g','y'])
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
# plt.show();

# ######
# all_d_t_sampled_plot_test,all_z_t_sampled_test,loss_test,all_d_posterior_test,all_z_posterior_mean_test = test(model,testX,testY,0,"test")
# latents = z_true[0,-testX.shape[1]:,:]
# categories = all_d_t_sampled_plot_test[:,-1,:].reshape(-1)
# accuracy = sum(states.numpy().reshape(-1)[-testX.shape[1]:] == categories)/len(categories)
# classification_scores(states.numpy().reshape(-1)[-testX.shape[1]:],categories)
# colormap = np.array(['r', 'b','g','y'])
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
# plt.show();


# In[59]:


# forecaststep = 1
# MC_S = 200
# forecast_MC,forecast_d_MC,forecast_z_MC = model._forecastingMultiStep(trainX,trainY,forecaststep,MC_S)

# # forecast_z_MC.shape
# # forecast_d_MC.shape

# forecast_d_MC_final = np.mean(forecast_d_MC[:,-1,:,:],axis = 0)
# forecast_z_MC_final = np.mean(forecast_z_MC[:,-1,:,:],axis = 0)

# forecast_d_MC_argmax = []
# for i in range(d_dim):
#     forecast_d_MC_argmax.append(np.sum(forecast_d_MC[:,-1,:,:] == i,axis=0))
# forecast_d_MC_argmax = np.argmax(np.array(forecast_d_MC_argmax),axis=0).reshape(-1)

# categories = forecast_d_MC_argmax
# accuracy = sum(states.numpy().reshape(-1)[(timestep+1):(trainX.shape[1]+timestep+1)] == categories)/len(categories)
# if accuracy < 0.5:
#     accuracy = 1-accuracy
# print("Accuracy:",accuracy)
# classification_scores(states.numpy().reshape(-1)[1:(trainX.shape[1]+1)],categories)
# latents = forecast_z_MC_final

# colormap = np.array(['r', 'b','g','y'])
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
# plt.show();

# normalized = True
# if normalized:
#     all_testForecast = normalize_invert(forecast_MC.squeeze(1).transpose(1,0,2),moments)
    
# testForecast_mean = np.mean(all_testForecast,axis = 1)
# testOriginal = RawDataOriginal[1:(trainX.shape[1]+1),:,:].reshape(-1,RawDataOriginal.shape[2])
# # testForecast_mean.shape
# # testOriginal.shape

# evaluation(testForecast_mean.T,testOriginal.T)


# In[60]:


forecaststep = 1
MC_S = 200
forecast_MC,forecast_d_MC,forecast_z_MC = model._forecastingMultiStep(testX,testY,forecaststep,MC_S)

# forecast_z_MC.shape
# forecast_d_MC.shape

forecast_d_MC_final = np.mean(forecast_d_MC[:,-1,:,:],axis = 0)
forecast_z_MC_final = np.mean(forecast_z_MC[:,-1,:,:],axis = 0)

forecast_d_MC_argmax = []
for i in range(d_dim):
    forecast_d_MC_argmax.append(np.sum(forecast_d_MC[:,-1,:,:] == i,axis=0))
forecast_d_MC_argmax = np.argmax(np.array(forecast_d_MC_argmax),axis=0).reshape(-1)

categories = forecast_d_MC_argmax
accuracy = sum(states.numpy().reshape(-1)[-testX.shape[1]:] == categories)/len(categories)

if accuracy < 0.5:
    accuracy = 1 - accuracy
# print("Accuracy:",accuracy)
classification_scores(states.numpy().reshape(-1)[-testX.shape[1]:],categories)
latents = forecast_z_MC_final

colormap = np.array(['r', 'b','g','y'])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
plt.show();

normalized = True
if normalized:
    all_testForecast = normalize_invert(forecast_MC.squeeze(1).transpose(1,0,2),moments)
    
testForecast_mean = np.mean(all_testForecast,axis = 1)
testOriginal = RawDataOriginal[-testX.shape[1]:,:,:].reshape(-1,RawDataOriginal.shape[2])
# testForecast_mean.shape
# testOriginal.shape

evaluation(testForecast_mean.T,testOriginal.T)


# In[61]:


latents = z_true[0,-testX.shape[1]:,:]
colormap = np.array(['r', 'b','g','y'])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(latents[:,0],latents[:,1],latents[:,2],c=colormap[categories])
plt.show();


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




