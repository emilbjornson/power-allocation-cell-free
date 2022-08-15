# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:34:04 2021

@author: Mahmoud Zaher
"""


# This script is used for training the CDNN models for the different
# optimization objectives and precoding schemes considered.
import tensorflow as tf
from keras import backend as k
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from os import makedirs
from sklearn import preprocessing
#import pandas as pd
import pathlib
from pickle import dump

base_path = pathlib.Path().absolute()
filename = str(base_path) + '/pAssign_storage/'
models_filename = str(base_path) + '/pAssignModels/DNNmodels-clustered/'


#reproducible results using Keras
sd = 42# Here sd means seed.

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(sd)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random as rn
rn.seed(sd)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(sd)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(sd)

# 5. Configure a new global `tensorflow` session
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)


# Loading input to the NN

mu_MR_sumSE_DNN = np.load(filename + 'dataset_mu_MR_WMMSE_ADMM.npy',allow_pickle=True)  
mu_RZF_sumSE_DNN = np.load(filename + 'dataset_mu_RZF_WMMSE_ADMM.npy',allow_pickle=True)   
mu_MR_PF_DNN = np.load(filename + 'dataset_mu_MR_WMMSE_PF_ADMM.npy',allow_pickle=True)   
mu_RZF_PF_DNN = np.load(filename + 'dataset_mu_RZF_WMMSE_PF_ADMM.npy',allow_pickle=True)
betas_DNN = np.load(filename + 'dataset_betas.npy',allow_pickle=True)
     

#Maximum downlink transmit power per BS (mW)
Pmax = 1000
K = betas_DNN.shape[0]
L = betas_DNN.shape[1]
NoOfSetups = betas_DNN.shape[2]

# Make sure the sum over the K UEs gives Pmax for each AP in each setup- (might not be necessary)
for n in range(NoOfSetups):
    mu_MR_sumSE_DNN[:,:,n] = mu_MR_sumSE_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_MR_sumSE_DNN[:,:,n],2), axis=0) )) )
    mu_RZF_sumSE_DNN[:,:,n] = mu_RZF_sumSE_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_RZF_sumSE_DNN[:,:,n],2), axis=0) )) )
    mu_MR_PF_DNN[:,:,n] = mu_MR_PF_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_MR_PF_DNN[:,:,n],2), axis=0) )) )
    mu_RZF_PF_DNN[:,:,n] = mu_RZF_PF_DNN[:,:,n] * np.sqrt( Pmax/(np.max (np.sum(np.power(mu_RZF_PF_DNN[:,:,n],2), axis=0) )) )

# Maximum number of epochs
Num_epoch = 30
# Batch size
N_batch_size = 128
K_initializer = 'random_normal'
B_initializer = 'random_normal'
cluster_size = 3


##########################     MODEL FFNN   ##########################

# Optimizer
# adam = optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-07) #Decay = 0.1 not working

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0., patience=20, verbose=0, mode='auto')
callback = [early_stopping]

for l in range(9, 10, cluster_size):
    model = Sequential()
    model.add(Dense(128, input_dim=cluster_size*K, activation="linear", name = 'layer1', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(512, input_dim=128, activation="elu", name = 'layer2', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(256, input_dim=512, activation="tanh", name = 'layer3', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(128, input_dim=256, activation="tanh", name = 'layer4', kernel_initializer = K_initializer, bias_initializer=B_initializer))
    model.add(Dense(cluster_size*(K+1), input_dim=128, activation="relu", name = 'layer5', kernel_initializer = K_initializer, bias_initializer=B_initializer))

    NoOfSetups = betas_DNN.shape[2]
    #Preparing inputs for NN
    #beta vector preparation (removing outliers and scaling)
    if l == 15:
        l = 13
    betas = np.zeros((NoOfSetups, cluster_size*K))
    for c in range(0, cluster_size):
        betas[:,c*K:(c+1)*K] = betas_DNN[:,l+c,:].T
    betas = 10*np.log10(betas*1000) # dB scale
    big_values = []
    for i in range(0, NoOfSetups):
        if np.any(betas[i,:] > 37):
            big_values = big_values + [i]
    betas = np.delete(betas, big_values, axis=0)
    NoOfSetups = betas.shape[0]
    scaler = preprocessing.RobustScaler(
                                with_centering=False,
                                with_scaling=True,
                                quantile_range=(25.0, 75.0),
                                copy=True,)
    
    
    betas = scaler.fit_transform(betas)
    DNNinput = betas
    x_train = DNNinput[0:NoOfSetups-100,:]
    
    ### Choose one of the following 4 options to train the model for
    # mu_MR_sumSE_DNN, mu_RZF_sumSE_DNN, mu_MR_PF_DNN, mu_RZF_PF_DNN
    ## mu preparation
    mu = np.zeros((NoOfSetups, cluster_size*K))
    temp = np.zeros((NoOfSetups-100, cluster_size))
    for c in range(0, cluster_size):
        mu[:,c*K:(c+1)*K] = np.abs(np.delete(mu_MR_sumSE_DNN[:,l+c,:].T, big_values, axis=0))
        temp[:,c:c+1] = np.sqrt( np.reshape(sum((mu[0:NoOfSetups-100,c*K:(c+1)*K].T) ** 2), (NoOfSetups-100, 1)) / Pmax)

    y_train = np.concatenate((mu[0:NoOfSetups-100,:], temp), axis=1)
    small_values = []
    small_val = 5 / np.sqrt(Pmax)
    for i in range(0, NoOfSetups-100):
        counter = 0
        for j in range(0, cluster_size):
            if y_train[i,cluster_size*K+j] < small_val:
                counter += 1
        if counter == cluster_size:
            small_values = small_values + [i]
        
    y_train = np.delete(y_train, small_values, axis=0)
    NoOfSetups = y_train.shape[0]
    
#    y_train[y_train < 0.001] = 0.001
    # Normalization with sqrt(Pmax) must be done separately for each AP as follows
    for c in range(0, cluster_size):
        y_train[:,c*K:(c+1)*K] = np.sqrt(K) * tf.keras.utils.normalize(y_train[:,c*K:(c+1)*K], axis=1)

    
    #############################################################
    
    x_train = np.delete(x_train, small_values, axis=0)
    
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    
    print(model.summary())


    k.set_value(model.optimizer.lr, 0.001)
    history = model.fit(x_train, y_train, epochs = Num_epoch, batch_size = N_batch_size, validation_split = 0.1, callbacks=callback)
    k.set_value(model.optimizer.lr, 0.0001)
    history2 = model.fit(x_train, y_train, epochs = 20, batch_size = N_batch_size, validation_split = 0.1, callbacks=callback)

    x_test = DNNinput[NoOfSetups-100:NoOfSetups,:]
    ## Assign y_test based on the model choice above
    y_test = mu[NoOfSetups-100:NoOfSetups,:]
    for c in range(0, cluster_size):
        y_test[:,c*K:(c+1)*K] = np.sqrt(K) * tf.keras.utils.normalize(y_test[:,c*K:(c+1)*K], axis=1)
    y_predictions = model.predict(x_test)
    test_mse = np.mean((y_test - y_predictions[:,0:cluster_size*K])**2)
    test_mseAP4 = np.mean((y_test[:,0:20] - y_predictions[:,0:20])**2)
    print('Test MSE:' + str(test_mse))
    print('Test MSEAP4:' + str(test_mseAP4))
    
    # Save models (You may adjust the name based on the chosen model)
    model.save(models_filename + 'Trained_models_for_mu_RZF_sumSE_CDNN_for_AP' + str(l+1))
    print('Saved model %s' % models_filename + str(l+1))

# Save Robust Scaler for betas (only required once)
# dump(scaler, open(models_filename + 'scaler.pkl', 'wb'))

# For plotting the loss curve
# a = history.history['loss']
# b = history2.history['loss']
# loss = np.concatenate((a,b))

# a = history.history['val_loss']
# b = history2.history['val_loss']
# val_loss = np.concatenate((a,b))

# factor = np.mean(np.var(y_predictions,axis = 0))
# loss = loss/factor
# val_loss = val_loss/factor

# Xvals = np.linspace(1,50,50)
# fig = plt.figure(figsize=(16,12))
# plt.plot(loss, '-.', color='#000000ff', label='Train', linewidth=4)  

# plt.plot(val_loss, color='r', label='Test', linewidth=4)
                    
# plt.legend(loc="upper right",prop={'size':32})
# plt.grid(True)
# plt.rc('xtick',labelsize=36)
# plt.rc('ytick',labelsize=36)
# plt.xlim([0, 50])
# plt.ylim([0, 0.5])
# plt.xlabel('Number of epochs', fontsize=34)
# plt.ylabel('Loss', fontsize=34)
