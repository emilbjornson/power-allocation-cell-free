# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:59:59 2021

@author: Mahmoud Zaher
"""

import numpy as np
import tensorflow as tf
import pathlib
from pickle import load
import time

# This function is for the CDNN model predictions
def pred_func(betas_DNN, Pmax, NoOfSetups, modelname, cluster_size):
    base_path = pathlib.Path().absolute()
    foldername = str(base_path) + '/pAssignModels/DNNmodels-clustered/'
    scaler = load(open(foldername + 'scaler.pkl', 'rb'))
    K = betas_DNN.shape[0]
    L = betas_DNN.shape[1]
    nbrOfSetups = betas_DNN.shape[2]
    mu = np.zeros((K+1,L,nbrOfSetups))
    
    for l in range(0, L, cluster_size):
        if l == 15:
            l = 13
        filename = modelname + str(l+1)
        DDNN_model = tf.keras.models.load_model(foldername + filename)
        
        start_time = time.perf_counter()
        
        betas = np.zeros((NoOfSetups, cluster_size*K))
        for c in range(0, cluster_size):
            betas[:,c*K:(c+1)*K] = betas_DNN[:,l+c,:].T
            
        betas = 10*np.log10(betas*1000) # dB scale
    
        betas = scaler.transform(betas)
        DNNoutput = DDNN_model.predict(betas).T
        for c in range(0, cluster_size):
            index = list(np.arange(c*K,(c+1)*K)) + [cluster_size*K + c]
            mu[:,l+c,:]  = DNNoutput[index]
            
        stop_time = time.perf_counter() - start_time
        
    return mu , stop_time