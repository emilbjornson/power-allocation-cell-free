#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 23:34:47 2019

@author: Sucharita Chakraborty
"""
import numpy as np

def Calculate_SINR_and_SE_DL(signal, interference, prelogfactor, gammaEqual, Pmax):
    L = signal.shape[0]
    K = signal.shape[1]
    
    SE_MR_equal = np.zeros(K)
    
    #Scale the square roots of power coefficients to satisfy all the per-AP power constraints
    #These coefficients correspond to the vectors\vect{\mu}_k in (8) in the paper
    gammaEqual = gammaEqual*np.sqrt(Pmax)/np.max(np.linalg.norm(gammaEqual,axis=0))
    
    #Compute the SEs as in (6) in the paper
    for k in range(0,K):
        SINRnumerator = np.power(np.matmul(signal[:,k:k+1].reshape(1,L), gammaEqual[k:k+1,:].reshape(L,1)),2)
        SINRdenominator = 1-SINRnumerator
        for i in range(0,K):
            SINRdenominator = SINRdenominator+np.matmul(gammaEqual[i:i+1,:].reshape(1,L), np.matmul(interference[:,:,k,i],gammaEqual[i:i+1,:].reshape(L,1)))
   
        SE_MR_equal[k] = prelogfactor*np.log2(1 + SINRnumerator/SINRdenominator)

    
    return SE_MR_equal
