# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:55:48 2021

@author: Mahmoud Zaher
"""


import numpy as np

# This function is used to perform the pilot assignment procedure described in the paper
def assign_pilots(K, tau_p, betas):
    assigned = -1 * np.ones(K)
    assigned[0:tau_p] = np.random.permutation(tau_p)
    for k in range(tau_p, K):
        l = np.argmax(betas[k,:])
        interference = np.zeros(tau_p)
        for tau in range(0, tau_p):
            interference[tau] = np.sum(betas[assigned == tau,l])
            
        assigned[k] = np.argmin(interference)
    
    return assigned