#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:42:20 2019

@author: Sucharita Chakraborty
"""

import numpy as np
def sorted_SE(SE):
    
    K = SE.shape[0]
    nbrOfSetups = SE.shape[1]
    A=np.reshape(SE[:,0:nbrOfSetups],(K*nbrOfSetups,1))
    sorted_SE = A[A[:,0].argsort(kind='mergesort')]
    
    return (sorted_SE)
