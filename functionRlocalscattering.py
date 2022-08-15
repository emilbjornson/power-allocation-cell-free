# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:15:20 2019

@author: Sucharita Chakraborty
"""

import numpy as np 
import math
from scipy.linalg import toeplitz

def R(M,theta,ASDdeg):
    ASD = ASDdeg*math.pi/180
    antennaSpacing = 0.5
    #The correlation matrix has a Toeplitz structure, so we only need to
    #compute the first row of the matrix
    firstRow = np.zeros((M,1), dtype='complex')
    
    for column in range(0,M):
    
        #Compute the approximated integral as in (2.24)
        firstRow[column] = np.exp(1j*2*math.pi*antennaSpacing*math.sin(theta)*column)*np.exp(-ASD**2/2 * ( 2*math.pi*antennaSpacing*math.cos(theta)*column )**2)
    
    R = toeplitz(firstRow)
    return R