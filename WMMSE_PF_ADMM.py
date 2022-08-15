# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:36:28 2021

@author: Mahmoud Zaher
"""

import numpy as np


#The functions in this script implement proportional fairness objective using ADMM

def WMMSE_ADMM_timing(L,K, Pmax, a_MR, B_MR):
    
    #Initialize the square roots of the power coefficients
    mu_MR_WMMSE = 0.1*np.sqrt(Pmax/K)*np.ones((L,K))
    
    #Solution accuracy (epsilon_wmmse) 0.01
    delta = 0.001
    
    #ADMM penalty parameter 0.001
    penalty = 0.001
   
        
    #Initialize the objective function as zero
    objLower=0

    #This is for computing the objective function and computing the other terms later

    SINRnumerator = np.power( np.abs( np.sum(mu_MR_WMMSE * a_MR , axis = 0)), 2)
    SINRdenominator = np.ones(K)
    for k in range (0,K):
        for i in range(0,K):
            SINRdenominator[k] = SINRdenominator[k]+ mu_MR_WMMSE[:,i:i+1].T @ B_MR[:,:,k,i] @ mu_MR_WMMSE[:,i:i+1]
            
    SINR = SINRnumerator/(SINRdenominator-SINRnumerator)
    
    #Current objective function
    objUpper = np.sum( np.log(np.log2(1 + SINR)) )

    #Continue iterations until stopping criterion in (52) is satisfied (prelogfactors are omitted)
    while np.power(np.abs(objUpper - objLower), 2) > delta:
        
        #Update the old objective by the current objective        
        objLower = objUpper
        
        #Equation (53)
        v = np.sqrt(SINRnumerator) / SINRdenominator

        #Equation (56)
        e = 1 - SINRnumerator/SINRdenominator
        
        #Equation (55)
        w = -1/(e * np.log(e)) 
        
        #Make preparations for ADMM algorithm in Algorithm 1
        Ainv = np.zeros((L,L,K), dtype = 'float')
        c = np.zeros((L,K), dtype = 'float')
        
        for k in range (0,K):
            A = (penalty/2)*np.eye(L) 
            for i in range (0,K):
                A = A + w[i] * v[i]**2 * B_MR[:,:,i,k]
                
            Ainv[:,:,k] = np.linalg.inv(A)
                
            
            c[:,k] = w[k] * v[k] * a_MR[:,k]
        # Dual variable initialization for ADMM
        g = np.zeros((L,K))

        #Start the ADMM
        
        #Initial large difference in (51) to start the algoritm
        diff = 100
    
        #Set the iteration counter for ADMM
        inner_iteration = 0
        
        #Perturbed random initialization mentioned in the paper
        q = mu_MR_WMMSE*(1+np.random.rand(L,K))
        
        #Run Algorithm 1 until the stopping criterion in (51) is satisfied
        while diff>0.001: 
            inner_iteration += 1
            
            
            #Update the first block of primal variables as in (48)

            for k in range (0,K):
               
                c2= c[:,k:k+1] + (penalty/2) * (q[:,k:k+1] + g[:,k:k+1])
                
                mu_MR_WMMSE[:,k:k+1]=  Ainv[:,:,k] @ c2
        
           
            #Update the second block of primal variables as in (49)
  
            q = mu_MR_WMMSE - g
            q_norm = np.linalg.norm(q,axis=1)
            for l in np.argwhere(q_norm>np.sqrt(Pmax)):
                q[l,:] = q[l,:]*np.sqrt(Pmax)/q_norm[l]
            
        
            #Update dual variable g as in (50) 
            g = q - mu_MR_WMMSE + g
            
            #To prevent any misconvergence issues in the first iterations, we guarentee at least 5 ADMM iterations are run
            if inner_iteration>5:
                
                diff = np.linalg.norm(mu_MR_WMMSE-q,'fro')/np.linalg.norm(mu_MR_WMMSE,'fro')
                
            
        # Update the variables and compute the new objective
   
        SINRnumerator = np.power( np.abs( np.sum(mu_MR_WMMSE  * a_MR , axis = 0)), 2)
        SINRdenominator = np.ones(K)
        for k in range (0,K):
            for i in range(0,K):
                SINRdenominator[k] = SINRdenominator[k]+mu_MR_WMMSE[:,i:i+1].T@ B_MR[:,:,k,i]@ mu_MR_WMMSE[:,i:i+1]
        
        SINR = SINRnumerator/(SINRdenominator-SINRnumerator)
        objUpper = np.sum( np.log(np.log2(1+SINR)) )
    
    #Square roots of the power coefficients    

    return mu_MR_WMMSE.T
