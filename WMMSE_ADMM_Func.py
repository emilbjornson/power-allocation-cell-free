#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:20:11 2020

@author: Sucharita Chakraborty
"""

import numpy as np


#The functions in this script implement Algorithm 2 and Algorithm 1 using ADMM

#This function is for saving run time, its implementataion is the same as the last function in this script.
#The difference from the last function is that arrays except power coefficients are not saved for a fair comparison of run times.

def WMMSE_ADMM_timing(L,K, Pmax, a_MR, B_MR):
    
    #Initialize the square roots of the power coefficients
    mu_MR_WMMSE = 0.1*np.sqrt(Pmax/K)*np.ones((L,K))
    
    #Solution accuracy (epsilon_wmmse)
    delta = 0.01
    
    #ADMM penalty parameter
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
    objUpper = np.sum( np.log2(1 + SINR) )

    #Continue iterations until stopping criterion in (52) is satisfied (prelogfactors are omitted)
    while np.power(np.abs(objUpper - objLower), 2) > delta:
        
        #Update the old objective by the current objective        
        objLower = objUpper
        
        #Equation (53)
        v = np.sqrt(SINRnumerator) / SINRdenominator

        #Equation (56)
        e = 1 - SINRnumerator/SINRdenominator
        
        #Equation (55)
        w = 1/e 
        
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
        objUpper = np.sum(np.log2(1+SINR))
    
    #Square roots of the power coefficients    

    return mu_MR_WMMSE.T



def WMMSE_ADMM_iteration(L,K, Pmax, prelogFactor, a_MR, B_MR):
    
    mu_MR_WMMSE = 0.1*np.sqrt(Pmax/K)*np.ones((L,K))
    delta = 0.01
    
    penalty = 0.001
    

    objLower=0    


    SINRnumerator = np.power( np.abs( np.sum(mu_MR_WMMSE * a_MR , axis = 0)), 2)
    SINRdenominator = np.ones(K)
    for k in range (0,K):
        for i in range(0,K):
            SINRdenominator[k] = SINRdenominator[k]+ mu_MR_WMMSE[:,i:i+1].T @ B_MR[:,:,k,i] @ mu_MR_WMMSE[:,i:i+1]
            
    SINR = SINRnumerator/(SINRdenominator-SINRnumerator)
    objUpper = np.sum( np.log2(1 + SINR) )
    
    SE_WMMSE_ADMM = np.zeros((200), dtype = 'float')
    objval = np.zeros((200))
    iteration = 0
    objval_1blk = np.zeros((1000))
    objval_2blk = np.zeros((1000))
    while np.power(np.abs(objUpper - objLower), 2) > delta:
                
        SE_WMMSE_ADMM[iteration] = prelogFactor * np.sum( np.log2(1 + SINR) )

        objLower=objUpper
        v = np.sqrt(SINRnumerator) / SINRdenominator

        e = 1 - SINRnumerator/SINRdenominator
        w = 1/e 
        
        
        A2 = np.zeros((L,L,K), dtype = 'float')
        Ainv = np.zeros((L,L,K), dtype = 'float')
        c2 = np.zeros((L,K), dtype = 'float')
        
        for k in range (0,K):
            for i in range (0,K):
                A2[:,:,k] = A2[:,:,k] + w[i] * v[i]**2  * B_MR[:,:,i,k]
                
            Ainv[:,:,k] = np.linalg.inv(A2[:,:,k]+(penalty/2)*np.eye(L))
            
            c2[:,k] = w[k] * v[k]  * a_MR[:,k]
        # Variable initialization for ADMM
        g = np.zeros((L,K))
        
        
        
        #Start the ADMM
        
        diff = 100
        inner_iteration = 0
        q = mu_MR_WMMSE*(1+np.random.rand(L,K))
        while diff>0.001: 
            
            for k in range (0,K):
               
                c= c2[:,k:k+1] + (penalty/2) * (q[:,k:k+1] + g[:,k:k+1])
                
                mu_MR_WMMSE[:,k:k+1]=  Ainv[:,:,k] @ c
           
            
                
                
            if iteration==0:
                objvalk = np.zeros((K))
                for k in range(0, K):
                    objvalk[k] = mu_MR_WMMSE[:,k:k+1].T @ A2[:,:,k] @ mu_MR_WMMSE[:,k:k+1] - 2 * c2[:,k:k+1].T @ mu_MR_WMMSE[:,k:k+1]
                objval_1blk[inner_iteration] = np.sum (objvalk)   
                
            q = mu_MR_WMMSE - g
            q_norm=np.linalg.norm(q,axis=1)
            for l in np.argwhere(q_norm>np.sqrt(Pmax)):
                q[l,:]=q[l,:]*np.sqrt(Pmax)/q_norm[l]
                
            if iteration==0:
                objvalk = np.zeros((K))
                for k in range(0, K):
                    objvalk[k] = q[:,k:k+1].T @ A2[:,:,k] @ q[:,k:k+1] - 2 * c2[:,k:k+1].T @ q[:,k:k+1]
                objval_2blk[inner_iteration] = np.sum (objvalk)
            
          
                
            inner_iteration += 1
           
          
        
            #Update dual variable g
            g = q - mu_MR_WMMSE + g
            
#            
            if inner_iteration>5:
                
                diff = np.linalg.norm(mu_MR_WMMSE-q,'fro')/np.linalg.norm(mu_MR_WMMSE,'fro')
                
            
       

        if iteration==0:
            inner_iteration2 = inner_iteration
            
        SINRnumerator = np.power( np.abs( np.sum(mu_MR_WMMSE  * a_MR , axis = 0)), 2)
        SINRdenominator = np.ones(K)
        for k in range (0,K):
            for i in range(0,K):
                SINRdenominator[k] = SINRdenominator[k]+mu_MR_WMMSE[:,i:i+1].T@ B_MR[:,:,k,i]@ mu_MR_WMMSE[:,i:i+1]
        
        SINR = SINRnumerator/(SINRdenominator-SINRnumerator)
        objUpper = np.sum(np.log2(1+SINR))
        
        A1 = np.zeros((L,L,K), dtype = 'float')
        c1 = np.zeros((L,K), dtype = 'float')
        for k in range (0,K):
            for i in range (0,K):
                A1[:,:,k] = A1[:,:,k] + w[i] * v[i]**2  * B_MR[:,:,i,k]
                
            
            c1[:,k] = w[k] * v[k]* a_MR[:,k]

        objvalk = np.zeros((K))
        for k in range(0, K):
            objvalk[k] = mu_MR_WMMSE[:,k:k+1].T @ A1[:,:,k] @ mu_MR_WMMSE[:,k:k+1] - 2 * c1[:,k:k+1].T @ mu_MR_WMMSE[:,k:k+1]
        objval[iteration] = np.sum (objvalk)

        iteration += 1
      
    return (mu_MR_WMMSE.T, SE_WMMSE_ADMM, objval, iteration, objval_1blk, objval_2blk, inner_iteration2)

            