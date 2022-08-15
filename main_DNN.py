#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:24:20 2021

@author: Mahmoud Zaher
"""



import numpy as np
import numpy.matlib
import pickle
import matplotlib.pyplot as plt
import scipy.linalg as sl
import time
import pathlib
import pilot_assignment
np.random.seed(4873256)#int(time.time()))   # To get the same UE distribution every time you try something new

base_path = pathlib.Path().absolute()
filename = str(base_path) + '/new_storage/'
result_vectors_save = str(base_path) + '/results_vectors/'

#Number of APs
L = 16

#Number of UEs 
K = 20

#Select length of pilot of UEs
tau_p = 10 #Orthogonal sequences if tau_p=K, else tau_p = 10

#Select length of coherence block
tau_c = 200

prelogFactor = (tau_c-tau_p)/(tau_c)

#Number of AP antennas
M = 4

#Select the number of setups with random UE locations
nbrOfSetups = 2000

#Select the number of channel realizations per setup
nbrOfRealizations = 100

## Model parameters

#Set the length in meters of the total square area
squareLength = 1000

#Number of APs per dimension
nbrAPsPerDim = np.int(np.sqrt(L))

#Pathloss exponent
alpha = 3.67

#Average channel gain in dB at a reference distance of 1 meter.
constantTerm = -30.5

#Standard deviation of shadow fading
sigma_sf = 1

#Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2 #Half wavelength distance

#Distance between APs in vertical/horizontal direction
interAPDistance = np.int(squareLength/nbrAPsPerDim)

## Propagation parameters

#Communication bandwidth
B = 20e6

#Total uplink transmit power per UE (mW)
p = 100

#Maximum downlink transmit power per AP (mW)
Pmax = 1000

#Compute downlink power per UE in case of equal power allocation
rhoEqual = (Pmax/K)*np.ones((K,L))

#Square roots of power coefficients for equal power allocation
gammaEqual = np.sqrt(rhoEqual)   

#Prepare power coefficients for the benchmark in [12]
rho_Giovanni19 = np.zeros((K,L,nbrOfSetups)) 

#Vertical distance between APs and UEs
distanceVertical = 10

#Define noise figure at AP (in dB)
noiseFigure = 7

#Compute noise power
noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

#Angular standard deviation in the local scattering model (in degrees)
ASDdeg = 10

#Store identity matrix of size M x M
eyeM = np.identity(M)

#Prepare to save simulation results
#Preallocate SE terms for MR and RZF precoding schemes


#Equal power allocation
SE_MR_equal = np.zeros((K,nbrOfSetups))
SE_RZF_equal = np.zeros((K,nbrOfSetups))

#The benchmark in [12]
SE_MR_Giovanni19 = np.zeros((K,nbrOfSetups))
SE_RZF_Giovanni19 = np.zeros((K,nbrOfSetups))

#Proposed algorithm with ADMM and WMMSE for sum-SE maximization
SE_MR_WMMSE_ADMM = np.zeros((K,nbrOfSetups))
SE_RZF_WMMSE_ADMM = np.zeros((K,nbrOfSetups))

#DNN allocation
SE_MR_sumSE_DDNN = np.zeros((K,nbrOfSetups))
SE_RZF_sumSE_DDNN = np.zeros((K,nbrOfSetups))

SE_MR_PF_DDNN = np.zeros((K,nbrOfSetups))
SE_RZF_PF_DDNN = np.zeros((K,nbrOfSetups))

SE_MR_sumSE_DDNN_SI = np.zeros((K,nbrOfSetups))
SE_RZF_sumSE_DDNN_SI = np.zeros((K,nbrOfSetups))

SE_MR_PF_DDNN_SI = np.zeros((K,nbrOfSetups))
SE_RZF_PF_DDNN_SI = np.zeros((K,nbrOfSetups))

SE_MR_sumSE_CDNN3 = np.zeros((K,nbrOfSetups))
SE_RZF_sumSE_CDNN3 = np.zeros((K,nbrOfSetups))

SE_MR_PF_CDNN3 = np.zeros((K,nbrOfSetups))
SE_RZF_PF_CDNN3 = np.zeros((K,nbrOfSetups))

SE_MR_sumSE_DDNN_without = np.zeros((K,nbrOfSetups))
SE_RZF_sumSE_DDNN_without = np.zeros((K,nbrOfSetups))

SE_MR_PF_DDNN_without = np.zeros((K,nbrOfSetups))
SE_RZF_PF_DDNN_without = np.zeros((K,nbrOfSetups))

############################################

#Prepare array for pilot indices of K UEs for all setups
pilotIndex = np.zeros((K))

#Prepare arrays to save square roots of the power coefficients

#Sum-SE maximization, WMMSE

#ADMM implementation
mu_MR_WMMSE_ADMM = np.zeros((K,L,nbrOfSetups))
mu_RZF_WMMSE_ADMM = np.zeros((K,L,nbrOfSetups))


#Prepare arrays for run time

#Sum-SE maximization, WMMSE

#ADMM implementation
stop_MR_WMMSE_ADMM = np.zeros((nbrOfSetups))
stop_RZF_WMMSE_ADMM = np.zeros((nbrOfSetups))

############################################################
#PF maximization, WMMSE
#All arrays initializations

#ADMM implementation
mu_MR_WMMSE_PF_ADMM = np.zeros((K,L,nbrOfSetups))
mu_RZF_WMMSE_PF_ADMM = np.zeros((K,L,nbrOfSetups))

stop_MR_WMMSE_PF_ADMM = np.zeros((nbrOfSetups))
stop_RZF_WMMSE_PF_ADMM = np.zeros((nbrOfSetups))

SE_MR_WMMSE_PF_ADMM = np.zeros((K,nbrOfSetups))
SE_RZF_WMMSE_PF_ADMM = np.zeros((K,nbrOfSetups))

# Datasets initializations for prediction
dataset_a_MR = np.zeros((L,K,nbrOfSetups))
dataset_a_RZF = np.zeros((L,K,nbrOfSetups))
dataset_UEpositions = np.zeros((K,nbrOfSetups), dtype = 'complex')
dataset_betas = np.zeros((K,L,nbrOfSetups))
dataset_angletoUE = np.zeros((K,L,nbrOfSetups))
dataset_B_MR = np.zeros((L,L,K,K,nbrOfSetups), dtype = 'complex')
dataset_B_RZF = np.zeros((L,L,K,K,nbrOfSetups), dtype = 'complex')
############################################################

#Get AP locations and keep them fixed for all the setups
APpositions = np.load(filename + 'APpositions.npy')
APXpositions = APpositions.real
APYpositions = APpositions.imag

#Go through each random setup
for n in range(0, nbrOfSetups):
    #Output simulation progress
    print(n, 'setups out of', nbrOfSetups)
    
    UEpositions = np.zeros((K,1), dtype = 'complex')
    distances = np.zeros((K,L))
    
    #Prepare to store normalized spatial correlation matrices
    R = np.zeros((M,M,K,L), dtype = 'complex')
    
    #Prepare to store average channel gain numbers (in dB)
    channelGaindB = np.zeros((K,L))
    #Generate random UE locations together
    posXY = np.random.uniform(
            low = 0,
            high = squareLength,
            size = (K,2))
    UEXpositions = posXY[:,0:1]
    UEYpositions = posXY[:,1:2]
    UEpositions = UEXpositions + 1j*UEYpositions
    start = time.perf_counter()
    angletoUE = np.zeros((K,L))    
      
    for k in range(0,K):
        Xdist = np.matlib.repmat(UEXpositions[k,0], L, 1) - APXpositions
        Xdistabs = np.abs(Xdist)
        temp = np.asarray(Xdistabs > squareLength/2).nonzero()[0]
        Xdist[temp,0] = (squareLength - Xdistabs[temp,0]) * np.sign(-Xdist[temp,0])
        Ydist = np.matlib.repmat(UEYpositions[k,0], L, 1) - APYpositions
        Ydistabs = np.abs(Ydist)
        temp = np.asarray(Ydistabs > squareLength/2).nonzero()[0]
        Ydist[temp,0] = (squareLength - Ydistabs[temp,0]) * np.sign(-Ydist[temp,0])
        distances[k,:] = np.sqrt( distanceVertical**2 + Xdist[:,0]**2 + Ydist[:,0]**2)
        channelGaindB[k,:] = constantTerm - alpha*10*np.log10(distances[k,:])
        
        #Go through all APs
        for j in range(0,L):
            #Compute nominal angle between the new UE k and AP l
            angletoUE[k,j] = np.angle(Xdist[j] + 1j*Ydist[j])
            
            import functionRlocalscattering
                    
            R[:,:,k,j] = functionRlocalscattering.R(M,angletoUE[k,j],ASDdeg)
           
    
    end = time.perf_counter() - start
    print('\n Time: ', end)
    #Generate random perturbations (shadowing) truncated at 3 dB
    # for k1 in range(0,K):
    #     perturbation = sigma_sf*np.random.randn(1,L)
    #     bool1 = np.logical_or(perturbation > 3, perturbation < -3)
    #     while np.sum(bool1) != 0:
    #         perturbation[bool1] = sigma_sf*np.random.randn(1,np.sum(bool1)).reshape(np.sum(bool1))
    #         bool1 = np.logical_or(perturbation > 3, perturbation < -3)
            
    #     channelGainPerturbed = channelGaindB[k1,:] + perturbation
    #     channelGaindB[k1,:] = channelGainPerturbed
    
    channelGainOverNoise = channelGaindB - noiseVariancedBm
    H = np.zeros((M,nbrOfRealizations,K,L), dtype = 'complex')
    CH = np.sqrt(0.5)*( np.random.randn(M,nbrOfRealizations,K,L)+1j*np.random.randn(M,nbrOfRealizations,K,L) )
    betas = np.zeros((K,L))
    CorrR = np.zeros((M,M,K,L), dtype = 'complex')
    
    for j2 in range(0,L):
        for k2 in range(0,K):
            betas[k2,j2] = (10**(channelGainOverNoise[k2,j2]/10))
            CorrR[:,:,k2,j2] = betas[k2,j2] * R[:,:,k2,j2]
            Rsqrt = sl.sqrtm(CorrR[:,:,k2,j2])
            H[:,:,k2,j2] = np.matmul(Rsqrt,CH[:,:,k2,j2])
            
    #Perform channel estimation
    #Pilot assignment
    pilotIndex = pilot_assignment.assign_pilots(K, tau_p, betas)
    #For random pilot assignment
    #pilotIndex = np.mod(np.random.permutation(K), tau_p)
    
    #Generate realizations of normalized noise
    Np = np.sqrt(0.5)*(np.random.randn(M,nbrOfRealizations,L,tau_p) + 1j*np.random.randn(M,nbrOfRealizations,L,tau_p))
    
    #Prepare to store results
    Hhat = np.zeros((M,nbrOfRealizations,K,L), dtype = 'complex')
    Hhat_MMSE_MeanSquare = np.zeros((K,L), dtype = 'float')
    
    
    #Go through all APs
    for l in range(0, L):
        
        for t in range(0, tau_p):
            #Compute processed pilot signal for all UEs that use pilot t
            yp = np.sqrt(p*tau_p)* np.sum( H[:,:,t==pilotIndex,l], 2 ) + Np[:,:,l,t]
            
            #Compute the matrix that is inverted in the MMSE estimator
            PsiInv = (p*tau_p* np.sum( CorrR[:,:,t==pilotIndex,l],2 ) + eyeM)
            
            #Go through all UEs that use pilot t
            for k in  np.argwhere(t==pilotIndex):
                RPsi = np.matmul(CorrR[:,:,np.int(k),l],np.linalg.inv(PsiInv))
                Hhat[:,:,np.int(k),l] = np.sqrt(p*tau_p)*np.matmul( RPsi, yp )
                Hhat_MMSE_MeanSquare[np.int(k),l] =  (p * tau_p/ M) * np.real(np.trace (  np.matmul( RPsi, CorrR[:,:,np.int(k),l] ) )  )
                
              
    w_MR = np.zeros((M,K,L), dtype = 'complex')
    w_RZF = np.zeros((M,K,L), dtype = 'complex')
    
    # a_MR[:,k] is a_k in the paper
    a_MR = np.zeros((L,K), dtype = 'complex')
    a_RZF = np.zeros((L,K), dtype = 'complex')
    
    # B_MR[:,:,k,i] is B_{ki} in the paper
    B_MR = np.zeros((L,L,K,K), dtype = 'complex')
    B_RZF = np.zeros((L,L,K,K), dtype = 'complex') 
    
    interf_MR = np.zeros((K,K,L), dtype = 'complex')
    interf_RZF = np.zeros((K,K,L), dtype = 'complex')
    
    interf2_MR = np.zeros((K,K,L), dtype = 'float')
    interf2_RZF = np.zeros((K,K,L), dtype = 'float')

    
    for n1 in range(0,nbrOfRealizations):
        for j3 in range(0,L):
            V_MR = Hhat[:,n1,:,j3]
            V_RZF = np.matmul(np.linalg.inv(p*np.matmul(V_MR, np.conj(V_MR).T) + eyeM), V_MR) 
            w_MR[:,:,j3] = V_MR/np.linalg.norm(V_MR,axis=0)
            w_RZF[:,:,j3] = V_RZF/np.linalg.norm(V_RZF, axis=0)
           
        
        for j4 in range(0,L):
            for k4 in range (0,K): 
                a_MR[j4,k4] = a_MR[j4,k4] + np.matmul( np.conj(H[:,n1,k4,j4]), w_MR[:,k4,j4] )/nbrOfRealizations
                a_RZF[j4,k4] = a_RZF[j4,k4] + np.matmul( np.conj(H[:,n1,k4,j4]), w_RZF[:,k4,j4] )/nbrOfRealizations
                
                for i4 in range(0,K):
                    interf_MR[k4,i4,j4] = interf_MR[k4,i4,j4] + np.matmul( np.conj(H[:,n1,k4,j4]), w_MR[:,i4,j4] )/nbrOfRealizations
                    interf_RZF[k4,i4,j4] = interf_RZF[k4,i4,j4] + np.matmul( np.conj(H[:,n1,k4,j4]), w_RZF[:,i4,j4] )/nbrOfRealizations
                    interf2_MR[k4,i4,j4] = interf2_MR[k4,i4,j4] + np.power(  np.abs(  np.matmul( np.conj(H[:,n1,k4,j4]), w_MR[:,i4,j4]) ), 2 )/nbrOfRealizations
                    interf2_RZF[k4,i4,j4] = interf2_RZF[k4,i4,j4] + np.power(  np.abs(  np.matmul( np.conj(H[:,n1,k4,j4]), w_RZF[:,i4,j4]) ), 2 )/nbrOfRealizations
    
        
    for k5 in range(0,K):
        for i5 in range(0,K):
            B_MR[:,:,k5,i5] = np.matmul(interf_MR[k5,i5,:].reshape(L,1), np.conj(interf_MR[k5,i5,:].reshape(1,L))  )
            B_RZF[:,:,k5,i5] = np.matmul(interf_RZF[k5,i5,:].reshape(L,1), np.conj(interf_RZF[k5,i5,:].reshape(1,L))  )
    
     
    for j5 in range(0,L):
        rho_Giovanni19[:,j5,n] = Pmax*(np.sqrt(Hhat_MMSE_MeanSquare[:,j5])) / (np.sum(np.sqrt(Hhat_MMSE_MeanSquare[:,j5])))      

        B_MR[j5,j5,:,:] = interf2_MR[:,:,j5]
        B_RZF[j5,j5,:,:] = interf2_RZF[:,:,j5]
        
    a_MR = np.abs(a_MR)
    a_RZF = np.abs(a_RZF)
    B_MR = np.real(B_MR)
    B_RZF = np.real(B_RZF)
    
    
    dataset_a_MR[:,:,n] = a_MR
    dataset_a_RZF[:,:,n] = a_RZF
    dataset_B_MR[:,:,:,:,n] = B_MR
    dataset_B_RZF[:,:,:,:,n] = B_RZF
    dataset_UEpositions[:,n] = UEpositions[:,0]
    dataset_betas[:,:,n] = betas
    dataset_angletoUE[:,:,n] = angletoUE
    
    import SpectralEfficiencyDownlink
    SE_MR_equal[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,gammaEqual,Pmax)
    SE_RZF_equal[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,gammaEqual,Pmax)

    SE_MR_Giovanni19[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,np.sqrt(rho_Giovanni19[:,:,n]),Pmax)
    SE_RZF_Giovanni19[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,np.sqrt(rho_Giovanni19[:,:,n]),Pmax)
    

    # Sum-SE ADMM
    import WMMSE_ADMM_Func
    print('WMMSE ADMM\n')
    start_MR_WMMSE = time.perf_counter()
    mu_MR_WMMSE_ADMM[:,:,n] = WMMSE_ADMM_Func.WMMSE_ADMM_timing(L, K, Pmax, a_MR, B_MR)
    stop_MR_WMMSE_ADMM[n] = time.perf_counter() - start_MR_WMMSE
    
    SE_MR_WMMSE_ADMM[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,mu_MR_WMMSE_ADMM[:,:,n],Pmax)
    
    start_RZF_WMMSE = time.perf_counter()
    mu_RZF_WMMSE_ADMM[:,:,n] = WMMSE_ADMM_Func.WMMSE_ADMM_timing(L, K, Pmax, a_RZF, B_RZF)
    stop_RZF_WMMSE_ADMM[n] = time.perf_counter() - start_RZF_WMMSE

    SE_RZF_WMMSE_ADMM[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,mu_RZF_WMMSE_ADMM[:,:,n],Pmax)
    
    
    # Proportional Fairness ADMM
    import WMMSE_PF_ADMM
    print('WMMSE PF ADMM\n')
    start_MR_WMMSE = time.perf_counter()
    mu_MR_WMMSE_PF_ADMM[:,:,n] = WMMSE_PF_ADMM.WMMSE_ADMM_timing(L, K, Pmax, a_MR, B_MR)
    stop_MR_WMMSE_PF_ADMM[n] = time.perf_counter() - start_MR_WMMSE
    
    SE_MR_WMMSE_PF_ADMM[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_MR,B_MR,prelogFactor,mu_MR_WMMSE_PF_ADMM[:,:,n], Pmax)
    
    start_RZF_WMMSE = time.perf_counter()
    mu_RZF_WMMSE_PF_ADMM[:,:,n] = WMMSE_PF_ADMM.WMMSE_ADMM_timing(L, K, Pmax, a_RZF, B_RZF)
    stop_RZF_WMMSE_PF_ADMM[n] = time.perf_counter() - start_RZF_WMMSE

    SE_RZF_WMMSE_PF_ADMM[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(a_RZF,B_RZF,prelogFactor,mu_RZF_WMMSE_PF_ADMM[:,:,n], Pmax)
    

dataset_B_MR = np.real(dataset_B_MR)
dataset_B_RZF = np.real(dataset_B_RZF)

## Sum-SE DDNN
modelname_MR_sumSE = 'Trained_models_for_mu_MR_sumSE_DDNN_for_AP'
modelname_RZF_sumSE = 'Trained_models_for_mu_RZF_sumSE_DDNN_for_AP'
import Predictions_DDNN
muMR_DDNN_sumSE, MR_sumSE_DDNN_time = Predictions_DDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_MR_sumSE)
muRZF_DDNN_sumSE, RZF_sumSE_DDNN_time = Predictions_DDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_RZF_sumSE)

muMR_DDNN_sumSE_scaling = muMR_DDNN_sumSE[K:,:,:]
muRZF_DDNN_sumSE_scaling = muRZF_DDNN_sumSE[K:,:,:]
muMR_DDNN_sumSE_scaling[muMR_DDNN_sumSE_scaling > 1] = 1
muRZF_DDNN_sumSE_scaling[muRZF_DDNN_sumSE_scaling > 1] = 1
start_time = time.perf_counter()
muMR_DDNN_sumSE_scaling = muMR_DDNN_sumSE_scaling * np.sqrt(Pmax)
muRZF_DDNN_sumSE_scaling = muRZF_DDNN_sumSE_scaling * np.sqrt(Pmax)

for n in range(0,nbrOfSetups): 
    muMR_DDNN_sumSE[0:K,:,n] = muMR_DDNN_sumSE[0:K,:,n] * np.repeat(muMR_DDNN_sumSE_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muMR_DDNN_sumSE[0:K,:,n], axis=0) )
    muRZF_DDNN_sumSE[0:K,:,n] = muRZF_DDNN_sumSE[0:K,:,n] * np.repeat(muRZF_DDNN_sumSE_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muRZF_DDNN_sumSE[0:K,:,n], axis=0) )

scaling_time = (time.perf_counter() - start_time)/2

for n in range(0,nbrOfSetups):
    SE_MR_sumSE_DDNN[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_MR[:,:,n], dataset_B_MR[:,:,:,:,n], prelogFactor, muMR_DDNN_sumSE[0:K,:,n], Pmax)
    SE_RZF_sumSE_DDNN[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_RZF[:,:,n], dataset_B_RZF[:,:,:,:,n], prelogFactor, muRZF_DDNN_sumSE[0:K,:,n], Pmax)


## PF DDNN
modelname_MR_PF = 'Trained_models_for_mu_MR_PF_DDNN_for_AP'
modelname_RZF_PF = 'Trained_models_for_mu_RZF_PF_DDNN_for_AP'
muMR_DDNN_PF, MR_PF_DDNN_time = Predictions_DDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_MR_PF)
muRZF_DDNN_PF, RZF_PF_DDNN_time = Predictions_DDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_RZF_PF)

muMR_DDNN_PF_scaling = muMR_DDNN_PF[K:,:,:]
muRZF_DDNN_PF_scaling = muRZF_DDNN_PF[K:,:,:]
muMR_DDNN_PF_scaling[muMR_DDNN_PF_scaling > 1] = 1
muRZF_DDNN_PF_scaling[muRZF_DDNN_PF_scaling > 1] = 1
muMR_DDNN_PF_scaling = muMR_DDNN_PF_scaling * np.sqrt(Pmax)
muRZF_DDNN_PF_scaling = muRZF_DDNN_PF_scaling * np.sqrt(Pmax)

for n in range(0,nbrOfSetups): 
    muMR_DDNN_PF[0:K,:,n] = muMR_DDNN_PF[0:K,:,n] * np.repeat(muMR_DDNN_PF_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muMR_DDNN_PF[0:K,:,n], axis=0) )
    muRZF_DDNN_PF[0:K,:,n] = muRZF_DDNN_PF[0:K,:,n] * np.repeat(muRZF_DDNN_PF_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muRZF_DDNN_PF[0:K,:,n], axis=0) )


for n in range(0,nbrOfSetups):
    SE_MR_PF_DDNN[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_MR[:,:,n], dataset_B_MR[:,:,:,:,n], prelogFactor, muMR_DDNN_PF[0:K,:,n], Pmax)
    SE_RZF_PF_DDNN[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_RZF[:,:,n], dataset_B_RZF[:,:,:,:,n], prelogFactor, muRZF_DDNN_PF[0:K,:,n], Pmax)


## Sum-SE DDNN_SI
modelname_MR_sumSE = 'Trained_models_for_mu_MR_sumSE_DDNN_for_AP'
modelname_RZF_sumSE = 'Trained_models_for_mu_RZF_sumSE_DDNN_for_AP'
import Predictions_DDNN
muMR_DDNN_sumSE, MR_sumSE_DDNN_SI_time = Predictions_DDNN.pred_func_extrainput(dataset_betas, Pmax, nbrOfSetups, modelname_MR_sumSE)
muRZF_DDNN_sumSE, RZF_sumSE_DDNN_SI_time = Predictions_DDNN.pred_func_extrainput(dataset_betas, Pmax, nbrOfSetups, modelname_RZF_sumSE)

muMR_DDNN_sumSE_scaling = muMR_DDNN_sumSE[K:,:,:]
muRZF_DDNN_sumSE_scaling = muRZF_DDNN_sumSE[K:,:,:]
muMR_DDNN_sumSE_scaling[muMR_DDNN_sumSE_scaling > 1] = 1
muRZF_DDNN_sumSE_scaling[muRZF_DDNN_sumSE_scaling > 1] = 1
muMR_DDNN_sumSE_scaling = muMR_DDNN_sumSE_scaling * np.sqrt(Pmax)
muRZF_DDNN_sumSE_scaling = muRZF_DDNN_sumSE_scaling * np.sqrt(Pmax)

for n in range(0,nbrOfSetups): 
    muMR_DDNN_sumSE[0:K,:,n] = muMR_DDNN_sumSE[0:K,:,n] * np.repeat(muMR_DDNN_sumSE_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muMR_DDNN_sumSE[0:K,:,n], axis=0) )
    muRZF_DDNN_sumSE[0:K,:,n] = muRZF_DDNN_sumSE[0:K,:,n] * np.repeat(muRZF_DDNN_sumSE_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muRZF_DDNN_sumSE[0:K,:,n], axis=0) )


for n in range(0,nbrOfSetups):
    SE_MR_sumSE_DDNN_SI[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_MR[:,:,n], dataset_B_MR[:,:,:,:,n], prelogFactor, muMR_DDNN_sumSE[0:K,:,n], Pmax)
    SE_RZF_sumSE_DDNN_SI[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_RZF[:,:,n], dataset_B_RZF[:,:,:,:,n], prelogFactor, muRZF_DDNN_sumSE[0:K,:,n], Pmax)


## PF DDNN_SI
modelname_MR_PF = 'Trained_models_for_mu_MR_PF_DDNN_for_AP'
modelname_RZF_PF = 'Trained_models_for_mu_RZF_PF_DDNN_for_AP'
muMR_DDNN_PF, MR_PF_DDNN_SI_time = Predictions_DDNN.pred_func_extrainput(dataset_betas, Pmax, nbrOfSetups, modelname_MR_PF)
muRZF_DDNN_PF, RZF_PF_DDNN_SI_time = Predictions_DDNN.pred_func_extrainput(dataset_betas, Pmax, nbrOfSetups, modelname_RZF_PF)

muMR_DDNN_PF_scaling = muMR_DDNN_PF[K:,:,:]
muRZF_DDNN_PF_scaling = muRZF_DDNN_PF[K:,:,:]
muMR_DDNN_PF_scaling[muMR_DDNN_PF_scaling > 1] = 1
muRZF_DDNN_PF_scaling[muRZF_DDNN_PF_scaling > 1] = 1
muMR_DDNN_PF_scaling = muMR_DDNN_PF_scaling * np.sqrt(Pmax)
muRZF_DDNN_PF_scaling = muRZF_DDNN_PF_scaling * np.sqrt(Pmax)

for n in range(0,nbrOfSetups): 
    muMR_DDNN_PF[0:K,:,n] = muMR_DDNN_PF[0:K,:,n] * np.repeat(muMR_DDNN_PF_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muMR_DDNN_PF[0:K,:,n], axis=0) )
    muRZF_DDNN_PF[0:K,:,n] = muRZF_DDNN_PF[0:K,:,n] * np.repeat(muRZF_DDNN_PF_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muRZF_DDNN_PF[0:K,:,n], axis=0) )


for n in range(0,nbrOfSetups):
    SE_MR_PF_DDNN_SI[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_MR[:,:,n], dataset_B_MR[:,:,:,:,n], prelogFactor, muMR_DDNN_PF[0:K,:,n], Pmax)
    SE_RZF_PF_DDNN_SI[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_RZF[:,:,n], dataset_B_RZF[:,:,:,:,n], prelogFactor, muRZF_DDNN_PF[0:K,:,n], Pmax)


cluster_size = 3
## Sum-SE CDNN
modelname_MR_sumSE = 'Trained_models_for_mu_MR_sumSE_CDNN_for_AP'
modelname_RZF_sumSE = 'Trained_models_for_mu_RZF_sumSE_CDNN_for_AP'
import Predictions_CDNN
muMR_CDNN_sumSE, MR_sumSE_CDNN_time = Predictions_CDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_MR_sumSE, cluster_size)
muRZF_CDNN_sumSE, RZF_sumSE_CDNN_time = Predictions_CDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_RZF_sumSE, cluster_size)

muMR_CDNN_sumSE_scaling = muMR_CDNN_sumSE[K:,:,:]
muRZF_CDNN_sumSE_scaling = muRZF_CDNN_sumSE[K:,:,:]
muMR_CDNN_sumSE_scaling[muMR_CDNN_sumSE_scaling > 1] = 1
muRZF_CDNN_sumSE_scaling[muRZF_CDNN_sumSE_scaling > 1] = 1
muMR_CDNN_sumSE_scaling = muMR_CDNN_sumSE_scaling * np.sqrt(Pmax)
muRZF_CDNN_sumSE_scaling = muRZF_CDNN_sumSE_scaling * np.sqrt(Pmax)

for n in range(0,nbrOfSetups): 
    muMR_CDNN_sumSE[0:K,:,n] = muMR_CDNN_sumSE[0:K,:,n] * np.repeat(muMR_CDNN_sumSE_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muMR_CDNN_sumSE[0:K,:,n], axis=0) )
    muRZF_CDNN_sumSE[0:K,:,n] = muRZF_CDNN_sumSE[0:K,:,n] * np.repeat(muRZF_CDNN_sumSE_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muRZF_CDNN_sumSE[0:K,:,n], axis=0) )


for n in range(0,nbrOfSetups):
    SE_MR_sumSE_CDNN3[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_MR[:,:,n], dataset_B_MR[:,:,:,:,n], prelogFactor, muMR_CDNN_sumSE[0:K,:,n], Pmax)
    SE_RZF_sumSE_CDNN3[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_RZF[:,:,n], dataset_B_RZF[:,:,:,:,n], prelogFactor, muRZF_CDNN_sumSE[0:K,:,n], Pmax)


## PF CDNN
modelname_MR_PF = 'Trained_models_for_mu_MR_PF_CDNN_for_AP'
modelname_RZF_PF = 'Trained_models_for_mu_RZF_PF_CDNN_for_AP'
muMR_CDNN_PF, MR_PF_CDNN_time = Predictions_CDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_MR_PF, cluster_size)
muRZF_CDNN_PF, RZF_PF_CDNN_time = Predictions_CDNN.pred_func(dataset_betas, Pmax, nbrOfSetups, modelname_RZF_PF, cluster_size)

muMR_CDNN_PF_scaling = muMR_CDNN_PF[K:,:,:]
muRZF_CDNN_PF_scaling = muRZF_CDNN_PF[K:,:,:]
muMR_CDNN_PF_scaling[muMR_CDNN_PF_scaling > 1] = 1
muRZF_CDNN_PF_scaling[muRZF_CDNN_PF_scaling > 1] = 1
muMR_CDNN_PF_scaling = muMR_CDNN_PF_scaling * np.sqrt(Pmax)
muRZF_CDNN_PF_scaling = muRZF_CDNN_PF_scaling * np.sqrt(Pmax)

for n in range(0,nbrOfSetups): 
    muMR_CDNN_PF[0:K,:,n] = muMR_CDNN_PF[0:K,:,n] * np.repeat(muMR_CDNN_PF_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muMR_CDNN_PF[0:K,:,n], axis=0) )
    muRZF_CDNN_PF[0:K,:,n] = muRZF_CDNN_PF[0:K,:,n] * np.repeat(muRZF_CDNN_PF_scaling[:,:,n], K, axis=0) / np.max( np.linalg.norm(muRZF_CDNN_PF[0:K,:,n], axis=0) )


for n in range(0,nbrOfSetups):
    SE_MR_PF_CDNN3[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_MR[:,:,n], dataset_B_MR[:,:,:,:,n], prelogFactor, muMR_CDNN_PF[0:K,:,n], Pmax)
    SE_RZF_PF_CDNN3[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_RZF[:,:,n], dataset_B_RZF[:,:,:,:,n], prelogFactor, muRZF_CDNN_PF[0:K,:,n], Pmax)


## Sum-SE DDNN without heuristic and total power adjustment
modelname_MR_sumSE = 'Trained_models_for_mu_MR_sumSE_DDNN_for_AP'
modelname_RZF_sumSE = 'Trained_models_for_mu_RZF_sumSE_DDNN_for_AP'
import Predictions_DDNN
muMR_DDNN_sumSE_without, MR_sumSE_DDNN_time_without = Predictions_DDNN.pred_func_without(dataset_betas, Pmax, nbrOfSetups, modelname_MR_sumSE)
muRZF_DDNN_sumSE_without, RZF_sumSE_DDNN_time_without = Predictions_DDNN.pred_func_without(dataset_betas, Pmax, nbrOfSetups, modelname_RZF_sumSE)

start_time = time.perf_counter()
muMR_DDNN_sumSE_scaling = np.sqrt(Pmax)
muRZF_DDNN_sumSE_scaling = np.sqrt(Pmax)

for n in range(0,nbrOfSetups): 
    muMR_DDNN_sumSE_without[0:K,:,n] = muMR_DDNN_sumSE_without[0:K,:,n] * muMR_DDNN_sumSE_scaling / np.max( np.linalg.norm(muMR_DDNN_sumSE_without[0:K,:,n], axis=0) )
    muRZF_DDNN_sumSE_without[0:K,:,n] = muRZF_DDNN_sumSE_without[0:K,:,n] * muRZF_DDNN_sumSE_scaling / np.max( np.linalg.norm(muRZF_DDNN_sumSE_without[0:K,:,n], axis=0) )

scaling_time = (time.perf_counter() - start_time)/2

for n in range(0,nbrOfSetups):
    SE_MR_sumSE_DDNN_without[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_MR[:,:,n], dataset_B_MR[:,:,:,:,n], prelogFactor, muMR_DDNN_sumSE_without[0:K,:,n], Pmax)
    SE_RZF_sumSE_DDNN_without[:,n] = SpectralEfficiencyDownlink.Calculate_SINR_and_SE_DL(dataset_a_RZF[:,:,n], dataset_B_RZF[:,:,:,:,n], prelogFactor, muRZF_DDNN_sumSE_without[0:K,:,n], Pmax)

#Sort the SE values for CDF plots
import CalculateSortedSE
sorted_SE_MR_Equal = CalculateSortedSE.sorted_SE(SE_MR_equal)
sorted_SE_RZF_Equal = CalculateSortedSE.sorted_SE(SE_RZF_equal)

sorted_SE_MR_Giovanni19 = CalculateSortedSE.sorted_SE(SE_MR_Giovanni19)
sorted_SE_RZF_Giovanni19 = CalculateSortedSE.sorted_SE(SE_RZF_Giovanni19)

sorted_SE_MR_WMMSE_ADMM = CalculateSortedSE.sorted_SE(SE_MR_WMMSE_ADMM)
sorted_SE_RZF_WMMSE_ADMM = CalculateSortedSE.sorted_SE(SE_RZF_WMMSE_ADMM)

sorted_SE_MR_WMMSE_PF_ADMM = CalculateSortedSE.sorted_SE(SE_MR_WMMSE_PF_ADMM)
sorted_SE_RZF_WMMSE_PF_ADMM = CalculateSortedSE.sorted_SE(SE_RZF_WMMSE_PF_ADMM)

sorted_SE_MR_sumSE_DDNN = CalculateSortedSE.sorted_SE(SE_MR_sumSE_DDNN)
sorted_SE_RZF_sumSE_DDNN = CalculateSortedSE.sorted_SE(SE_RZF_sumSE_DDNN)

sorted_SE_MR_PF_DDNN = CalculateSortedSE.sorted_SE(SE_MR_PF_DDNN)
sorted_SE_RZF_PF_DDNN = CalculateSortedSE.sorted_SE(SE_RZF_PF_DDNN)

sorted_SE_MR_sumSE_DDNN_SI = CalculateSortedSE.sorted_SE(SE_MR_sumSE_DDNN_SI)
sorted_SE_RZF_sumSE_DDNN_SI = CalculateSortedSE.sorted_SE(SE_RZF_sumSE_DDNN_SI)

sorted_SE_MR_PF_DDNN_SI = CalculateSortedSE.sorted_SE(SE_MR_PF_DDNN_SI)
sorted_SE_RZF_PF_DDNN_SI = CalculateSortedSE.sorted_SE(SE_RZF_PF_DDNN_SI)

sorted_SE_MR_sumSE_CDNN3 = CalculateSortedSE.sorted_SE(SE_MR_sumSE_CDNN3)
sorted_SE_RZF_sumSE_CDNN3 = CalculateSortedSE.sorted_SE(SE_RZF_sumSE_CDNN3)

sorted_SE_MR_PF_CDNN3 = CalculateSortedSE.sorted_SE(SE_MR_PF_CDNN3)
sorted_SE_RZF_PF_CDNN3 = CalculateSortedSE.sorted_SE(SE_RZF_PF_CDNN3)


sorted_SE_MR_sumSE_DDNN_without = CalculateSortedSE.sorted_SE(SE_MR_sumSE_DDNN_without)
sorted_SE_RZF_sumSE_DDNN_without = CalculateSortedSE.sorted_SE(SE_RZF_sumSE_DDNN_without)


# Calculations for sum-rate plots
sum_SE_MR_Equal = np.sum(SE_MR_equal, axis=0)
sum_SE_MR_Equal.sort()
sum_SE_RZF_Equal = np.sum(SE_RZF_equal, axis=0)
sum_SE_RZF_Equal.sort()

sum_SE_MR_Giovanni19 = np.sum(SE_MR_Giovanni19, axis=0)
sum_SE_MR_Giovanni19.sort()
sum_SE_RZF_Giovanni19 = np.sum(SE_RZF_Giovanni19, axis=0)
sum_SE_RZF_Giovanni19.sort()

sum_SE_MR_WMMSE_ADMM = np.sum(SE_MR_WMMSE_ADMM, axis=0)
sum_SE_MR_WMMSE_ADMM.sort()
sum_SE_RZF_WMMSE_ADMM = np.sum(SE_RZF_WMMSE_ADMM, axis=0)
sum_SE_RZF_WMMSE_ADMM.sort()

sum_SE_MR_WMMSE_PF_ADMM = np.sum(SE_MR_WMMSE_PF_ADMM, axis=0)
sum_SE_MR_WMMSE_PF_ADMM.sort()
sum_SE_RZF_WMMSE_PF_ADMM = np.sum(SE_RZF_WMMSE_PF_ADMM, axis=0)
sum_SE_RZF_WMMSE_PF_ADMM.sort()

sum_SE_MR_sumSE_DDNN = np.sum(SE_MR_sumSE_DDNN, axis=0)
sum_SE_MR_sumSE_DDNN.sort()
sum_SE_RZF_sumSE_DDNN = np.sum(SE_RZF_sumSE_DDNN, axis=0)
sum_SE_RZF_sumSE_DDNN.sort()

sum_SE_MR_PF_DDNN = np.sum(SE_MR_PF_DDNN, axis=0)
sum_SE_MR_PF_DDNN.sort()
sum_SE_RZF_PF_DDNN = np.sum(SE_RZF_PF_DDNN, axis=0)
sum_SE_RZF_PF_DDNN.sort()

sum_SE_MR_sumSE_CDNN3 = np.sum(SE_MR_sumSE_CDNN3, axis=0)
sum_SE_MR_sumSE_CDNN3.sort()
sum_SE_RZF_sumSE_CDNN3 = np.sum(SE_RZF_sumSE_CDNN3, axis=0)
sum_SE_RZF_sumSE_CDNN3.sort()

sum_SE_MR_PF_CDNN3 = np.sum(SE_MR_PF_CDNN3, axis=0)
sum_SE_MR_PF_CDNN3.sort()
sum_SE_RZF_PF_CDNN3 = np.sum(SE_RZF_PF_CDNN3, axis=0)
sum_SE_RZF_PF_CDNN3.sort()


sum_SE_MR_sumSE_DDNN_without = np.sum(SE_MR_sumSE_DDNN_without, axis=0)
sum_SE_MR_sumSE_DDNN_without.sort()
sum_SE_RZF_sumSE_DDNN_without = np.sum(SE_RZF_sumSE_DDNN_without, axis=0)
sum_SE_RZF_sumSE_DDNN_without.sort()


Yvals = np.linspace(0,1,K*nbrOfSetups)
Yvals_sum = np.linspace(0,1,nbrOfSetups)

### -------------------------------- Figures -------------------------------------
fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_MR_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)

plt.plot(sorted_SE_MR_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)

plt.plot(sorted_SE_MR_WMMSE_ADMM, Yvals, color='#000000ff',  marker=3, markevery=1000, label='sumSE ADMM', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_sumSE_DDNN, Yvals, color='b',  marker=3, markevery=1000, label='sumSE DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_sumSE_CDNN3, Yvals, color='goldenrod',  marker=3, markevery=1000, label='sumSE CDNN', linewidth=4,markersize=15)

plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_MR_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)

plt.plot(sorted_SE_MR_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)

plt.plot(sorted_SE_MR_WMMSE_PF_ADMM, Yvals, color='r',  marker=3, markevery=1000, label='PF ADMM', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_PF_DDNN, Yvals, color='c',  marker=3, markevery=1000, label='PF DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_PF_CDNN3, Yvals, color='brown',  marker=3, markevery=1000, label='PF CDNN', linewidth=4,markersize=15)

plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_RZF_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sorted_SE_RZF_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sorted_SE_RZF_WMMSE_ADMM, Yvals, color='#000000ff',  marker=3, markevery=1000, label='sumSE ADMM', linewidth=4,markersize=15) 

plt.plot(sorted_SE_RZF_sumSE_DDNN, Yvals, color='b',  marker=3, markevery=1000, label='sumSE DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_RZF_sumSE_CDNN3, Yvals, color='goldenrod',  marker=3, markevery=1000, label='sumSE CDNN', linewidth=4,markersize=15)
                    
plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_RZF_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sorted_SE_RZF_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sorted_SE_RZF_WMMSE_PF_ADMM, Yvals, color='r',  marker=3, markevery=1000, label='PF ADMM', linewidth=4,markersize=15)

plt.plot(sorted_SE_RZF_PF_DDNN, Yvals, color='c',  marker=3, markevery=1000, label='PF DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_RZF_PF_CDNN3, Yvals, color='brown',  marker=3, markevery=1000, label='PF CDNN', linewidth=4,markersize=15)
                    
plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)



fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_MR_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)

plt.plot(sorted_SE_MR_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)

plt.plot(sorted_SE_MR_WMMSE_ADMM, Yvals, color='#000000ff',  marker=3, markevery=1000, label='sumSE ADMM', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_sumSE_DDNN, Yvals, color='b',  marker=3, markevery=1000, label='sumSE DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_sumSE_DDNN_SI, Yvals, color='darkgoldenrod',  marker=3, markevery=1000, label='sumSE DDNN-SI', linewidth=4,markersize=15)

plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_MR_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)

plt.plot(sorted_SE_MR_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)

plt.plot(sorted_SE_MR_WMMSE_PF_ADMM, Yvals, color='r',  marker=3, markevery=1000, label='PF ADMM', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_PF_DDNN, Yvals, color='c',  marker=3, markevery=1000, label='PF DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_PF_DDNN_SI, Yvals, color='saddlebrown',  marker=3, markevery=1000, label='PF DDNN-SI', linewidth=4,markersize=15)

plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_RZF_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sorted_SE_RZF_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sorted_SE_RZF_WMMSE_ADMM, Yvals, color='#000000ff',  marker=3, markevery=1000, label='sumSE ADMM', linewidth=4,markersize=15) 

plt.plot(sorted_SE_RZF_sumSE_DDNN, Yvals, color='b',  marker=3, markevery=1000, label='sumSE DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_RZF_sumSE_DDNN_SI, Yvals, color='darkgoldenrod',  marker=3, markevery=1000, label='sumSE DDNN-SI', linewidth=4,markersize=15)
                    
plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_RZF_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sorted_SE_RZF_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sorted_SE_RZF_WMMSE_PF_ADMM, Yvals, color='r',  marker=3, markevery=1000, label='PF ADMM', linewidth=4,markersize=15)

plt.plot(sorted_SE_RZF_PF_DDNN, Yvals, color='c',  marker=3, markevery=1000, label='PF DDNN', linewidth=4,markersize=15)

plt.plot(sorted_SE_RZF_PF_DDNN_SI, Yvals, color='saddlebrown',  marker=3, markevery=1000, label='PF DDNN-SI', linewidth=4,markersize=15)
                    
plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_MR_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)

plt.plot(sorted_SE_MR_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)

plt.plot(sorted_SE_MR_WMMSE_ADMM, Yvals, color='#000000ff',  marker=3, markevery=1000, label='sumSE ADMM', linewidth=4,markersize=15)

plt.plot(sorted_SE_MR_WMMSE_PF_ADMM, Yvals, color='r',  marker=3, markevery=1000, label='PF ADMM', linewidth=4,markersize=15)

plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sorted_SE_RZF_Equal,  Yvals, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sorted_SE_RZF_Giovanni19, Yvals, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sorted_SE_RZF_WMMSE_ADMM, Yvals, color='#000000ff',  marker=3, markevery=1000, label='sumSE ADMM', linewidth=4,markersize=15) 

plt.plot(sorted_SE_RZF_WMMSE_PF_ADMM, Yvals, color='r',  marker=3, markevery=1000, label='PF ADMM', linewidth=4,markersize=15)
                    
plt.legend(loc="lower right",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('SE per UE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sum_SE_MR_Equal,  Yvals_sum, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sum_SE_MR_Giovanni19, Yvals_sum, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sum_SE_MR_WMMSE_ADMM, Yvals_sum, color='#000000ff',  marker=3, markevery=50, label='sumSE ADMM', linewidth=4,markersize=15) 

plt.plot(sum_SE_MR_sumSE_DDNN, Yvals_sum, color='b',  marker=3, markevery=50, label='sumSE DDNN', linewidth=4,markersize=15)

plt.plot(sum_SE_MR_sumSE_DDNN_without, Yvals_sum, color='m',  marker=3, markevery=50, label='sumSE DDNN without', linewidth=4,markersize=15)
                    
plt.legend(loc="upper left",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
# plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('Total SE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sum_SE_RZF_Equal,  Yvals_sum, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sum_SE_RZF_Giovanni19, Yvals_sum, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sum_SE_RZF_WMMSE_ADMM, Yvals_sum, color='#000000ff',  marker=3, markevery=50, label='sumSE ADMM', linewidth=4,markersize=15) 

plt.plot(sum_SE_RZF_sumSE_DDNN, Yvals_sum, color='b',  marker=3, markevery=50, label='sumSE DDNN', linewidth=4,markersize=15)

plt.plot(sum_SE_RZF_sumSE_CDNN3, Yvals_sum, color='goldenrod',  marker=3, markevery=50, label='sumSE CDNN', linewidth=4,markersize=15)
                    
plt.legend(loc="upper left",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
# plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('Total SE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)


fig = plt.figure(figsize=(16,12))
plt.plot(sum_SE_RZF_Equal,  Yvals_sum, color='#000000ff', label='Equal power', linewidth=4)  

plt.plot(sum_SE_RZF_Giovanni19, Yvals_sum, '-.', color='#000000ff', label='[12]', linewidth=4)   

plt.plot(sum_SE_RZF_WMMSE_PF_ADMM, Yvals_sum, color='r',  marker=3, markevery=50, label='PF ADMM', linewidth=4,markersize=15)

plt.plot(sum_SE_RZF_PF_DDNN, Yvals_sum, color='c',  marker=3, markevery=50, label='PF DDNN', linewidth=4,markersize=15)

plt.plot(sum_SE_RZF_PF_CDNN3, Yvals_sum, color='brown',  marker=3, markevery=50, label='PF CDNN', linewidth=4,markersize=15)

plt.legend(loc="upper left",prop={'size':32})
plt.grid(True)
plt.rc('xtick',labelsize=36)
plt.rc('ytick',labelsize=36)
# plt.xlim([0.0, 5])
plt.ylim([0, 1])
plt.xlabel('Total SE [bit/s/Hz]', fontsize=34)
plt.ylabel('CDF', fontsize=34)




print('\n Time taken for WMMSE ADMM (MR): ', np.mean(stop_MR_WMMSE_ADMM) )
print('\n Time taken for WMMSE PF ADMM (MR): ', np.mean(stop_MR_WMMSE_PF_ADMM) )

print('\n Time taken for WMMSE ADMM (RZF): ', np.mean(stop_RZF_WMMSE_ADMM) )
print('\n Time taken for WMMSE PF ADMM (RZF): ', np.mean(stop_RZF_WMMSE_PF_ADMM) )

# The comutation time is calculated by averaging over 100 setups
print('\n Time taken for sumSE DDNN (MR): ',  (MR_sumSE_DDNN_time*L+scaling_time)/nbrOfSetups)
print('\n Time taken for sumSE DDNN (RZF): ',  (RZF_sumSE_DDNN_time*L+scaling_time)/nbrOfSetups)
print('\n Time taken for PF DDNN (MR): ',  (MR_PF_DDNN_time*L+scaling_time)/nbrOfSetups)
print('\n Time taken for PF DDNN (RZF): ',  (RZF_PF_DDNN_time*L+scaling_time)/nbrOfSetups)

print('\n Time taken for sumSE DDNN-SI (MR): ',  (MR_sumSE_DDNN_SI_time*L+scaling_time)/nbrOfSetups)
print('\n Time taken for sumSE DDNN-SI (RZF): ',  (RZF_sumSE_DDNN_SI_time*L+scaling_time)/nbrOfSetups)
print('\n Time taken for PF DDNN-SI (MR): ',  (MR_PF_DDNN_SI_time*L+scaling_time)/nbrOfSetups)
print('\n Time taken for PF DDNN-SI (RZF): ',  (RZF_PF_DDNN_SI_time*L+scaling_time)/nbrOfSetups)

print('\n Time taken for sumSE CDNN (MR): ',  (MR_sumSE_CDNN_time*L+scaling_time)/(cluster_size*nbrOfSetups))
print('\n Time taken for sumSE CDNN (RZF): ',  (RZF_sumSE_CDNN_time*L+scaling_time)/(cluster_size*nbrOfSetups))
print('\n Time taken for PF CDNN (MR): ',  (MR_PF_CDNN_time*L+scaling_time)/(cluster_size*nbrOfSetups))
print('\n Time taken for PF CDNN (RZF): ',  (RZF_PF_CDNN_time*L+scaling_time)/(cluster_size*nbrOfSetups))