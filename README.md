Learning-Based Downlink Power Allocation in Cell-Free Massive MIMO Systems
=====================================================================================================

This code package is related to the following scientific article:

Mahmoud Zaher, Özlem Tuğfe Demir, Emil Björnson, Marina Petrova, “[Learning-Based Downlink Power Allocation in Cell-Free Massive MIMO Systems](https://arxiv.org/pdf/2109.03128.pdf),” IEEE Transactions on Wireless Communications, vol. 22, no. 1, pp. 174-188, Jan. 2023.

The package contains a simulation environment, based on Python, that reproduces the numerical results in the article. Please read the "Content of the Package" section below before trying to run the code.

We encourage you to also perform reproducible research!


## Abstract of Article

This paper considers a cell-free massive multiple- input multiple-output (MIMO) system that consists of a large number of geographically distributed access points (APs) serving multiple users via coherent joint transmission. The downlink performance of the system is evaluated, with maximum ratio and regularized zero-forcing precoding, under two optimization objectives for power allocation: sum spectral efficiency (SE) maximization and proportional fairness. We present iterative centralized algorithms for solving these problems. Aiming at a less computationally complex and also distributed scalable solution, we train a deep neural network (DNN) to approximate the same network-wide power allocation. Instead of training our DNN to mimic the actual optimization procedure, we use a heuristic power allocation, based on large-scale fading (LSF) parameters, as the pre-processed input to the DNN. We train the DNN to refine the heuristic scheme, thereby providing higher SE, using only local information at each AP. Another distributed DNN that exploits side information assumed to be available at the central processing unit is designed for improved performance. Further, we develop a clustered DNN model where the LSF parameters of a small number of APs, forming a cluster within a relatively large network, are used to jointly approximate the power coefficients of the cluster.

## Content of the Package

The main_DNN.py script is responsible for generating the figures in the article. It can be used to generate datasets that are utilized to train and test the neural networks (NNs) proposed in the article and solve the WMMSE problems using the related functions in the code package. <b>To generate data, the parts responsible for utilizing the trained models in the main_DNN.py should be commented out and the required variables for the models' input and labelled output saved.</b> The file APLocation_Generation.py is used to generate the AP positions that should be saved and kept fixed afterwards throughout the simulations.

To train the DDNN and DDNN-SI models, the Distributed_FFNN.py can be used after generating the required datasets. The Clustered_FFNN.py is used for the CDNN model. <b>In these two files, the scaler.pkl should be saved once for a given simulation setup (dataset).</b> Please note the names of the datasets and trained NN models while saving in order to use them in the main script and the functions that it utilizes (Predictions_DDNN.py, Predictions_CDNN.py).


## Associated datasets

The datasets are associated with the non-orthogonal pilot assignment case in the article. The input and labelled output for training the models therein, and the computed SE performance for the conventional optimization approaches are provided. All files, except for the 'APpositions' file should be grouped in a folder named 'pAssign_storage' and located in the same folder as the code package.

https://zenodo.org/record/7524622#.Y76Uy3aZOF5


## Acknowledgements

This work was supported by the FFL18-0277 grant from the Swedish Foundation for Strategic Research.

## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.
