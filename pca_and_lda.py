# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:32:51 2024

@author: pilli
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut


def computing_projection_hist(proj_matrix: np.ndarray, dataset: np.ndarray, labels: np.ndarray, name: str) -> np.ndarray:
    '''
    
    Args:
    
        proj_matrix : np.ndarray
                      Projection matrix computed through PCA or LDA
            
        dataset : np.ndarray
                  Input Dataset
        
        labels : np.ndarray
                 Input Labels
        
        name : str
               Graph description

    Returns:
        
        DP : np.ndarray
             Projected Data 

    '''
    DP = proj_matrix.T @ dataset
    D0,D1 = ut.create_mask_binary(DP, labels)

    print(f'D0 (Fake) mean: {D0.mean()} - D1 (True) mean: {D1.mean()}')
   
    plt.figure()
    
    '''
        If Density=True is used to display a histogram, the total area under the histogram will 
        be normalized to 1, so that the histogram represents a probability density,
        where the height of the histogram bins reflects the estimated probability density
        for each data interval.
        
    '''
    
    plt.hist(D0[0, :] if len(DP.shape) > 1 else D0 , bins = 10, density= True,  alpha=0.4, edgecolor='0.4', label='Fake Class')
    plt.hist(D1[0, :] if len(DP.shape) > 1 else D1, bins = 10, density= True,  alpha=0.4, edgecolor='0.4', label='True Class')
    
    
    plt.legend()
    plt.title(f'{name}')
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig(f'{name}')
    plt.show()
    
    return DP
    
def computing_projection_scat(proj_matrix, dataset, labels, name) -> np.ndarray:
    '''
    

    Args:
       
        proj_matrix : np.ndarray
                      Projection matrix computed through PCA or LDA
            
        dataset : np.ndarray
                  Input Dataset
        
        labels : np.ndarray
                 Input labels
        
        name : str
               Graph description

    Returns:
    
    DP : np.ndarray
         Projected Data 

    '''
    
    DP = proj_matrix.T @ dataset
    D0,D1 = ut.create_mask_binary(DP, labels)
   
    
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], alpha=0.5, label='Fake Class')
    plt.scatter(D1[0, :], D1[1, :], alpha=0.5, label='True Class')
    plt.legend()
    plt.title(f'{name}')
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig(f'{name}')
    plt.show()
    
    return DP



def search_t(t: float, LVAL: np.ndarray, s, DVAL_lda: np.ndarray):
    
    '''

    Args:
        t : float
            Threshold computed as average between means
        
        LVAL : np.ndarray
               Validation labels
      
        s : int
            Shape for the predicted labels np.ndarray 
        
        DVAL_lda : np.ndarray
            Projected validation data in the LDA subspace 
            
        Returns:
            None

    '''
    
    rate_array = []
    
    for thr in np.arange(t-2.0, t+2.0, 0.05):
        if thr != t:
            PVAL = ut.predict_labels(s, thr, DVAL_lda) - 1
            error_rate = ut.compute_error_rate(LVAL, PVAL)
            rate_array.append((thr, error_rate))
         
        
    return min(rate_array, key=lambda x: x[1]) #I'm returing the min in term of "error_rate"

def search_m(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray):
    
    '''

    Args:
        DTR : np.ndarray
              Input training data
        
        LTR : np.ndarray
              Training labels  
            
        DVAL : np.ndarray
              Validation data
        
        LVAL : np.ndarray
              Validation labels
              
        Returns:
            None

    '''
    
    rate_array = []
   
    for m in range(1,DTR.shape[0]):
        P = ut.get_PCA_projection_matrix(DTR, m)
        DTR_reduced_with_PCA  =  P.T @ DTR
        DVAL_reduced_with_PCA =  P.T @ DVAL
        print(f'Now both training and validation data are rapresenting with {m} features')
        
        Sbt, Swt = ut.computing_between_within_covariance(DTR_reduced_with_PCA, LTR)
        W_bin_new = ut.get_LDA_projection_matrix(Sbt, Swt, 1)
        #print(f'W_bin_new shape: {W_bin_new.shape}')
        
        if m != 1:
            W_bin_new = - W_bin_new
            
        name_training = 'Model training set LDA after PCA features {num}' 
        DTR_lda = computing_projection_hist(W_bin_new, DTR_reduced_with_PCA, LTR, name_training.format(num = m)) 
        name_validation = 'Validation set LDA after PCA features {num}'
        DVAL_lda = computing_projection_hist(W_bin_new, DVAL_reduced_with_PCA, LVAL, name_validation.format(num = m)) 
        
        #where DTR_lda and DVAL_lda are the data projected to the new direction found by LDA 
        
        threshold = ((DTR_lda[0, LTR == 0]).mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Note that projected samples have only 1 dimension
        PVAL = ut.predict_labels(LVAL.shape[0], threshold, DVAL_lda) -1
        error_rate = ut.compute_error_rate(LVAL, PVAL)
        print(f'The error rate projecting data on the LDA direction, preprocessing with PCA (m = {m}) and using t: {threshold} is: {error_rate} / {LVAL.shape[0]} =  {error_rate/LVAL.shape[0]}')
        rate_array.append((m, error_rate))
        
    return min(rate_array, key=lambda x: x[1]) #I'm returing the min in term of "error_rate"
    
def projection_on_ith_direction(P: np.ndarray, D: np.ndarray, L:np.ndarray, verbose: bool = False):
    
    '''

    Args:
        P : np.ndarray
            PCA projection matrix
        
        D : np.ndarray
              Input training data  
            
        L : np.ndarray
              Training labels
        
        verbose : bool
            If setted print some information on the direction over which the data are projected on              
    Returns:
        None

    '''
    
    if verbose:
        print(f'P: {P}\n P shape:  {P.shape}')    

    for direction in range(P.shape[0]):
        if verbose:
            print(f'direction n.{direction} : {P[:, direction]} - shape: {P[:, direction].shape}')
        name = 'PCA - projection on sigle direction {}'.format(direction + 1)
        computing_projection_hist(P[:, direction].reshape(P.shape[0], 1), D, L, name)    
    
    
def pca_and_lda():
    D, L = ut.load()
   

    P = ut.get_PCA_projection_matrix(D, D.shape[0])   # I'm taking all the columns of U so all the directions 
                                                      # for analyzing all of them. The columns are already sorted 
                                                      # in descending order
                               
    
    projection_on_ith_direction(P, D, L, False)
    
        
    P2 = ut.get_PCA_projection_matrix(D, 2) # i'm taking the two directions with the highest variance
    computing_projection_scat(P2, D, L, 'PCA - projection on the two principal directions')
    

    computing_projection_scat(P[:, 2:4], D, L, 'PCA - pojection on the third and fourth directions')
    computing_projection_scat(P[:, 4:], D, L, 'PCA - projection on the fifth and sixth directions')
        
    
    Sb, Sw = ut.computing_between_within_covariance(D, L)
    print(f'Between class covariance:\n{Sb}\nWithin class covariance:\n{Sw}')
    
    W = ut.get_LDA_projection_matrix(Sb, Sw, 1) 
    #print(f'W: {W} - W shape: {W.shape}')
    D_projected = computing_projection_hist(W, D, L, 'LDA - projection on the principal direction') 
    print('D_projected shape: ', D_projected.shape)
    
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    print(f'DTR shape: {DTR.shape} - DVAL shape: {DVAL.shape}')
    
    
    # We now use only the training data and labels to learn the projection matrix W
    Sbt, Swt = ut.computing_between_within_covariance(DTR, LTR)
   
    #W_bin = generalized_eigenvalue_problem(Sbt, Swt, 1)
    W_bin = ut.get_LDA_projection_matrix(Sbt, Swt, 1)
    print(f'W_bin: {W_bin} - W_bin shape: {W_bin.shape}')
    
    # We can compare the hinstogram of the projected validation samples
    # to the hinsogram of the projected model training samples
    
    DTR_lda = computing_projection_hist(-W_bin, DTR, LTR, 'Model training set (DTR, LTR)') # this will apply LDA on DTR
    DVAL_lda = computing_projection_hist(-W_bin, DVAL, LVAL, 'Validation set (DVAL, LVAl)') # we are using the learned W for projecting the validation samples
    #where DTR_lda and DVAL_lda are the data projected to the new direction found by LDA 
    
    # Computing the threshold
    threshold = ((DTR_lda[0, LTR == 0]).mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Note that projected samples have only 1 dimension
    PVAL = ut.predict_labels(LVAL.shape[0], threshold, DVAL_lda) -1
    error_rate = ut.compute_error_rate(LVAL, PVAL)
    print(f'The error rate using LDA direction and threshold: {threshold} is: {error_rate} / {LVAL.shape[0]} =  {error_rate/LVAL.shape[0]}')
    
    
    # Evaluation with best_t
    best_t, error_rate_found = search_t(0, LVAL, LVAL.shape[0], DVAL_lda)
    PVAL = ut.predict_labels(LVAL.shape[0], best_t, DVAL_lda) -1
    error_rate = ut.compute_error_rate(LVAL, PVAL)
    print(f'The error rate using LDA direction and best threshold: {best_t} is: {error_rate} / {LVAL.shape[0]} =  {error_rate/LVAL.shape[0]}')
    
    # Preprocessing with PCA 
    best_m, error_rate = search_m(DTR, LTR, DVAL, LVAL)
    print(f'Best value of m : {best_m} - error_rate: {error_rate} / {LVAL.shape[0]} =  {error_rate/LVAL.shape[0]}')