# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:25:46 2024

@author: pilli
"""

import numpy as np
import utilities as ut

def svm_analysis():
    D, L = ut.load()
    
    
    #(DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D[:, :60], L[:60])
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    print(f'DTR shape: {DTR.shape} - DVAL shape: {DVAL.shape}')
    
    
    K = 1
    # ---- linear SVM ----
    EDTR = ut.compute_extended_training_data(DTR, K)
    EDVAL = ut.compute_extended_training_data(DVAL, K)
    print(f'\nEDTR shape: {EDTR.shape}')
    print(f'\nEDVAL shape: {EDVAL.shape}')
    
    H = ut.compute_H_SVM(EDTR, LTR)
  
    
    target_π = 0.1
    ut.train_and_test_SVM(EDTR, LTR, EDVAL, LVAL,  target_π, H, K, f'Linear SVM - Min and Actual DCF for different C - π = {str(target_π).replace(".", ",")} - no centering ')
    
    # ---- linear SVM, centered data ----
    DTR_mean = ut.dataset_mean(DTR) 
    CDTR = DTR - DTR_mean
    CDVAL = DVAL - DTR_mean
    EDTR = ut.compute_extended_training_data(CDTR, K)
    EDVAL = ut.compute_extended_training_data(CDVAL, K)
    H = ut.compute_H_SVM(EDTR, LTR)
    
    target_π = 0.1
    ut.train_and_test_SVM(EDTR, LTR, EDVAL, LVAL,  target_π, H, K, f'Linear SVM - Min and Actual DCF for different C - π = {str(target_π).replace(".", ",")} - centered data ')
    
    
    '''
    EDTR = ut.compute_extended_training_data(DTR, K)
    EDVAL = ut.compute_extended_training_data(DVAL, K)
    EDTR_mean = ut.dataset_mean(EDTR) 
    CEDTR = EDTR - EDTR_mean
    CEDVAL = EDVAL - EDTR_mean 
    
    H = ut.compute_H_SVM(CEDTR, LTR)
 
    target_π = 0.1
    #ut.train_and_test_SVM(CEDTR, LTR, CEDVAL, LVAL,  target_π, H, K, f'Linear SVM - Min and Actual DCF for different C - π = {str(target_π).replace(".", ",")} - centered data ')
    '''
    
    # ---- Polinomial kernel SVM ----
    c = 1
    d = 2
    K = 0.0 # K = square_root(ε) , if ε = 0 then K = 0
    ε = K**2 
    
    ut.train_and_test_kernel_SVM(DTR, LTR, DVAL, LVAL, target_π, K, d, c, ε, 'poly-d', f'polynomial kernel SVM - Min and Actual DCF for different C - π = {str(target_π).replace(".", ",")} - no centering ')
    
    # ---- RBF kernel ----
    K = 1.0
    ε = K**2 
    c = 0
    d = 0 
    
    ut.train_and_test_kernel_SVM(DTR, LTR, DVAL, LVAL, target_π, K, d, c, ε, 'RBF', f'RBF kernel SVM - Min and Actual DCF for different C - π = {str(target_π).replace(".", ",")} - no centering ')