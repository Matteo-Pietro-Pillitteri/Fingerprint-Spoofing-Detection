# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 18:56:20 2024

@author: pilli
"""
import utilities as ut
import numpy as np

def logistic_regression_analysis():
    D, L = ut.load()
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
   
  
    print('\n\n---- All samples -----')
    target_π = 0.1
    ut.train_and_test_with_different_lambda(DTR, LTR, DVAL, LVAL, target_π, -4.0, 2.0, 13, False, f'LR - Min and Actual DCF for different λ - π = {str(target_π).replace(".", ",")} - All samples')
    
    
    print('\n\n---- One out 50 -----')
    ut.train_and_test_with_different_lambda(DTR[:, ::50], LTR[::50], DVAL, LVAL, target_π, -4.0, 2.0, 13, False, f'LR one out 50- Min and Actual DCF for different λ - π = {str(target_π).replace(".", ",")} - One out 50 samples')
    
    
    print('\n\n---- Prior weighted model -----')
    print(f'Training label for the true class:{np.sum(LTR == 1)} - empirical prior true class: {(np.sum(LTR == 1)/LTR.size):.2%}')
    print(f'Training label for the fake class: np.sum(LTR == 0) - empirical prior true class: {(np.sum(LTR == 0)/LTR.size):.2%}')
    ut.train_and_test_with_different_lambda(DTR, LTR, DVAL, LVAL, target_π, -4.0, 2.0, 13, True, f'Weighted LR - Min and Actual DCF for different λ - π = {str(target_π).replace(".", ",")} - All samples')
    
    print('\n\n---- Quadratic Logistic Regression model -----')
    expanded_DTR = ut.expanded_feature_space(DTR)
    expanded_DVAL = ut.expanded_feature_space(DVAL)
    ut.train_and_test_with_different_lambda(expanded_DTR, LTR, expanded_DVAL, LVAL, target_π, -4.0, 2.0, 13, False, f'Quadratic LR - Min and Actual DCF for different λ - π = {str(target_π).replace(".", ",")} - All samples')
    
    
    print('\n\n---- Regularized model -----')
    print('---- Centering -----')
    training_mean = ut.dataset_mean(DTR)
    DTRC = DTR - training_mean
    DVALC = DVAL - training_mean
    ut.train_and_test_with_different_lambda(DTRC, LTR, DVALC, LVAL, target_π, -4.0, 2.0, 13, False, f'LR (Centering preprocessing) - Min and Actual DCF for different λ - π = {str(target_π).replace(".", ",")} - All samples')
    
    print('\n---- Z-Score Normalization -----')
    std = ut.mcol(DTR.std(1))
    ZNormDTR = (DTR- training_mean) / std
    ZNormDVAL = (DVAL - training_mean) /std
    ut.train_and_test_with_different_lambda(ZNormDTR, LTR, ZNormDVAL, LVAL, target_π, -4.0, 2.0, 13, False, f'LR (ZScore normalization) - Min and Actual DCF for different λ - π = {str(target_π).replace(".", ",")} - All samples')
    