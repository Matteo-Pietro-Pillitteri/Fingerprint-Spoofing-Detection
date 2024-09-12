# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:56:09 2024

@author: pilli
"""
import utilities as ut
import numpy as np


def gmm_analysis():
    D, L = ut.load()
    
    
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    print(f'DTR shape: {DTR.shape} - DVAL shape: {DVAL.shape}')
    
    target_π = 0.1
    print('\n\n----full model analysis----')
    ut.train_and_test_GMM(DTR, LTR, DVAL, LVAL, target_π, 32, 'full', 1e-6,  True, 0.01)
    
    print('\n\n----diagonal model analysis----')
    ut.train_and_test_GMM(DTR, LTR, DVAL, LVAL, target_π, 32, 'diagonal', 1e-6,  True, 0.01)
    
    
    idx_model = {
        0: 'bestLR',
        1: 'bestSVM',
        2: 'bestGMM'
    }
    
   
        
    models = ut.best_performing_candidate_for_method(target_π, 1.0, 1.0, idx_model, LVAL)
    LLRs_list = [model['score'].ravel() if idx == 1 else model['LLR'] for idx, model in enumerate(models)]
    
    ut.plot_multiple_bayes_error_plots(LLRs_list, LVAL, 4, idx_model, 'Comparision between bayes plot')
    
   
    