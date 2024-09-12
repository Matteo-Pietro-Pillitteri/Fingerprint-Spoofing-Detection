# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:54:43 2024

@author: pilli
"""

import utilities as ut
import numpy as np
import sys


def generative_models():
    Pc = 1/2
    D, L = ut.load()
    print(f'L shape: {L.shape}')
    print(f'D shape: {D.shape}')
    
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    print(f'DTR shape: {DTR.shape} - DVAL shape: {DVAL.shape}')

    μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(DTR, LTR)
    LLRs = ut.compute_log_likelihood_ratio(DVAL, μ0, Σ0, μ1, Σ1)
    
    # --- evaluation MVG  ---
    t = ut.computing_binary_tr_based_on_prior(Pc)
    print('Threshold: ', t)
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    # predict_labes return vectors of 2 and 1, i substract -1 to have an array of 1 and 0 
    accuracy, error_rate = ut.evaluation(predicted_labels - 1, DVAL.shape[1], LVAL)
    print(f'Binary Gaussian Classifier - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Tied Gaussian Model ---
    Σ = ut.general_within_class_covariance(samples_and_cov)
    LLRs =  LLRs = ut.compute_log_likelihood_ratio(DVAL, μ0, Σ, μ1, Σ)
    #print(f'LLRs: {LLRs} - Shape: {LLRs.shape}')
    
    # --- evaluation Binary Tied Case ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL.shape[1], LVAL)
    print(f'Binary Tied Gaussian Classifier - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- LDA re-arranged piece of code from lab3 ---
    Sbt, Swt = ut.computing_between_within_covariance(DTR, LTR)
    W_bin = ut.get_LDA_projection_matrix(Sbt, Swt, 1)
    print(f'W_bin shape: {W_bin.shape}')
   
    DTR_lda = (-W_bin).T @ DTR
    DVAL_lda = (-W_bin).T @ DVAL
                                         
    #computing the threshold for LDA
    threshold = ((DTR_lda[0, LTR == 0]).mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Note that projected samples have only 1 dimension
    predicted_labels = ut.predict_labels(LVAL.shape[0], threshold, DVAL_lda)
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL.shape[1], LVAL)
    print(f'LDA Classifier - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Naive Bayes Gaussian model ---
    LLRs = ut.compute_log_likelihood_ratio(DVAL, μ0, Σ0_diag, μ1, Σ1_diag)

    
    # --- evaluation Binary Naive Bayes ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL.shape[1], LVAL)
    print(f'Binary Naive Bayes Gaussian Classifier - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- covariance and correlation ---
    print(f'Covariance matrix Fake class:\n{Σ0}\nShape Σ0: {Σ0.shape}')
    print(f'Covariance matrix True class:\n{Σ1}\nShape Σ1: {Σ1.shape}')
    
    Corr0 = ut.correlation_matrix(Σ0)
    print(f'Correlation matrix Fake class:\n{Corr0}\nShape correlation matrix Fake Class: {Corr0.shape}')
    
    Corr1 = ut.correlation_matrix(Σ1)
    print(f'Correlation matrix True class:\n{Corr1}\nShape correlation matrix True Class: {Corr1.shape}')
    
    
    # --- Analysis by discarding the last 2 features
    DTR_last2 = DTR[:-2, :]
    DVAL_last2 = DVAL[:-2, :]
    
    μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(DTR_last2, LTR)
    LLRs = ut.compute_log_likelihood_ratio(DVAL_last2, μ0, Σ0, μ1, Σ1)
    
    # --- evaluation MVG (without last 2 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels - 1, DVAL.shape[1], LVAL)
    print(f'Binary Gaussian Classifier (4 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Tied Gaussian Model (without last 2 features)---
    Σ = ut.general_within_class_covariance(samples_and_cov)
    LLRs = ut.compute_log_likelihood_ratio(DVAL_last2, μ0, Σ, μ1, Σ)
    
    # --- evaluation Binary Tied Case (without last 2 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL_last2.shape[1], LVAL)
    print(f'Binary Tied Gaussian Classifier (4 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Naive Bayes Gaussian model (without last 2 features) ---
    LLRs = ut.compute_log_likelihood_ratio(DVAL_last2, μ0, Σ0_diag, μ1, Σ1_diag)
    
    # --- evaluation Binary Naive Bayes (without last 2 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL_last2.shape[1], LVAL)
    print(f'Binary Naive Bayes Gaussian Classifier (4 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Analysis by using features 1-2 --- 
    DTR12 = DTR[:2, :]
    DVAL12 = DVAL[:2, :]
    
    μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(DTR12, LTR)
    LLRs = ut.compute_log_likelihood_ratio(DVAL12, μ0, Σ0, μ1, Σ1)

    # --- evaluation MVG ( 1 - 2 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels - 1, DVAL12.shape[1], LVAL)
    print(f'Binary Gaussian Classifier (1-2 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Tied Gaussian Model (1 -2 features)---
    Σ = ut.general_within_class_covariance(samples_and_cov)
    LLRs = ut.compute_log_likelihood_ratio(DVAL12, μ0, Σ, μ1, Σ)
    
    # --- evaluation Binary Tied Case (1 - 2 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL12.shape[1], LVAL)
    print(f'Binary Tied Gaussian Classifier (1-2 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Naive Bayes Gaussian model ( 1 - 2 features) ---
    LLRs = ut.compute_log_likelihood_ratio(DVAL12, μ0, Σ0_diag, μ1, Σ1_diag)

    # --- evaluation Binary Naive Bayes ( 1 - 2  features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL12.shape[1], LVAL)
    print(f'Binary Naive Bayes Gaussian Classifier (1-2 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Analysis by using features 3-4 --- 
    DTR34 = DTR[2:4, :]
    DVAL34 = DVAL[2:4, :]
    
    μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(DTR34, LTR)
    LLRs = ut.compute_log_likelihood_ratio(DVAL34, μ0, Σ0, μ1, Σ1)

    # --- evaluation MVG ( 3 - 4 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels - 1, DVAL34.shape[1], LVAL)
    print(f'Binary Gaussian Classifier (3-4 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Tied Gaussian Model (3 -4 features)---
    Σ = ut.general_within_class_covariance(samples_and_cov)
    LLRs = ut.compute_log_likelihood_ratio(DVAL34, μ0, Σ, μ1, Σ)
  
    # --- evaluation Binary Tied Case (3 - 4 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL34.shape[1], LVAL)
    print(f'Binary Tied Gaussian Classifier (3-4 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Binary Naive Bayes Gaussian model ( 3 - 4 features) ---
    LLRs = ut.compute_log_likelihood_ratio(DVAL34, μ0, Σ0_diag, μ1, Σ1_diag)

    # --- evaluation Binary Naive Bayes ( 3-4 features) ---
    predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
    accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL34.shape[1], LVAL)
    print(f'Binary Naive Bayes Gaussian Classifier (3 - 4 features) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
    
    # --- Classification after preprocessing with PCA
    for m in range(1,DTR.shape[0]):
        P = ut.get_PCA_projection_matrix(DTR, m)
        DTR_reduced_with_PCA  =  P.T @ DTR
        DVAL_reduced_with_PCA =  P.T @ DVAL
        
        μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(DTR_reduced_with_PCA, LTR)
        LLRs = ut.compute_log_likelihood_ratio(DVAL_reduced_with_PCA, μ0, Σ0, μ1, Σ1)
        
        # --- evaluation MVG (PCA preprocessing) ---
        predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
        accuracy, error_rate = ut.evaluation(predicted_labels - 1, DVAL_reduced_with_PCA.shape[1], LVAL)
        print(f'Binary Gaussian Classifier (PCA preprossessing - m: {m}) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
        
        # --- Binary Tied Gaussian Model (PCA preprocessing)---
        Σ = ut.general_within_class_covariance(samples_and_cov)
        LLRs = ut.compute_log_likelihood_ratio(DVAL_reduced_with_PCA, μ0, Σ, μ1, Σ)
        
        # --- evaluation Binary Tied Case (PCA preprocessing) ---
        predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
        accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL_reduced_with_PCA.shape[1], LVAL)
        print(f'Binary Tied Gaussian Classifier (PCA preprossessing - m: {m}) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
        
        # --- Binary Naive Bayes Gaussian model (PCA preprocessing) ---
        LLRs = ut.compute_log_likelihood_ratio(DVAL_reduced_with_PCA, μ0, Σ0_diag, μ1, Σ1_diag)
 
        # --- evaluation Binary Naive Bayes (PCA preprocessing)  ---
        predicted_labels = ut.predict_labels(LLRs.shape[0], t, ut.mrow(LLRs))
        accuracy, error_rate = ut.evaluation(predicted_labels-1, DVAL_reduced_with_PCA.shape[1], LVAL)
        print(f'Binary Naive Bayes Gaussian Classifier (PCA preprossessing - m: {m}) - Accuracy: {accuracy:.2%}- Error rate: {error_rate:.2%}')
    
        