# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:24:03 2024

@author: pilli
"""

import numpy as np
import utilities as ut


def predictions_and_DCF(min_DCF_dict: dict, calibration_losses: dict, π: float,  Cfn: float, Cfp: float, binary_llrs: np.ndarray, LVAL: np.ndarray, PCA: str = 'PCA: No', Model: str = 'MVG'):
    '''
    
    
    Args:
        min_DCF_dict : dict
        
        calibration_losses : dict
        
        π : float
            Prior probability
            
        Cfn : float
            C0,1 we are predicting 0, so we are classifing the sample
            as false while the actual class is True. We can denote with C(0) 
            the expected cost for predicting the False Class for the test sample
            
        Cfp : float
            C1,0 we are predicting 1, so we are classifing the sample
            as true while the actual class is False. We can denote with C(1)
            the expected cost for predicting the True Class for the test sample
            
        binary_llrs : np.ndarray
            Log-likelihood ratio scores
            
        LVAL : np.ndarray
            Labels for the validation data
        
        PCA : str, optional
            Str value used for some check. The default is 'PCA: No'
        
        Model : str, optional
            The default is 'MVG'
        
    Returns:
        None
    
    Comments:
        In this function, binary optimal bayes decision is performed using the threshold corresponding to the effective prior. The function here
        is taking in input effective priors.   The DCF computed using the threshold corresponding to the effective prior is called Actual DCF. 
    '''
    
    print(f'\nModel: {Model} - {PCA}')
    predicted_labels = ut.binary_opt_bayes_decision(π, Cfn, Cfp, binary_llrs) 
    M = ut.confusion_matrix(predicted_labels, LVAL) 
    DCF = ut.compute_binary_DCF(π, Cfn, Cfp, M)
    normalized_DCF = ut.binary_normalized_DCF(DCF, π, Cfn, Cfp) 
    minDCF = ut.compute_min_DCF(π, Cfn, Cfp, binary_llrs, LVAL)
    
    calibration_loss = normalized_DCF - minDCF # Note: in this case our normalized DCF refers to the 'Actual' DCF
    if PCA != 'PCA: No':
        m = PCA.split('=')[1]
        
    model_datails = Model + ' (without PCA)' if PCA == 'PCA: No' else Model + f' (PCA (m = {m}))'
    if π not in list(min_DCF_dict.keys()):
        min_DCF_dict[π] = [(model_datails,minDCF)]
        calibration_losses[π] = [(model_datails, (calibration_loss/minDCF)*100)]
    else:
        min_DCF_dict[π].append((model_datails,minDCF))
        calibration_losses[π].append((model_datails,(calibration_loss/minDCF)*100))
        
    print(f'unormalized DCF: {round(DCF, 4)} - normalized DCF: {round(normalized_DCF, 4)} - minimum DCF: {round(minDCF,4)} - calibration loss: {round(calibration_loss,4)}')

def bayes_decisions():
    
    min_DCF_dict = {}
    calibration_losses = {}
    
    D, L = ut.load() 
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    
    
    '''
        five application:
        • (0.5, 1.0, 1.0)
        • (0.9, 1.0, 1.0)
        • (0.1, 1.0, 1.0)
        • (0.5, 1.0, 9.0)
        • (0.5, 9.0, 1.0)
    '''
    effective_π1 = ut.compute_effective_prior(0.5, 1.0, 1.0)
    print(f'Application n.1 is equivalent to the application ({effective_π1}, 1.0, 1.0)\n')
    
    effective_π2 = ut.compute_effective_prior(0.9, 1.0, 1.0)
    print(f'Application n.2 is equivalent to the application ({effective_π2}, 1.0, 1.0)\n')
    
    effective_π3 = ut.compute_effective_prior(0.1, 1.0, 1.0)
    print(f'Application n.3 is equivalent to the application ({effective_π3}, 1.0, 1.0)\n')
    
    effective_π4 = ut.compute_effective_prior(0.5, 1.0, 9.0)
    print(f'Application n.4 is equivalent to the application ({effective_π4}, 1.0, 1.0)\n')
    
    effective_π5 = ut.compute_effective_prior(0.5, 9.0, 1.0)
    print(f'Application n.5 is equivalent to the application ({effective_π5}, 1.0, 1.0)\n')
    
    '''
        We focus now our attention on:
        • (0.1, 1.0, 1.0)
        • (0.5, 1.0, 9.0)
        • (0.9, 1.0, 1.0)
        
    '''
    π1 = effective_π3
    π2 = effective_π1
    π3 = effective_π2
    
    
    for application in [π1, π2, π3]:  # These are effective priors
        print(f'\n\n----------Analysis application: ({application}, 1.0, 1.0)--------------')
        # Without PCA 
    
        # MVG
        μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(DTR, LTR)
        LLRs = ut.compute_log_likelihood_ratio(DVAL, μ0, Σ0, μ1, Σ1)
        predictions_and_DCF(min_DCF_dict, calibration_losses, application, 1.0, 1.0, LLRs, LVAL)
      
        # Tied
        Σ = ut.general_within_class_covariance(samples_and_cov)
        LLRs = ut.compute_log_likelihood_ratio(DVAL, μ0, Σ, μ1, Σ)
        predictions_and_DCF(min_DCF_dict, calibration_losses, application, 1.0, 1.0, LLRs, LVAL, 'PCA: No', 'Tied')
         
        # Naive Bayes
        LLRs = ut.compute_log_likelihood_ratio(DVAL, μ0, Σ0_diag, μ1, Σ1_diag)
        predictions_and_DCF(min_DCF_dict, calibration_losses, application, 1.0, 1.0, LLRs, LVAL, 'PCA: No', 'NB')
       
    
        # Preprocessing with PCA 
        for m in range(1, DTR.shape[0] + 1):
            
            P = ut.get_PCA_projection_matrix(DTR, m)
            DTR_reduced_with_PCA  =  P.T @ DTR
            DVAL_reduced_with_PCA =  P.T @ DVAL
           
            # MVG
            μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(DTR_reduced_with_PCA, LTR)
            LLRs = ut.compute_log_likelihood_ratio(DVAL_reduced_with_PCA, μ0, Σ0, μ1, Σ1)
            predictions_and_DCF(min_DCF_dict, calibration_losses, application, 1.0, 1.0, LLRs, LVAL, f'PCA: Yes, m = {m}')
            
            # Tied 
            Σ = ut.general_within_class_covariance(samples_and_cov)
            LLRs = ut.compute_log_likelihood_ratio(DVAL_reduced_with_PCA, μ0, Σ, μ1, Σ)
            predictions_and_DCF(min_DCF_dict, calibration_losses, application, 1.0, 1.0, LLRs, LVAL, f'PCA: Yes, m = {m}', 'Tied')
           
            # Naive Bayes
            LLRs = ut.compute_log_likelihood_ratio(DVAL_reduced_with_PCA, μ0, Σ0_diag, μ1, Σ1_diag)
            predictions_and_DCF(min_DCF_dict, calibration_losses, application, 1.0, 1.0, LLRs, LVAL, f'PCA: Yes, m = {m}', 'NB')
           
        
    print('\nAnalysis based on min DCF')
    for key, value in min_DCF_dict.items():
        best_m, best_min_DCF = min(value, key=lambda x: x[1])
        print(f'\nBest model based on the min DCF for the application ({key}, 1.0, 1.0) is {best_m} with a minDCF of {best_min_DCF}')
    

    
    print('\nHow much the calibration loss corresponds to the minDCF in percentage?')
    for key, value in calibration_losses.items():
        print(f'\n\n---------------------Percentages for application ({key}, 1.0, 1.0)-----------------\n')
        # are the modell well calibrated ? For each model i compute how much the calibration loss corresponds to the minDCF 
        # and if this percentage is lower then 10% of the minimimumDCF i can conclude that that model (with that configuration of PCA for instance) is well calibrated
        mvg_percentages = []
        tied_percentages = []
        nb_percentages = []
        
        for item in value:
            if item[0].startswith('MVG'):
                mvg_percentages.append(item)
            elif item[0].startswith('Tied'):
                tied_percentages.append(item)
            elif item[0].startswith('NB'):
                nb_percentages.append(item)
                
                
        print("MVG percentages:")
        for item in mvg_percentages:
            print(item)
        
        print("\nTied percentages:")
        for item in tied_percentages:
            print(item)
        
        print("\nNB percetanges:")
        for item in nb_percentages:
            print(item)
                
        
        models_calibrated = [True if model[1] < 10.0 else False for model in value]
        # i consider overall that the models are well calibrated if out of 18 configuration at least 13/18 are well calibrated, so if 72% of the models are well calibrated for that applicatioon
        model_well_calibrated_for_application = 'well calibrated' if sum(models_calibrated) >= 13 else 'not well calibrated'
        print(f'Overall for the considered application, the models are {model_well_calibrated_for_application}')
    
        
    
    # Searching the best PCA configuration for each model for the specified applicaiton
    # IMPORTANT NOTE: in this part of the project it was considered one possible PCA setup as "No PCA / Whitout PCA" and another one "PCA with m == 6" 
    idx_model = {
        0: 'MVG',
        1: 'Tied',
        2: 'NB'
    }
    
    best_conf = ut.searching_best_PCA_configuration_for_minDCF(0.1, min_DCF_dict, [el[1] for el in list(idx_model.items())])
    print(best_conf)
    

    LLRs_list = []
    for idx, (best_m, min_dcf) in enumerate(best_conf):
        m = best_m if m != 0 else 'no PCA'
        print(f'Model: {idx_model[idx]} - best PCA setup: m  = {best_m} - Minimum DCF obtained: {round(min_dcf, 5)}')
        
        
        if m != 0:
            P = ut.get_PCA_projection_matrix(DTR, best_m)
            DTR_reduced_with_PCA  =  P.T @ DTR
            DVAL_reduced_with_PCA =  P.T @ DVAL
            
            new_DTR = DTR_reduced_with_PCA
            new_DVAL = DVAL_reduced_with_PCA
        else:
            new_DTR = DTR
            new_DVAL = DVAL
            
        
        μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag = ut.compute_statistic_for_gaussian_classifiers(new_DTR, LTR)
        if idx == 0:
            LLRs = ut.compute_log_likelihood_ratio(new_DVAL, μ0, Σ0, μ1, Σ1)
        elif idx == 1:    
            Σ = ut.general_within_class_covariance(samples_and_cov)
            LLRs = ut.compute_log_likelihood_ratio(new_DVAL, μ0, Σ, μ1, Σ)
        else:
            LLRs = ut.compute_log_likelihood_ratio(new_DVAL, μ0, Σ0_diag, μ1, Σ1_diag)
        
        LLRs_list.append(LLRs)
        plot_description = f'PCA with m = {best_m}' if m != 0 else 'no PCA'
        ut.bayes_error_plot(LLRs, LVAL, 4, 0, f'Bayes error plot for model {idx_model[idx]}, {plot_description}')
    
    ut.plot_multiple_bayes_error_plots(LLRs_list, LVAL, 4, idx_model, 'Comparision between bayes plot for the MVG, Tied and Naive Bayes Gaussian classifiers')
  
        
    