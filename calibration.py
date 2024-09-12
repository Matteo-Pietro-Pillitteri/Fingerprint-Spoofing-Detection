# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:04:00 2024

@author: pilli
"""

import utilities as ut
import numpy as np



def calibration():
    
    bestGMM = np.load('bestGMM.npy', allow_pickle=True)
    bestGMM.item()['model'] = bestGMM.item()['variant'] + ' GMM' 
    bestLR = np.load('bestLR.npy', allow_pickle=True)
    bestSVM = np.load('bestSVM.npy', allow_pickle=True)
    
    
    idx_best_model = {
        0: bestGMM,
        1: bestLR,
        2: bestSVM,
    }
    
    
    target_π = 0.1

    D, L = ut.load()
    (DTR, LTR), (DVAL, LVAL) = ut.split_db_2to1(D, L)
    print(f'DTR shape: {DTR.shape} - DVAL shape: {DVAL.shape}')
    
    
    fused_scores = []
    for model in list(idx_best_model.values()):
    
        items = model.item()
        model_name = items['model']
        model_scores = items['LLR'] if 'SVM' not in model_name else items['score'].ravel()
        fused_scores.append(model_scores)
        print('model_scores shape: ', model_scores.shape)
        shuffled_scores, shuffled_labels = ut.shuffle_scores(model_scores, LVAL)
        print('shuffled_scores: ', shuffled_scores.shape)
        print('shuffled_labels: ', shuffled_labels.shape)
        # --- Raw scores ---
        actDCF, minDCF = ut.predictions_and_DCF(target_π, 1.0, 1.0, shuffled_scores, shuffled_labels) # RISULTATI GIUSTI CONTROLLATI DAL MIO REPORT COINCIDONO 
        print(f'{model_name} - Raw scores - actual DCF : {round(actDCF, 4)} - minimum DCF: {round(minDCF, 4)}\n\n')
        pre_calibration_dcfs_model = ut.bayes_error_plot(shuffled_scores, shuffled_labels, 3, 0.001, f'Bayes error plot {model_name} - Raw scores')
            
        cal_scores, cal_labels, best_train_π = ut.train_calibration_model(shuffled_scores, shuffled_labels, target_π)
        
        # --- Calibrated scores ---      
        actDCF, minDCF = ut.predictions_and_DCF(target_π, 1.0, 1.0, cal_scores, cal_labels)
        print(f'Best performing calibration trasformation for {model_name}: calibration model trained on: {best_train_π} - Calibrated scores - K-fold - actual DCF : {round(actDCF, 4)} - minimum DCF: {round(minDCF, 4)}\n\n')
        ut.bayes_error_plot(cal_scores, cal_labels, 3, 0.001, f'Bayes error {model_name} - Calibrated scores - K-fold', pre_calibration_dcfs_model)
        
        items['cal_scores'] = cal_scores
        items['LCAL'] = cal_labels
        items['best_cal_train_π'] = best_train_π
        
        # Train final calibration model with the best prior found with the k-fold procedure
        LR_parameters_final_calibrator, _ , _ = ut.trainPriorWeightedLogReg(ut.mrow(model_scores), LVAL, 0.0, best_train_π)
        print('LR_parameters_final_calibrator: ', LR_parameters_final_calibrator)
        items['calibrator_parameters'] = LR_parameters_final_calibrator
        
        np.save(f'final {model_name}', items)
    
    # Fusion 
    lowest_actDCF = np.inf
    best_train_π = target_π
    ut.train_calibration_for_fusion(target_π, True, fused_scores, LVAL)
    ut.train_calibration_for_fusion(target_π, False, fused_scores, LVAL)


    # evaluation 
    # We need to compute the score for this evaluation set with the delivered model
    DEVAL, LEVAL = ut.load('Project/evalData.txt')
    π_emp = np.sum(LEVAL == 1) / LEVAL.size
    print('Evaluation empirical prior: ', π_emp)
    
    print(f'DEVAL shape: {DEVAL.shape} - LEVAL shape: {LEVAL.shape}')
    cal_scores_list = []
    eval_scores_list = []
    delivered_system = np.load('final diagonal GMM.npy', allow_pickle= True)  
    eval_scores, eval_cal_scores = ut.evaluate_system_on_application(target_π, DTR, LTR, DEVAL, LEVAL, LVAL, delivered_system)
    cal_scores_list.append(eval_cal_scores)
    eval_scores_list.append(eval_scores)
    
    final_LR = np.load('final Quadratic LR.npy', allow_pickle= True) 
    eval_scores, eval_cal_scores = ut.evaluate_system_on_application(target_π, DTR, LTR, DEVAL, LEVAL, LVAL, final_LR)
    cal_scores_list.append(eval_cal_scores)
    eval_scores_list.append(eval_scores)
    
    final_SVM = np.load('final RBG SVM.npy', allow_pickle= True)
    eval_scores, eval_cal_scores = ut.evaluate_system_on_application(target_π, DTR, LTR, DEVAL, LEVAL, LVAL, final_SVM)
    cal_scores_list.append(eval_cal_scores)
    eval_scores_list.append(eval_scores)
    
    final_fusion = np.load('Fusion - unshuffled scores.npy', allow_pickle= True)
    actDCF, minDCF, cal_scores = ut.evaluate_calibration_model(target_π, 1.0, 1.0, final_fusion.item()['calibrator_parameters'], np.vstack(eval_scores_list), LEVAL) 
    print(f'Fusion -- evaluation set --> actual DCF : {round(actDCF, 3)} - minimum DCF: {round(minDCF, 3)}')
  
    cal_scores_list.append(cal_scores) 
    
    # Actual DCF error plots
    idx_final_systems = {
        0: 'final GMM',
        1: 'final LR',
        2: 'final SVM',
        3: 'final fusion'
    }
    
    ut.plot_actual_DCF_error_plots(cal_scores_list, LEVAL, 3, idx_final_systems, 'Actual DCF error plots for final systems')
    ut.plot_multiple_bayes_error_plots(cal_scores_list, LEVAL, 3, idx_final_systems, 'Comparision between Bayes Error Plots')
    

    # selected approach: GMM
    
    gmm_models = []
    diagonal_GMMs = np.load('Model\\diagonal GMM.npy', allow_pickle=True)
    gmm_models.append(diagonal_GMMs)
    full_GMMs = np.load('Model\\full GMM.npy', allow_pickle=True)
    gmm_models.append(full_GMMs)
    
    for variant in gmm_models:
        for model in variant:
            ut.evaluate_GMMs_on_evaluation(target_π, DEVAL, LEVAL, model)
   
    delivered_system = np.load('final diagonal GMM.npy', allow_pickle= True)  
    ut.evaluate_GMMs_on_evaluation(target_π, DEVAL, LEVAL, delivered_system.item())
     
    
    
   
                                                         