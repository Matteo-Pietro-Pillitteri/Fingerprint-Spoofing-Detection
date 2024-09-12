# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:55:11 2024

@author: pilli
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy
import math
import random
from scipy.signal import find_peaks


''' Managing data '''


def load(file_name: str = 'trainData.txt') -> tuple[np.ndarray, np.ndarray]:
    '''
    

    Args:
    
    file_name : str, optional
                Name of file from which load the data. The default is 'trainData.txt'.

    Returns:
    
    D : np.ndarray
        Loaeded data
        
    L : np.ndarray
        Loaded labels.

    '''
    
    labels = []
    samples = []

    with open(file_name, 'r') as f:
        
        for index, line in enumerate(f):
            splitted_line = line.split(',')
            labels.append(splitted_line[-1].rstrip())
            L = np.array(labels, dtype = np.int8)
            sample_array = np.array(splitted_line[:-1], dtype=np.float32).reshape((6,1))
            samples.append(sample_array)
            D = np.hstack(samples)    
            
    return D, L


def dataset_mean(dataset: np.ndarray):
    '''
    
    Args:
    
        dataset : np.ndarray
                  Input dataset

    Returns:
        
        mu : np.ndarray
             Mean of the dataset as column vector
    
    '''
    mu = dataset.mean(1)
   
    return mcol(mu)

def mcol(_1dvec: np.ndarray) -> np.ndarray:
    '''
    

    Args:
        _1dvec : np.ndarray
                 1-D input numpy array

    Returns:
        _1dvec.reshape(_1dvec.shape[0], 1) : np.ndarray
                                             Column vector
    '''
    
    return _1dvec.reshape(_1dvec.shape[0], 1)

def mrow(_1dvec: np.ndarray) -> np.ndarray:
    '''
    

    Args:
    
        _1dvec : np.ndarray
                 1-D input numpy array

    Returns:
        _1dvec.reshape(1, _1dvec.shape[0]) : np.ndarray
                                             Row vector

    '''
    return _1dvec.reshape(1, _1dvec.shape[0])

def covariance_matrix(dataset_centered: np.ndarray) -> np.ndarray:
    '''
    

    Args:
        dataset_centered : np.ndarray
                           Input data

    Returns:
        Σ : np.ndarray
            Covariance matrix

    '''
    
    Σ = (dataset_centered @ dataset_centered.T) / float(dataset_centered.shape[1]) 
    return Σ

def correlation_matrix(Σ: np.ndarray) -> np.ndarray:
    '''
    

    Args:
        Σ : np.ndarray
            Covariance matrix

    Returns:
        corr_matrix : np.ndarray
                      Correlation matrix

    '''
    
    corr_matrix = Σ / (mcol(Σ.diagonal() ** 0.5) *mrow(Σ.diagonal() ** 0.5))
    return corr_matrix

def compute_mu_C(dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    

    Args:
    
        dataset : np.ndarray
                  Input data

    Returns:
        
        mu : np.ndarray
             Mean vector
             
        C : np.ndarray
            Covariance matrix

    '''
    
    mu = dataset_mean(dataset)
    DC = dataset-mu
    C = covariance_matrix(DC)
    return mu, C

def create_mask_binary(dp: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    

    Args:
       
        dp : np.ndarray
             Input data
             
        l : np.ndarray
            Input labels

    Returns:
        
        D0 : np.ndarray
             Data of the Fake Class
             
        D1 : np.ndarray
             Data of the True Class

    '''
    
    M0 = (l == 0)
    M1 = (l == 1)
    
    D0 = dp[:, M0] 
    D1 = dp[:, M1]

    return D0,D1

def create_mask(dp: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    

    Args:
       
        dp : np.ndarray
             Input data
             
        l : np.ndarray
            Input labels

    Returns:
        
        D0 : np.ndarray
             Data of the Class 0
             
        D1 : np.ndarray
             Data of the Class 1
            
        D2 : np.ndarray
             Data of the Class 2

    '''
    
    M0 = (l == 0)
    M1 = (l == 1)
    M2 = (l == 2)
    
    D0 = dp[:, M0] # i'm taking all the row related to label '0'
    D1 = dp[:, M1]
    D2 = dp[:, M2]

    return D0,D1,D2


def create_mask_binary_iris(dp: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    

    Args:
       
        dp : np.ndarray
             Input data
             
        l : np.ndarray
            Input labels

    Returns:
        
             
        D1 : np.ndarray
             Data of the Class 1
            
        D2 : np.ndarray
             Data of the Class 2

    '''
    
    M1 = (l == 1)
    M2 = (l == 2)
    
    D1 = dp[:, M1]
    D2 = dp[:, M2]

    return D1,D2


def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)

''' --------------------------------------------------- '''



''' Data distribution'''
def computation_and_plot(D: np.ndarray, L: np.ndarray, feature: int = 0, b: int = 10):
    '''
        Args:
            D : np.ndarray
                Input data
            
            L : np.ndarary
                Input labels
            
            feature : int, optional
                      Feature index. Defeault value is 0 
            
            b : int, optional
                Number of bins. Default value is 10
                
        Returns:
            None
    
    '''
    M0 = (L == 0)
    M1 = (L == 1)

    
    D0 = D[:, M0] # i'm taking all the row related to label '0'
    D1 = D[:, M1]
 
    
    plt.figure()
    hist_fake, _, _ = plt.hist(D0[feature, :], bins = b, density= True,  alpha=0.4, edgecolor='0.4', label='Fake Class')
    hist_true, _, _ = plt.hist(D1[feature, :], bins = b, density= True,  alpha=0.4, edgecolor='0.4', label='True Class')
    # plt.hist returns n, bins, patches
    # n : array or list of arrays -> the values of the histogram bins

    plt.legend()
    plt.title(f'feature n.{feature} - bins n.{b}')
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig(f'feature num {feature} - hist with bins {b}')
    plt.show()
    
    # find_peaks takes a 1-D array an difns a local maxima by simple comparision of neighboring values.
    peaks_fake, _ = find_peaks(hist_fake, 0.308)
    peaks_true, _ = find_peaks(hist_true, 0.308)
    
    print(f'feature n.{feature} - Peaks for Fake class: {len(peaks_fake)}')
    print(f'feature n.{feature} - Peaks for True class: {len(peaks_true)}')
    
    if feature % 2 == 0:
        plt.figure()
        plt.scatter(D0[feature, :], D0[feature+1, :], alpha=0.5, label='Fake Class')
        plt.scatter(D1[feature, :], D1[feature+1, :], alpha=0.5, label='True Class')
        
        
        # By considerind the Dataset samples for each label and cosidering only the 2 features
        D0f = D0[feature:feature+2, :]
        D1f = D1[feature:feature+2, :]
        mu_fake = dataset_mean(D0f)
        mu_true = dataset_mean(D1f)
        
        plt.scatter(mu_fake[0], mu_fake[1], marker='2', color='blue', linewidth=2, alpha=1, s=200,  label ='Mean Fake Class')
        plt.scatter(mu_true[0], mu_true[1], marker='2', color='orange', linewidth=2, alpha=1, s=200, label='Mean True Class')
        print(f'\n\nResults for {feature} - {feature+1}')
        print(f'Mean fake class: \n{mu_fake} \n Mean true class: \n{mu_true} ')
        euclidean_distance = np.sqrt(np.sum((mu_fake - mu_true)**2))
        print(f'Euclidean distance: {euclidean_distance}')
        
        var_fake = mcol(D0f.var(1))
        var_true = mcol(D1f.var(1))
        print(f'Var fake class: \n{var_fake} \nVar true class: \n{var_true}')
        
        
        plt.xlabel(f'feature num {feature}')
        plt.ylabel(f'feature num {feature+1}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'feature {feature} - feature - {feature+1} scat')
        plt.show()



''' --------------------------------------------------- '''


''' Dimensionality reduction - Preprocessing '''


def get_PCA_projection_matrix(D: np.ndarray, dim: int) -> np.ndarray:
    '''
    
    Args:
        D : np.ndarray
            Input dataset
        
        dim : int 
            It identifies the number of dimensions to keep in the PCA subspace
            
    Returns:
        P : np.ndarray
            PCA Projection Matrix
    '''
    
    mu = dataset_mean(D)
    DC = D-mu
    C = covariance_matrix(DC)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:dim]
    return P

def get_LDA_projection_matrix(SB: np.ndarray, SW: np.ndarray, m: int) -> np.ndarray:
    '''
    Args:
        SB : np.ndarary
             Between Class Covariance 
        
        
        SW : np.ndarary
             Whitin Class Covariance
        
        m : int
            LDA space dimensions (at least C-1)
             
    Returns:
        P1.T @ P2 : np.ndarray
            LDA Projection Matrix
    
    '''
    
    U, s, _ = np.linalg.svd(SW)
    P1 = np.dot(np.dot(U, np.diag(1.0/(s**0.5))), U.T)
    SBt = (P1 @ SB) @ P1.T
    USBt, _ , _ = np.linalg.svd(SBt)
    P2 = USBt[:, 0:m]  
    return P1.T @ P2


def computing_between_within_covariance(dataset: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    
    Args:
        dataset : np.ndarray 
                  Input data
        
        labels : np.ndarray
                 Input labels
                 
    Returns:
        Sw : np.ndarary
             Whitin Class Covariance
        
        Sb : np.ndarary
             Between Class Covariance 
        
    '''
    
    mu = dataset_mean(dataset)
    classes_mean_and_samples = []
    
    D0,D1 = create_mask_binary(dataset, labels) 
    mu0 = dataset_mean(D0)
    classes_mean_and_samples.append((mu0, D0.shape[1]))
    mu1 = dataset_mean(D1)
    classes_mean_and_samples.append((mu1, D1.shape[1]))
 
   
    C0 = covariance_matrix(D0-mu0)
    C1 = covariance_matrix(D1-mu1)

    
    D0w = D0.shape[1] * C0
    D1w = D1.shape[1] * C1
    weighted_class_data = np.array([D0w, D1w]) 
    Sw = np.sum(weighted_class_data, axis = 0) / dataset.shape[1]  # dataset.shape[1] = N number of samples 
   
    between_class_covariance_terms = np.zeros((2, dataset.shape[0], dataset.shape[0])) # dataset shape num_features x samples --> dataset.shape[0] = num_features
    for idx, (mean,samples) in enumerate(classes_mean_and_samples):
        new_term = np.array(samples * ((mean - mu) @ (mean - mu).T))
        between_class_covariance_terms[idx] = new_term
        
    Sb = np.sum(between_class_covariance_terms, axis = 0) / dataset.shape[1]
    return Sb,Sw

''' --------------------------------------------------- '''

''' MVG '''

def logpdf_GAU_ND(x: np.ndarray, mu: np.ndarray, C: np.ndarray) -> np.ndarray:
    '''
    
    
    Args:
        x : np.ndarray
            Input Data
            
        mu : np.ndarray
             Mean of the input samples

        C : np.ndarray
            Covariance Matrix

    Returns:
        log_densities : np.ndarray        
            Array that contains the log densities corresponding to the input
    '''
  
    log_densities = np.zeros(x.shape[1])
    
    M = x.shape[0]   #M = num of features 
    inv = np.linalg.inv(C) 
    _, log_det = np.linalg.slogdet(C) 
    for idx, sample in enumerate(x.T): #By iterating on x.T i'm iterating on the column 
        sample = mcol(sample)    
        esp = (sample-mu).T @ inv @ (sample-mu) 
        term = -((M*np.log(2*np.pi))+ log_det + esp) / 2
        log_densities[idx] = term
        
    return log_densities

def loglikelihood(X: np.ndarray, m_ML, C_ML) -> np.float64: 
    '''
    Args:
        X : np.ndarray
            Input samples
            
    Returns:
        ll : np.float64
            Log-likelihood that corresponds to the sum of the log-density computed over all the samples
    '''
    
    ll = logpdf_GAU_ND(X, m_ML, C_ML).sum()
    return ll

def general_within_class_covariance(samples_and_cov: list):
    
    samples = sum(nc for nc, _ in samples_and_cov)
    total = sum(nc* sigmac for nc, sigmac in samples_and_cov)
    return total / samples # Computation done accordingly with Lab5 formula

def compute_statistic_for_gaussian_classifiers(D: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    '''
    
    Args:
        D : np.ndarray
            Input data
        
        L : np.ndarray
            Input labels
            
    Returns:
        μ1 : np.ndarray
             Mean vector for the True Class     
        
        Σ1 : np.ndarray
             Covariance matrix for True Class
        
        μ0 : np.ndarray
             Mean vector for the False Class 
        
        Σ0 : np.ndarray
             Covariance matrix for the False Class
        
        samples_and_cov : list
                          List containing tuple made by (sample of class k, covariance matrix for class k)
        
        Σ1_diag : np.ndarray
                  Diagonal Covariance matrix for True Class          
        
        Σ0_diag : np.ndarray
                  Diagonal Covariance matrix for False Class

    '''
    μ1, Σ1 = compute_mu_C(D[:, L == 1])
    samples_and_cov = [(D[:, L == 1].shape[1], Σ1)]
    Σ1_diag = Σ1 * np.identity(Σ1.shape[0])
    
    μ0, Σ0 = compute_mu_C(D[:, L == 0])
    samples_and_cov.append((D[:, L == 0].shape[1], Σ0))
    Σ0_diag = Σ0 * np.identity(Σ0.shape[0])
                           
    return μ1, Σ1, μ0, Σ0, samples_and_cov, Σ1_diag, Σ0_diag

def compute_log_likelihood_ratio(D: np.ndarray, μ0: np.ndarray, Σ0: np.ndarray, μ1: np.ndarray, Σ1: np.ndarray) -> np.ndarray:
    '''
    
    Args:
        D : np.ndarray
            Input data    
        
        μ0 : np.ndarray
             Mean vector for class 0 (False)
        
        Σ0 : np.ndarray
             Covariance matrix for class 0 (False)
        
        μ1 : np.ndarray
             Mean vector for class 1 (True)
             
        Σ1 : np.ndarray
             Covariance matrix for class 1 (True)
             
    Returns:
        ld_1 - ld_0 : np.ndarray
                      Computed log-densities
    '''
    
    ld_1 = logpdf_GAU_ND(D, μ1, Σ1)    # fx|c(x|1) = N(x|μ1, Σ1)
    ld_0 = logpdf_GAU_ND(D, μ0, Σ0)    # fx|c(x|0) = N(x|μ0, Σ0)
    
    return ld_1 - ld_0


''' --------------------------------------------------- '''


''' Classification '''

def predict_labels(s: int, t: float, val: np.ndarray) -> np.ndarray:
    '''
    

    Args:
        s : int
            Shape for the predicted labels np.ndarray
            
        t : float
            Threshold
            
        val : np.ndarray
              Validation data

    Returns:
        PVAL : np.ndarray
               Predicted labels

    '''
    
    PVAL = np.zeros(shape = s, dtype = np.int32)
    PVAL[val[0] >= t] = 2
    PVAL[val[0] < t] = 1
    return PVAL

def computing_binary_tr_based_on_prior(prior: float) -> np.float64: 
    '''
    

    Args:
    
        prior : float
            Prior probability

    Returns:
        t : np.float64
            Application based threshold for binary problem.

    '''
    p2 = prior
    p1 = 1 - prior
    t = - np.log(p2/p1)
    return t

def correct_predictions(ground_truth: np.ndarray, predictions: np.ndarray) -> int:
    '''
    
    Args:
        ground_truth : np.ndarray
            Real labels
            
        predictions :
            Model predictions / computed labels
            
    Returns:
        correct : int
                  Number of correctly classified samples computed as the number of y_i = c_i, where y_i is the prediction for sample i-th and c_i its true label
    
    '''
    
    
    correct = np.sum(ground_truth == predictions)
    return correct

def compute_error_rate(ground_truth: np.ndarray, predictions: np.ndarray) -> int:
    '''
    
    Args:
        ground_truth : np.ndarray
            Real labels
            
        predictions : np.ndarray
            Model predictions / computed labels
    
    Returns:
        errors : int
                  Number of incorrectly classified samples computed as the number of y_i != c_i, where y_i is the prediction for sample i-th and c_i its true label
    
    
    '''
    
    errors = np.sum(ground_truth != predictions)
    return errors

def evaluation(predictions: np.ndarray, num_samples: int, true_labels: np.ndarray) -> tuple[float, float]:
    '''
    
    Args:
        
        prediction : np.ndarray
            Model predictions / computed labels
        
        num_samples : int
                      Total amount of samples
                      
        true_labels : np.ndarray
                      Real labels
        
    Returns:
        acc : float
              Accuracy computed as number of correctly classified samples divided by the total amount of samples
             
        err : float
              Error rate computed as number of incorrectly classified samples divided by thte total amount of samples
        
    
    '''
    
    acc = correct_predictions(true_labels, predictions) / num_samples
    err = compute_error_rate(true_labels, predictions) / num_samples
    return acc, err

def compute_log_posterior(log_densities: np.ndarray, priors) -> np.ndarray:
    '''
    
    Args:
        log_densities : np.ndarray
            It contains the log_densities for some data
        
        priors : np.ndarray
            It contains the prior probabilities for different classes
    
    Returns:
        np.exp(logSPost) : np.ndarray
            Multidimensional Numpy array that contains the posterior probabilities
    
    '''

    logSJoint = log_densities + mcol(np.log(priors))
   
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis = 0))
    
    logSPost = logSJoint - logSMarginal
    
    return  np.exp(logSPost)

''' --------------------------------------------------- '''

''' Bayes Decision and model evaluation '''
def confusion_matrix(predictions, true_labels) -> np.ndarray:
    '''
    

    Args:
        predictions : TYPE
            Model predictions / computed labels
            
        true_labels : TYPE
            Real labels

    Returns:
        confusion_matrix : np.ndarray
            Computed Confusion Matrix

    '''
    num_classes = np.unique(true_labels).size
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            confusion_matrix[i, j] = np.count_nonzero(predictions[true_labels == j] == i)
                
    
    return confusion_matrix



def compute_bayes_treshold(π: float, Cfn: float, Cfp: float) -> np.float64:
    '''
    
    
    Args:
        π : float
            prior probability
        
        Cfn : float
              C0,1 we are predicting 0, so we are classifing the sample
              as false while the actual class is True. We can denote with C(0) 
              the expected cost for predicting the False Class for the test sample
    
        Cfp : float
              C1,0 we are predicting 1, so we are classifing the sample
              as true while the actual class is False. We can denote with C(1)
              the expected cost for predicting the True Class for the test sample

    Returns:
        t : numpy.float64
    

    '''
    num = π*Cfn
    den = (1-π)*Cfp
    t = - np.log(num/den)
    return t

    
    

def binary_opt_bayes_decision(π: float, Cfn: float, Cfp: float, LLRs: np.ndarray) -> np.ndarray:
    '''
    
    
    Args:
        π = float
            prior probability
        
        Cfn : float
              C0,1 we are predicting 0, so we are classifing the sample
              as false while the actual class is True. We can denote with C(0) 
              the expected cost for predicting the False Class for the test sample
    
        Cfp : float
              C1,0 we are predicting 1, so we are classifing the sample
              as true while the actual class is False. We can denote with C(1)
              the expected cost for predicting the True Class for the test sample

        LLrs : numpy.ndarray 
               Numpy array that contains the log-likelihood ratio for the test samples
        
    Returns:
        numpy.ndarray: predicted values PVAL
    

    '''
    t = compute_bayes_treshold(π, Cfn, Cfp)
    PVAL = np.zeros(shape = LLRs.shape, dtype = np.int32)  
    PVAL[LLRs > t] = 1
    PVAL[LLRs < t] = 0
    return PVAL

def error_rate_from_M(M: np.ndarray) -> float:
    '''
    Args
        M : np.ndarray
            Confusion matrix
        
    Returns
        error rate : float
        
    '''

    error_rate = (M[0,1] + M[1,0]) / np.sum(M)
    return error_rate
    
    

def compute_multiclass_DCF(M: np.ndarray, Cost: np.ndarray, vPrior: np.ndarray, rounded: bool = True) -> float:
    '''
    

    Args
        M : np.ndarray
            Confusion matrix
        
        Cost : np.ndarray
            Cost matrix
        
        vPrior : np.ndarray
            Column vector of priors probabilities
        
        rounded : bool
            It specifies if the result must be rounded or not

    Returns
        B : float
            Bayes risk for multiclass problem
        
        
    '''
  
    to_sum = []
    for column_idx, c in enumerate(M.T): # i'm iterating on the column
        Nc = sum(c)
        R_C = np.dot(c/Nc ,Cost.T[column_idx, :])
        to_sum.append(vPrior[column_idx, 0] * R_C)  
        # Note: vPrior is a column vector, so if i use vPrior[column_idx, :] it returns me something like [number] as an array
        
    B = sum(to_sum) 
    
    if rounded:
        return round(B, 7)
    
    return B
        
        
    
def compute_binary_DCF(π: float,  Cfn: float, Cfp: float, M: np.ndarray, rounded: bool = True) -> float:
    '''
    

    Args:
        π : float
            Prior probability of the true class.
            
        Cfn : float
              C0,1 we are predicting 0, so we are classifing the sample
              as false while the actual class is True. We can denote with C(0) 
              the expected cost for predicting the False Class for the test sample
    
        Cfp : float
              C1,0 we are predicting 1, so we are classifing the sample
              as true while the actual class is False. We can denote with C(1)
              the expected cost for predicting the True Class for the test sample
       
        M : numpy.ndarray
            Confusion matrix.
        
        rounded : bool
                  It specifies if the result must be rounded or not

    Returns
        B : float
            Bayes risk for binary problem 

    '''
    FNR = M[0,1] / (M[0,1] + M[1,1])
    FPR = M[1,0] / (M[0,0] + M[1,0])
    B = π*Cfn*FNR + ((1-π)*Cfp*FPR)
    
    if rounded:
        return round(B, 7) 
    
    return B
    
def binary_normalized_DCF(DCF: float, π: float,  Cfn: float, Cfp: float, rounded: bool = True) -> float:
     '''
     
     
     Args:
        DCF : float
               unnormalized Detection cost function/bayes_risk.
        
        π : float
            Prior probability of the true class.
            
        Cfn : float
              C0,1 we are predicting 0, so we are classifing the sample
              as false while the actual class is True. We can denote with C(0) 
              the expected cost for predicting the False Class for the test sample
    
        Cfp : float
              C1,0 we are predicting 1, so we are classifing the sample
              as true while the actual class is False. We can denote with C(1)
              the expected cost for predicting the True Class for the test sample
              
        rounded : bool
                  It specifies if the result must be rounded or not
                  
     Returns
         NDCF : float
                normalized DCF

     '''
     B_dummy = min(π*Cfn, (1-π)*Cfp)
     NDCF = DCF / B_dummy
     
     if rounded:
         return round(NDCF, 7)
    
     return NDCF

def compute_min_DCF(π: float,  Cfn: float, Cfp: float, LLRs: np.ndarray, LVAL: np.ndarray) -> float:
    '''
    
    
    Args:
        π = float
            prior probability
        
        Cfn : float
              C0,1 we are predicting 0, so we are classifing the sample
              as false while the actual class is True. We can denote with C(0) 
              the expected cost for predicting the False Class for the test sample
    
        Cfp : float
              C1,0 we are predicting 1, so we are classifing the sample
              as true while the actual class is False. We can denote with C(1)
              the expected cost for predicting the True Class for the test sample
        
        LLrs : numpy.ndarray 
               Numpy array that contains the log-likelihood ratio for the test samples

        LVAL: numpy.ndarray
              Numpy array that contains the true labels of test samples
              
    Returns:
        min(DCF_array) : flaot
    

    '''
    
    range_t = np.concatenate(([-np.inf], LLRs, [np.inf])) #in this case is not a problem to have sorted or not sorted LLRs becouse the function will return the min DCF
   
    DCF_array = []
    for t in range_t:
        PVAL = np.zeros(shape = LLRs.shape, dtype = np.int32)  
        PVAL[LLRs > t] = 1
        PVAL[LLRs < t] = 0
        M = confusion_matrix(PVAL, LVAL)
        DCF = compute_binary_DCF(π, Cfn, Cfp, M, False)
        NDCF = binary_normalized_DCF(DCF, π, Cfn, Cfp, False)
        DCF_array.append(NDCF)
    
    return min(DCF_array)


def ROC_curve(LLRs: np.ndarray, LVAL: np.ndarray):
    '''
    

    Args:
    
        LLrs : numpy.ndarray 
               Numpy array that contains the log-likelihood ratio for the test samples
    
        LVAL: numpy.ndarray
              Numpy array that contains the true labels of test samples
          
    Returns:
        None.

    '''
    x = []
    y = []
    range_t = np.sort(np.concatenate(([-np.inf], LLRs, [np.inf]))) # I need to sort here the scores otherwise i'll otbtain segment between points in the graph 
    
    for t in range_t:
        PVAL = np.zeros(shape = LLRs.shape, dtype = np.int32)  
        PVAL[LLRs > t] = 1
        PVAL[LLRs < t] = 0
        M = confusion_matrix(PVAL, LVAL)
        FNR = M[0,1] / (M[0,1] + M[1,1])
        FPR = M[1,0] / (M[0,0] + M[1,0])
        
        x.append(round(FPR, 2))
        y.append(round((1 - FNR), 2)) #TPR
    

    FPR = np.array(x)
    TPR = np.array(y)
    plt.figure(figsize=(8, 6))
    plt.plot(FPR, TPR, color='blue', lw=3)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.savefig('ROC Curve lab part')
    plt.grid(True)
    plt.show()
    
def compute_effective_prior(π: float,  Cfn: float, Cfp: float) -> float:
    '''
    
    
    Args:
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
        
              
    Returns:
        effective_π: float
        effective prior computed with the formula that links Cfn, Cfp, π genuine class
    

    '''
    
    effective_π = (π*Cfn) / (π*Cfn +(1-π)*Cfp)
    return effective_π
   
def bayes_error_plot(LLRs: np.ndarray, LVAL: np.ndarray, prior_log_odds_range: int, ε: float = 0.001, graph_description: str = 'Bayes error plot', pre_calibration_dcfs: np.ndarray = np.array([])) -> np.ndarray:
    '''
    

    Args:
        LLrs : numpy.ndarray 
               Numpy array that contains the log-likelihood ratio for the test samples
        
        LVAL:  numpy.ndarray
              Numpy array that contains the true labels of test samples
        
        prior_log_odds_range : int
                               Prior log odds is given by the log of effective prior / (1- effective prior)
                               If the argument takes as value 3 for instance, the range will be [-3, 3].
                               
        ε : float
            Pseudocount used for computed the parameters related to the LLRs
            
        graph_description : str
            Description for saving graph figure in local directory
            
        pre_calibration_dcfs : np.ndarray
            Numpy array that contains all the values of normalized DCF pre-calibration for a considered range of applications
            
    Returns:
        
        dcf : np.ndarray
            Numpy array that contains all the values of normalized DCF for the considered range of applications

    '''

    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    
    y1 = []
    y2 = []
    effPriorLogOdds = np.linspace(-prior_log_odds_range, prior_log_odds_range, 21)
    for p in effPriorLogOdds:
        effective_π = 1 / (1+np.exp(-p))
        predictions = binary_opt_bayes_decision(effective_π, 1, 1, LLRs)
        M = confusion_matrix(predictions, LVAL)
        DCF = compute_binary_DCF(effective_π, 1, 1, M, False)
        NDCF = binary_normalized_DCF(DCF, effective_π, 1, 1, False)
        minDCF = compute_min_DCF(effective_π, 1, 1, LLRs, LVAL)
        
        y2.append(NDCF)
        y1.append(minDCF)
        
    dcf = np.array(y2)
    mindcf = np.array(y1)
    
    plt.figure(figsize=(8, 6))
    color = random.choice(tab_colors)
    plt.plot(effPriorLogOdds, dcf, label='actual DCF', color=color)
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', linestyle = '--', color=color)

    if  len(pre_calibration_dcfs) != 0:
        plt.plot(effPriorLogOdds, pre_calibration_dcfs.ravel(), label='actual DCF (pre-cal)', linestyle = ':', color=color)
        
    plt.ylim([0, 1.1])
    plt.xlim([-prior_log_odds_range, prior_log_odds_range])
    plt.xlabel(r'$\log(\frac{\pi}{1 - \pi})$')
    plt.ylabel('DCF')
    plt.legend()
    plt.title(f'{graph_description}')
    plt.savefig(f'{graph_description}')
    plt.show()
    
    return dcf

def plot_multiple_bayes_error_plots(LLRs_list: list, LVAL: np.ndarray, prior_log_odds_range: int, mapping: dict, graph_description: str):
    '''
    
    
    Args:
        LLRs_list : list
        
        LVAL : np.ndarray
            Numpy array that contains the true labels of test samples
            
        prior_log_odds_range : int
            Prior log odds is given by the log of effective prior / (1- effective prior)
            If the argument takes as value 3 for instance, the range will be [-3, 3].
        
        mapping : dict
            Dictionary that maps integers to Models
            
        graph_description : str
            Description for saving graph figure in local directory
    
    Returns:
        None
    '''
    plt.figure(figsize=(8, 6))
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
   
    
    
    for idx, (LLRs, color) in enumerate(zip(LLRs_list, tab_colors[:len(LLRs_list)])):
        y1 = []
        y2 = []
        effPriorLogOdds = np.linspace(-prior_log_odds_range, prior_log_odds_range, 21)
        for p in effPriorLogOdds:
            effective_π = 1 / (1+np.exp(-p))
            predictions = binary_opt_bayes_decision(effective_π, 1, 1, LLRs)
            M = confusion_matrix(predictions, LVAL)
            DCF = compute_binary_DCF(effective_π, 1, 1, M)
            NDCF = binary_normalized_DCF(DCF, effective_π, 1, 1)
            minDCF = compute_min_DCF(effective_π, 1, 1, LLRs, LVAL)
            
            y2.append(NDCF)
            y1.append(minDCF)
            
        dcf = np.array(y2)
        mindcf = np.array(y1)
        plt.plot(effPriorLogOdds, dcf, label=f'actual DCF {mapping[idx]}', color = color)
        plt.plot(effPriorLogOdds, mindcf, label=f'min DCF {mapping[idx]}', linestyle = '--', color = color)

    plt.ylim([0, 1.1])
    plt.xlim([-prior_log_odds_range, prior_log_odds_range])
    plt.xlabel(r'$\log(\frac{\pi}{1 - \pi})$')
    plt.ylabel('DCF')
    plt.grid(True)
    plt.legend()
    plt.title(f'{graph_description}')
    plt.savefig(f'{graph_description}')
    plt.show()


def searching_best_PCA_configuration_for_minDCF(π: float, min_dcf_dict: dict, list_of_models: list) -> list:
    '''
    
    
    Args:
        π : float
            Target application prior probability
            
        min_dcf_dict : dict
                       It is a dictionary that contains for each key list of tuples ( str(model&configuaration), float(minDCF)) 
                       e.g ('MVG (PCA (m= 1))', 0.36864759))
        
        list_of_models : list
                         It contains the name of the models for which we want to find the best PCA configuration for the specified application
                         
    Returns:
        best_conf : list
                    List of tuple (best_m, min_dcf_using_that_m)
    '''
    
    best_conf = []
    
    for model in list_of_models:
        model_list = np.array([value[1] for value in min_dcf_dict[π] if value[0].startswith(model)])
        # With value[1] i'm taking the second element of the tuple (model&configuration, minDCF)
       
        print(model_list)
        # model_list = model_list[1:] use this line if you don't want to consider the configuration without PCA 
        best_conf.append((np.argmin(model_list), np.min(model_list)))  # add +1 to np.argmin(model_list) if you don't want to consider the configuration without PCA
        
    return best_conf
        
    
''' --------------------------------------------------- '''

''' Logistic Regression '''


def expanded_feature_space(DTR: np.ndarray):
    '''
    

    Args
    
    DTR : np.ndarray
          input training data

    Returns
        expanded_DTR : np.ndarray
                       new training data with dimensionality D^2 + D

    '''
    
    output_dim = DTR.shape[0] ** 2 + DTR.shape[0]
    phi_DTR_in = mcol(np.ones(output_dim))
    print('phi_DTR initial shape: ', phi_DTR_in.shape)
    
    for x in DTR.T:
        x = mcol(x)
        xxt = x*x.T
        vec_xxt = (xxt).T.reshape(-1)
        col_vec = mcol(vec_xxt)
        phi_x = np.vstack([col_vec, x])
        phi_DTR = np.hstack([phi_DTR_in, phi_x])
        phi_DTR_in = phi_DTR
    

    expanded_DTR = phi_DTR[:, 1:]
    print('expanded_DTR: ', expanded_DTR.shape)
    return expanded_DTR
           
     
    

def trainLogReg(DTR: np.ndarray, LTR: np.ndarray, λ: float = 0.0):
    '''
    Description : this function embeds the objective function and its optimization.
    
    
    Args:
        DTR : np.ndarray
              input training data  
            
        LTR : np.ndarray
              training labels
        
        λ : float
            regularization coefficient
                         
    Returns:
        min_pos : list
                  it rapresents the estimated position of the minimum
        xf : float
             value of the objtective function at the minumum x 
             
        d : dict
            information returned by scipy.optmize.fmin_l_bfgs_b about the computation done by L-BFGS
                    
    '''
    
    
    def logreg_obj(v):
        '''
        
        Args:
            v : np.ndarray
                numpy array that packs all the model parameters
                             
        Returns:
            (obj, g) : tuple
                       it contains the Logistic Regression objective function and the gradient computed w.r to w and b
                        
        '''
        
        assert v.shape[0] == DTR.shape[0] + 1
        w, b = v[0:-1], v[-1]        
        S = (mcol(w).T @ DTR + b).ravel()
        ZTR = 2 * LTR - 1 # 2c -1 
        log_terms = np.logaddexp(0, -ZTR * S) # log(1 + e^-zi(w^T x + b))
        G = - ZTR / (1.0 + np.exp(ZTR * S)) # G shape: (66,)
        d_w = λ*w + (np.sum(mrow(G)* DTR, axis = 1)/ DTR.shape[1])
        d_b = np.sum(G)/DTR.shape[1]
        g = np.hstack([d_w, d_b]) 
    
        
        obj = (λ/2 * (w@w)) + np.sum(log_terms)/DTR.shape[1]
        return (obj, g)

    min_pos, xf, d = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = np.zeros(DTR.shape[0]+1))
    return (min_pos, np.format_float_scientific(xf, precision=6), d)

def predictions_and_DCF(π: float,  Cfn: float, Cfp: float, llrs: np.ndarray, LVAL: np.ndarray) ->  tuple[float, float]:
    '''
    
    
    Args:
        π : float
            prior probability
        
        Cfn : float
              C0,1 we are predicting 0, so we are classifing the sample
              as false while the actual class is True. We can denote with C(0) 
              the expected cost for predicting the False Class for the test sample
    
        Cfp : float
              C1,0 we are predicting 1, so we are classifing the sample
              as true while the actual class is False. We can denote with C(1)
              the expected cost for predicting the True Class for the test sample
        
        llrs : numpy.ndarray 
               Numpy array that contains the log-likelihood ratio for the test samples

        LVAL : numpy.ndarray
               Numpy array that contains the true labels of test samples
              
    Returns:
        actualDCF : float
        
        minDCF : float
    '''
    
    log_odds = np.log(π/(1-π))
    effective_π = 1 / (1+np.exp(-log_odds))
    predicted_labels = binary_opt_bayes_decision(effective_π, Cfn, Cfp, llrs)
    M = confusion_matrix(predicted_labels, LVAL)    
    DCF = compute_binary_DCF(effective_π, Cfn, Cfp, M)
    actualDCF = binary_normalized_DCF(DCF, effective_π, Cfn, Cfp)
    minDCF = compute_min_DCF(effective_π, Cfn, Cfp, llrs, LVAL)
    #print(f'\nactual DCF: {actualDCF} - minimum DCF: {minDCF}')
    
    return actualDCF, minDCF



def train_and_test_with_different_lambda(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray, π_target: float, min_λ: float, max_λ: float, possible_values: int, weighted: bool, graph_description: str):
    '''


    Args:
        DTR : np.ndarray
              Numpy array that contains the input data
            
        LTR : np.ndarray
              Numpy array that contains the training labels
            
        DVAL : np.ndarray
               Numpy array that contains the validation data
            
        LVAL : np.ndarray
               Numpy array that contains the true labels of test samples
            
        π_target : float
                   Target application prior
            
        min_λ : float
                Smallest considered lambda value
            
        max_λ : float
                Greatest considered lambda value
            
        possible_values : int
                Number of desidered lambda values
        
        weighetd : bool
            Boolean used to decide between the standard LR model or the weighted one.
            
        graph_description : str
            description for saving graph figure in local directory.

    Returns
        None

    '''
    
    λ_values = np.logspace(min_λ, max_λ, possible_values)
    #print('lambda values: ', λ_values)
    
    y1 = []
    y2 = []
    models = []

    plt.figure(figsize=(8, 6))
    
    for λ in λ_values: 
       
        min_pos, J_at_min, info = trainPriorWeightedLogReg(DTR, LTR, λ, π_target) if weighted else trainLogReg(DTR, LTR, λ)
        print(f'\n\nLR with λ coeffiecient equal to: {np.log10(λ)} : min_pos: {min_pos} - f_in_min: {J_at_min} - info: {info}')
        
        optimal_w = min_pos[0:-1]
        optimal_b = min_pos[-1]
        
        
        LPR = compute_LR_log_posterior_ratio(DVAL, optimal_w, optimal_b) # s(xt) = w^T * x_t + b
        π_emp = np.sum(LTR == 1) / LTR.size
        LLR = recover_LLR_from_LPR(LPR, π_target) if weighted else recover_LLR_from_LPR(LPR, π_emp)
        actualDCF, minDCF = predictions_and_DCF(π_target, 1.0, 1.0, LLR, LVAL)
        y1.append(minDCF)
        y2.append(actualDCF)
        
        
        model = {
            'l': np.log10(λ),
            'w': optimal_w,
            'b': optimal_b,
            'LLR': LLR
        }
        models.append(model)
        

    actualDCF = np.array(y2)
    minDCF = np.array(y1)
    
    print(f'Best ActualDCF: {np.min(minDCF)} - Best lambda based on min DCF: {λ_values[np.argmin(minDCF)]}')
    
    plt.plot(λ_values, actualDCF, label='actual DCF')
    plt.plot(λ_values, minDCF, label='min DCF')
    plt.xscale('log', base=10)
    plt.xlabel('λ')
    plt.ylabel('DCF')
    plt.grid(True)
    plt.yticks(np.arange(0.3, 1.1, step=0.1)) 
    plt.legend()
    plt.title(f'Minimum and Actual DCF for different λ values - π: {π_target}')
    plt.title(f'{graph_description}')
    plt.savefig(f'{graph_description}')
    plt.show()
    
    np.save(graph_description.split('-')[0].strip(), np.array(models))
    
    

def trainPriorWeightedLogReg(DTR: np.ndarray, LTR: np.ndarray, λ: float = 0.0, π_target: float = 0.8) -> tuple[list, float, dict]:
    '''
    Description : this function embeds the objective function and its optimization.
    
    
    Args:
        DTR : np.ndarray
              Input training data  
            
        LTR : np.ndarray
              Training labels
        
        λ : float
            Regularization coefficient
        
        π_target : float 
                   Target application prior
                         
    Returns:
        min_pos : list
                  It rapresents the estimated position of the minimum
        xf : float
             Value of the objtective function at the minumum x 
             
        d : dict
            Information returned by scipy.optmize.fmin_l_bfgs_b about the computation done by L-BFGS
    '''
    
    
    
    def logreg_obj(v):
        '''
        
        Args:
            v : np.ndarray
                Numpy array that packs all the model parameters
                             
        Returns:
            (obj, g) : tuple
                       It contains the Logistic Regression objective function and the gradient computed w.r to w and b
        '''
        
        
        assert v.shape[0] == DTR.shape[0] + 1
        w, b = v[0:-1], v[-1]     
        nT = np.sum(LTR == 1)
        nF = np.sum(LTR == 0)

        ε = [π_target/nT if c_i == 1 else (1-π_target)/nF for c_i in LTR]
        S = (mcol(w).T @ DTR + b).ravel()
        ZTR = 2 * LTR - 1 # 2c -1 
        log_terms = np.logaddexp(0, -ZTR * S) # log(1 + e^-zi(w^T x + b)) # log_terms shape: (66, )
        G = - ZTR / (1.0 + np.exp(ZTR * S)) 
        d_w = λ*w + (np.sum(ε*mrow(G)*DTR, axis = 1))
        d_b = np.sum(ε*G)
        g = np.hstack([d_w, d_b]) 
    
        obj = (λ/2 * (w@w)) + np.sum(ε*log_terms)
        return (obj, g)

    min_pos, xf, d = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = np.zeros(DTR.shape[0]+1))
    return (min_pos, np.format_float_scientific(xf, precision=6), d)


def compute_LR_log_posterior_ratio(DVAL: np.ndarray, optimal_w: np.ndarray, optimal_b: float) -> np.ndarray: # s(xt) = w^T * x_t + b
    '''
    
    
    Args:
        DVAL : np.ndarray
               Test data for which we have to compute the scores
            
        optimal_w : np.ndarray
                    W 1-D numpy array found by minimizing the Logistic Regression objective             
        
        optimal_b : float
                    B found by minimizing the Logistic Regression objective
                         
    Returns:
        scores : np.ndarray
                 1-D numpy array that contains the scores for the test samples s(x_t) = w^T * x_t + b
    '''
    
    score = (mcol(optimal_w).T @ DVAL + optimal_b).ravel()
    return score

def recover_LLR_from_LPR(LPR: np.ndarray, π_emp: float) -> np.ndarray:
    '''
    

    Args:
        LPR : np.ndarray
              1-D numpy array that contains the log-posterior ratio for the test samples expressed as scores s(x_t) = w^T * x_t + b
            
        π_emp : float
                Empirical class prior of the training data
                
    Returns:
        LLR : np.ndarray
              1-D numpy array that contains the log-likelihood ratios
    '''
    
    LLR = LPR - np.log(π_emp/(1-π_emp))
    return LLR

''' --------------------------------------------------- '''

''' SVM '''

def compute_extended_training_data(D: np.ndarray, K: int) -> np.ndarray:
    '''
    

    Args
        D : np.ndarray
            Dataset for which it is required the extended version
        
        K : int
            Parameter needed for controlling the regularization of the bias term 

    Returns
        ED : np.ndarray
             Extended matrix of the input data
    '''
    
    K_1d = np.ones(D.shape[1])*K
    ED = np.vstack([D, K_1d])
    return ED

def compute_H_SVM(D: np.ndarray, L: np.ndarray) -> np.ndarray:
    '''
    

    Args
        D : np.ndarray
            Data used for computing H.
        
        L : np.ndarray
            Labels used for computing H.

    Returns
        H : np.ndarray
            Matrix H used in the dual formulation of the SVM objective
        
    '''
    
   
    G_hat = np.dot(D.T, D)
    ZTR = 2 * L - 1 # 2c -1 
    L_hat = np.dot(mcol(ZTR), mrow(ZTR))
    H = L_hat*G_hat
    assert G_hat.shape == L_hat.shape
    
    
    return H 
    


def trainSVM(DTR: np.ndarray, LTR: np.ndarray, H, C: float) -> tuple[list, float, dict]:
    '''
    Description : this function embeds the objective function and its optimization.
    
    
    Args:
        DTR : np.ndarray
              Input training data  
            
        LTR : np.ndarray
              Training labels
              
        H : np.ndarray
            Matrix H used in the dual formulation of the SVM objective
        
        C : float
            Upper bound in the dual formulation constraint
            
            
    Returns:
        min_pos : list
                  it rapresents the estimated position of the minimum
        xf : float
             value of the objtective function at the minumum x 
             
        d : dict
            information returned by scipy.optmize.fmin_l_bfgs_b about the computation done by L-BFGS
                    
    '''
    
    
    def SVM_dual_obj(α: np.ndarray):
        '''
        
        Args:
            α : np.ndarray
                numpy array that packs all the model parameters
                             
        Returns:
            (obj, g) : tuple
                       it contains the Logistic Regression objective function and the gradient computed w.r to a
                        
        '''
        
        n = LTR.shape[0] 
        _1 = np.ones(n)    
        H_α = np.dot(H, α)
        
        d_α = H_α - _1
        obj = (0.5* α.T@H_α) - α.T@_1
        
        return obj, d_α

    min_pos, xf, d = scipy.optimize.fmin_l_bfgs_b(func = SVM_dual_obj, x0 = mcol(np.zeros(DTR.shape[1])), factr= 1.0, bounds = [(0, C) for _ in range(DTR.shape[1])])
    return (min_pos, xf, d)


def retrive_primal(DTR: np.ndarray, LTR: np.ndarray, optimal_α: list, C: float) -> tuple[np.ndarray, float]:
    '''

    Args
        DTR : np.ndarray
              input training data  
        
        LTR : np.ndarray
              training labels    
    
        optimal_α : list
                         optimal alpha values found by solving the dual solution
        
        C : float
            upper bound in the dual formulation constraint
                         

    Returns
        primal : float
                 primal solution retrieved from the dual solution

    '''
   
    
    ZTR = 2 * LTR - 1 # 2c -1 
    w = mcol((optimal_α * ZTR) @ DTR.T)
    hinge_loss = np.maximum(0, (1 - ZTR*(w.T @ DTR)))
    J = (0.5 *  np.linalg.norm(w)**2) + C*np.sum(hinge_loss)
    return w,J


def compute_SVM_scores(DVAL: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''
    
    
    Args
        DVAL : np.ndarray
               Numpy array that contains the validation data
               
        w : np.ndarray
            Numpy array that contains the optimal primal solutions
            
    Returns
        score : np.ndarray
                SVM scores
    '''
    
    
    scores = w.T @ DVAL
    return scores

def compute_kernel_SVM_scores(alphas: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    '''
    
    
    Args
        alphas : np.ndarray
            Numpy array that contains the optimal dual solutions
    
        LTR : np.ndarray
            Labels used for computing H.
    
        DVAL : np.ndarray
               Numpy array that contains the validation data
        
        kernel : np.ndarray
                 Kernel function result
            
    Returns
        score : np.ndarray
                SVM scores
    '''
    
    ZTR = 2 * LTR - 1 # 2c -1
    
    scores = (alphas * ZTR @ kernel)
    return mrow(scores)


def retrieve_kernel(D_i: np.ndarray, D_t: np.ndarray, kernel: str, d: int, c: int, gamma: float, ε: float) -> np.ndarray:
    '''
    

    Args
        D_i : np.ndarray
        
        D_t : np.ndarray

            
        kernel : str
            Name of the used kernel
        
        d : int
            Polynomial degree
            
        c: int
           Polynomial kernel hyperparameter value
            
        gamma : float
             RBF parameter
        
        ε : float
            Cstant value added to the kernel function for reproducing a (regularized) bias
    
        
    Returns
        K : np.ndarray
            Kernel
        
    '''

    if kernel=='poly-d':
        G_hat = np.dot(D_i.T, D_t)
        G_hat = G_hat + c
        K = (G_hat**d) + ε
    else:
        norm= mcol(np.sum(D_i**2, axis=0)) + mrow(np.sum(D_t**2, axis=0))- 2 * np.dot(D_i.T, D_t)
        K = np.exp(-gamma * norm) + ε
     
    return K

def compute_kernel_H_SVM(DTR: np.ndarray, LTR: np.ndarray, kernel: str, d: int, c: int, gamma: float, ε: float) -> np.ndarray:
    '''
    

    Args
        DTR : np.ndarray
            Data used for computing H.
            
        LTR : np.ndarray
            Labels used for computing H.
            
        kernel : str
            Name of the used kernel
            
        d : int
            Polynomial degree
            
        c: int
           Polynomial kernel hyperparameter value
            
        gamma : float
             RBF parameter
             
        ε : float
            Costant value added to the kernel function for reproducing a (regularized) bias
    
        
    Returns
        H : np.ndarray
            Matrix H used in the dual formulation of the SVM objective
        
    '''
    
    K = retrieve_kernel(DTR, DTR, kernel, d, c, gamma, ε) 
        
    ZTR = 2 * LTR - 1 # 2c -1
    L_hat = np.dot(mcol(ZTR), mrow(ZTR))
    H = L_hat*K
        
        
    return H


def train_and_test_SVM(EDTR: np.ndarray, LTR: np.ndarray,EDVAL: np.ndarray, LVAL: np.ndarray, π_target: float, H: np.ndarray, K: int, graph_description: str):
    '''
    

    Args:
    
        EDTR : np.ndarray
               Extended input training data
        
        LTR : np.ndarray
              Training labels
        
        EDVAL : np.ndarray
                Extended validation data
        
        LVAL : np.ndarray
               Validation labels
        
        π_target : float
                   Application prior
        
        H : np.ndarray
            Matrix used for training
        
        K : int
            Additional parameters due to the fact that the code it is considering a slighly modified objective fucntion 
            that leads to the regularization also of the bias and so sub-optimal decision. The effect is mitigate
            by considering x'_i = mcol([x_i, K])
            As K becomes larger, the effects of regularizing b become weaker. However, as K becomes larger, the
            dual problem also becomes harder to solve.
        
        graph_description : str
                            Description for the plotted graph 

    Returns:  
        None

    '''
    
    print('...training and test SVM for different value of C.....')
    
    C_values = np.logspace(-5, 0, 11)
    #print('C_values: ', C_values)
    
    y1 = []
    y2 = []
    models = []

    plt.figure(figsize=(8, 6))
    
    for C in C_values: 
    
        min_pos, L_a, info = trainSVM(EDTR, LTR, H, C)
        print('Dual objective: ', np.format_float_scientific(-L_a, precision=6))

        optW, primal = retrive_primal(EDTR, LTR, min_pos, C)
        print('Primal objective: ', np.format_float_scientific(primal, precision=6))    
      
        duality_gap = primal + L_a
        print('duality_gap: ', np.format_float_scientific(duality_gap, precision = 6))
        
        SVM_score = compute_SVM_scores(EDVAL, optW)  # computing prediction with S > or < 0
        predicted_labels = predict_labels(SVM_score.shape[1], 0, SVM_score)
        _, error_rate = evaluation(predicted_labels - 1, EDVAL.shape[1], LVAL)
        print(f'SVM - K: {K} - C: {C} - Error rate: {error_rate:.2%}\n\n')
        
        actualDCF, minDCF = predictions_and_DCF(π_target, 1.0, 1.0, SVM_score.ravel(), LVAL)
        y1.append(minDCF)
        y2.append(actualDCF)
        
        
        model = {
            'l': np.log10(C),
            'a': min_pos,
            'w': optW,
            'score': SVM_score
        }
        models.append(model)
        
    
    
    actualDCF = np.array(y2)
    minDCF = np.array(y1)
    
    print(f'Best min DCF: {np.min(minDCF)} - Best C based on min DCF: {C_values[np.argmin(minDCF)]}')
    
    plt.plot(C_values, actualDCF, label='actual DCF')
    plt.plot(C_values, minDCF, label='min DCF')
    
    
    plt.xscale('log', base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.grid(True)
    plt.yticks(np.arange(0.3, 1.1, step=0.1)) 
    plt.legend()
    plt.title(f'Minimum and Actual DCF for  C values - π: {π_target}')
    plt.title(f'{graph_description}')
    plt.savefig(f'{graph_description}')
    plt.show()
    
    np.save(graph_description.split('-')[0].strip(), np.array(models))
    
    

def train_and_test_kernel_SVM(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray, π_target: float, K: int,  d: int, c: int, ε: int, kernel_name: str, graph_description: str):
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
               
        π_target : float
                   Application prior
                   
        K : int
            Additional parameters due to the fact that the code it is considering a slighly modified objective fucntion 
            that leads to the regularization also of the bias and so sub-optimal decision. The effect is mitigate
            by considering x'_i = mcol([x_i, K])
            As K becomes larger, the effects of regularizing b become weaker. However, as K becomes larger, the
            dual problem also becomes harder to solve.
        
        d : int
            Degree of the polynomial expantion
        
        c : int 
            Polynomial kernel hyperparameter value
            
        ε : int 
        
        kernel_name : str
                      Name of the kernel that is used (e.g RBF)
        
        graph_description : str
                            Description for the plotted graph 
    
    Returns:
        None
    
    '''
    
    
    print('...training and test kernel SVM for  values of C.....')
    #print('kernel_name: ', kernel_name) 
    
    minC = -3 if kernel_name == 'RBF' else -5
    maxC = 2 if kernel_name == 'RBF' else 0
    C_values = np.logspace(minC, maxC, 11)
    #print('C_values: ', C_values)
    
    

    plt.figure(figsize=(8, 6))
    gamma_values = [0.0]
 
    if kernel_name == 'RBF':      
        gamma_values = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]
    
    for gamma in gamma_values:
        y1 = []
        y2 = []
        models = []
        
        H = compute_kernel_H_SVM(DTR, LTR, kernel_name, d, c, gamma, ε)
        for idx, C in enumerate(C_values): 
           
            min_pos, L_a, info = trainSVM(DTR, LTR, H, C)
            print('Dual objective: ', np.format_float_scientific(-L_a, precision=6))
            
                
            kernel = retrieve_kernel(DTR, DVAL, kernel_name, d, c, gamma, ε)
            SVM_score = compute_kernel_SVM_scores(min_pos, LTR, DVAL, kernel)  # computing prediction with S > or < 0
            predicted_labels = predict_labels(SVM_score.shape[1], 0, SVM_score)
            _, error_rate = evaluation(predicted_labels - 1, DVAL.shape[1], LVAL)
            print(f'SVM - K: {K} - C: {C} - Error rate: {error_rate:.2%}\n\n')
            
            actualDCF, minDCF = predictions_and_DCF(π_target, 1.0, 1.0, SVM_score.ravel(), LVAL)
        
            y1.append(minDCF)
            y2.append(actualDCF)
            
            #print('kernel  saved: ', kernel)
            model = {
                'l': np.log10(C), # "l" can assume values: -3, -2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2,0
                'a': min_pos,
                'score': SVM_score,
                'gamma': gamma if kernel_name == 'RBF' else 0.0,
                'kernel': kernel,
                'kernel_name': kernel_name
            }
            models.append(model)
            
        actualDCF = np.array(y2)
        minDCF = np.array(y1)
        
    
        gamma_str = f'e^{np.log(gamma)}' if len(gamma_values) != 0 else gamma_values[0]
        print(f'γ = {gamma_str} - Best minDCF: {np.min(minDCF)} - Best C based on min DCF: {C_values[np.argmin(minDCF)]}')
        plt.plot(C_values, actualDCF, label=f'actual DCF - γ = {gamma_str}')
        plt.plot(C_values, minDCF, label=f'min DCF - γ = {gamma_str}')
        
        if len(gamma_values) != 0:
            np.save(graph_description.split('-')[0].strip() + ' gamma = e^' + str(np.log(gamma)) , np.array(models))
        
        
      
    
    plt.xscale('log', base=10)
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.grid(True)
    plt.yticks(np.arange(0.3, 1.1, step=0.1)) 
    plt.legend()
    plt.title(f'Minimum and Actual DCF for different C values - π: {π_target}')
    plt.title(f'{graph_description}')
    plt.savefig(f'{graph_description}')
    plt.show()
    
    if len(gamma_values) == 0:
        np.save(graph_description.split('-')[0].strip(), np.array(models))

''' --------------------------------------------------- '''

'''GMM'''

def logpdf_GMM(X: np.ndarray, gmm: list) -> np.ndarray:
    
    '''
    
    Args
        X : np.ndarray
            Input set of samples for computing log-densities containted in a matrix
            
        gmm : list
            List of component parameters
        
    Returns
        logdens : np.ndarray
            computed log density for all samples Xi
    
    '''
    
    S = np.zeros((len(gmm), X.shape[1]))
    for g_idx,g in enumerate(gmm): 
         w, mu, C = g
         S[g_idx, :] = logpdf_GAU_ND(X, mu, C) + np.log(w) 
         
    logdens = scipy.special.logsumexp(S, axis = 0)
    
    return logdens


def cluster_posterior_distributions(X: np.ndarray, gmm: list) -> tuple[np.ndarray, np.ndarray]:
    
    '''
    
    Args
        X : np.ndarray
            Input set of samples for computing log-densities containted in a matrix
            
        gmm : list
            List of component parameters
        
    Returns
       Γ, marginal  : tuple[np.ndarray, np.ndarray]
           Γ is the matrix of posteriorprobabilites /  matrix of responsibilities
           marginal is an array of shape (N, ) whose component i will contain the log density for sample xi 
    '''
    
    S = np.zeros((len(gmm), X.shape[1]))
    for g_idx,g in enumerate(gmm): 
         w, mu, C = g
         S[g_idx, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)

    marginal = mrow(scipy.special.logsumexp(S, axis = 0))
    Γ = np.exp(S - marginal)
    
    return  Γ, marginal


def compute_statistics(X: np.ndarray, γ: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    Zg = np.sum(γ)
    Fg = mrow(np.sum(X * γ, axis = 1))
    Sg = (X * γ)@ X.T    
    
    return Zg, Fg, Sg

def smoothing_cov(Σ: np.ndarray, psi: float = 0.01):
    
    '''
    Args
    
    Σ : np.ndarray
        Input covariance matrix
    
    psi : float
        Lower bound for eigenvalues
    
    Returns:
        newΣ : np.ndarray
        smoothed covariance matrix with bounded eigenvalues
        
    '''
    U, s, _ = np.linalg.svd(Σ)
    s[s<psi] = psi
    newΣ = np.dot(U, mcol(s)*U.T)
    
    return newΣ

def update_gmm_parameters(Zg: np.ndarray, Fg: np.ndarray, Sg: np.ndarray, N: int, variant: str, eigCons: bool, psi: float = 0.01) -> tuple[list, np.ndarray]:
    
    
    '''
    Args
        Zg : np.ndarray
            Zero order statistics for all the g components
            
        Fg : np.ndarray
            First order statistics for all the g components
            
        Sg : np.ndarray
            Second order statistics for all the g components
        
        N : int 
            Number of samples
            
        variant : str
            String that rapresent the GMM variant to use. It can assume a value between 'full', 'diagonal', 'tied'
            
        eigCons : bool
            It decides if smoothing the covariance matrix or not. So it's just a flag for deciding if costraining the eigenvalues or not
        
        psi : float
            Lower bound for eigenvalues
        
        
            
    Returns:
        tied_g_contribution, current_gmm : tuple 
        
        where:
            current_gmm : list 
                new list of GMM parameters obtained by the input statistics, for all the components. 
                So the returned list will be [(w1, μ1, Σ1), (w2, μ2, Σ2) ....] 
            
            tied_g_contribution : np.ndarray
                contribution for the component g-th for the covariance matrix of the Tied Gaussian Model
    
    '''

    μ = mcol(Fg)/Zg
    Σ = Sg/Zg - (μ@μ.T)
    w = Zg / N
 
    if variant == 'diagonal':
        Σ = Σ * np.eye(Σ.shape[0]) 
    
    if eigCons and variant != 'tied':
       Σ = smoothing_cov(Σ)
         
    current_gmm = (w, μ, Σ)

    tied_g_contribution = 0
    if variant == 'tied':
        tied_g_contribution = w*Σ   
    
    return tied_g_contribution, current_gmm

def EM(X: np.ndarray, gmm: list, variant: str, Δ : float,  eigCons: bool, psi: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    '''
    Args
        X : np.ndarray
            Input set of samples for computing log-densities containted in a matrix
            
        gmm : list
            List of component parameters
        
        variant : str
            String that rapresent the GMM variant to use. It can assume a value between 'full', 'diagonal', 'tied'
            
        Δ : float
            EM stopping condition
        
        eigCons : bool
            It decides if smoothing the covariance matrix or not. So it's just a flag for deciding if costraining the eigenvalues or not
        
        psi : float
            Lower bound for eigenvalues
            
    Returns
        opt_μ, opt_Σ, opt_w : tuple[np.ndarray, np.ndarray, np.ndarray]
                tuple that contains the parameters of a GMM that maximize the likelihood for a traning set X
            
        
    '''
    
    assert variant in ['full', 'diagonal', 'tied']
    
    
    ll_t0 = logpdf_GMM(X, gmm).mean()
    ll_t1 = None
    current_gmm = gmm
    iteration = 0 
    
    while True:
        iteration += 1    
        #print(f'------- EM interation n. {iteration} --------------')
        
        # E-step 
        # Γ = matrix of responsabilities for all the componets 
        Γ, marginal = cluster_posterior_distributions(X, current_gmm)
        ll_t1 = logpdf_GMM(X, current_gmm).mean()
       
        
        # Stopping criteria
        if abs(ll_t1 - ll_t0) <= Δ and iteration != 1:    
            #print('final avg ll: ', ll_t1)
    
            return current_gmm 
          
        # M-step
        Zg = np.zeros((Γ.shape[0], 1))
        Fg = np.zeros((Γ.shape[0], X.shape[0]))
        Sg = np.zeros((Γ.shape[0], X.shape[0], X.shape[0]))
        
        current_gmm = []
        tied_matrix_contributions = []
        
        for gidx, γ in enumerate(Γ):
            Zg[gidx], Fg[gidx, :], Sg[gidx, :, :] = compute_statistics(X, γ)
            tied_g_contribution, cur_gmm = update_gmm_parameters(Zg[gidx][0], Fg[gidx, :], Sg[gidx, :, :], X.shape[1], variant, eigCons, psi)
            tied_matrix_contributions.append(tied_g_contribution)
            current_gmm.append(cur_gmm)
    
        assert gmm != current_gmm
        
        tied_gmm = []
        if variant == 'tied':
            #print('tied_matrix_contributions: ', tied_matrix_contributions)
           
            # smoothing covariance after tied update 
            tied_C = smoothing_cov(sum(tied_matrix_contributions))
            #print('tied_C: ', tied_C)
            
            for g in current_gmm:
                w, mu, C = g
                tied_gmm.append((w, mu, tied_C))
                
            current_gmm = tied_gmm
            
       
        #print('ll_t1: ', ll_t1)
        #print('ll_t0: ', ll_t0)
    
        ll_t0 = ll_t1
    
        
        
def split_GMM(gmm: list, alpha: float) -> list:
   
    '''
    Args
        gmm : list
            List of component parameters
        
        alpha: float
            Multiplied factor used for computing the displacemente vector d
    Returns 
        splitted_gmm : list 
            Splitted input G-components GMM into 2G-components GMM 
    '''
    
    
    splitted_gmm = []
    for g_idx,g in enumerate(gmm): 
         w, mu, C = g
         U, s, Vh = np.linalg.svd(C) 
         d = U[:, 0:1] * s[0]**0.5 * alpha
         splitted_gmm.extend([(w/2, mu - d, C), (w/2, mu + d, C)])
    
    return splitted_gmm

def LBG(X: np.ndarray, gmm: list, final_G: int, variant: str, Δ : float, eigCons: bool = False, psi: float = 0.01):
    
    '''
    Args
        X : np.ndarray
            Input set of samples for computing log-densities containted in a matrix
            
        gmm : list
            List of component parameters
        
        final_G : int
            number of desidered components for the final GMM.
            Note: at each iteration the algorithm produces a 2G-components GMM from a G-components GMM given in input
    
        variant : str
            String that rapresent the GMM variant to use. It can assume a value between 'full', 'diagonal', 'tied'
        
        Δ : float
            EM stopping condition
            
        eigCons : bool
            It decides if smoothing the covariance matrix or not. So it's just a flag for deciding if costraining the eigenvalues or not
        
        psi : float
            Lower bound for eigenvalues
            
    Returns
       final_gmm : list
            
    '''
    
    assert variant in ['full', 'diagonal', 'tied']
    
   
    assert final_G % 2 == 0 or final_G == 1
    
    w, mu, C = gmm[0]
    newC = C 
    
    if variant == 'diagonal':
        print('First component - diagonal')
        newC = C * np.eye(C.shape[0]) 
        
    if eigCons and variant != 'tied':
        newC = smoothing_cov(newC, psi)
   
    if variant == 'tied':
        tied_C = w*newC
        newC = smoothing_cov(tied_C, psi)
     
    final_gmm = [(w, mu, newC)]
    
    if final_G == 1:
        return final_gmm
    
    for it in range(int(math.log2(final_G))):
        new_gmm = split_GMM(final_gmm, 0.1)
        final_gmm = EM(X, new_gmm, variant, Δ ,eigCons, psi)
        
    return final_gmm    

def compute_components_range(maxG: int) -> list:
    '''
    
    Args
        maxG : int
            Upper bound for (the range of ) the numer of components (G) for which we want to iterate on 

    Returns
        components_range : list
            range of values rapresenting possible G - values on which we want to itereate on
    '''

    power_list = [2**i for i in range(1, int(math.log2(maxG)) +1) ]
    power_list.append(1)
    components_range = power_list
    components_range.sort()

    return components_range
    
def train_and_test_GMM(DTR: np.ndarray, LTR: np.ndarray, DVAL: np.ndarray, LVAL: np.ndarray, π_target: float, maxG: int, variant: str, Δ: float, eigCons: bool = False, psi: float = 0.01):
    
    '''
    Args
        DTR : np.ndarray
            Input set of training samples 
            
        LTR : np.ndarray
              training labels    
    
        DVAL : np.ndarray
               Numpy array that contains the validation data
               
        LVAL : np.ndarray
               Numpy array that contains the true labels of test samples
        
        π_target : float
            Target application prior

        maxG : int
            number of maximum components for the final GMM.
            Note: at each iteration the algorithm produces a 2G-components GMM from a G-components GMM given in input
    
        variant : str
            String that rapresent the GMM variant to use. It can assume a value between 'full', 'diagonal', 'tied'
            
         Δ : float
             EM stopping condition
        
        eigCons : bool
            It decides if smoothing the covariance matrix or not. So it's just a flag for deciding if costraining the eigenvalues or not
        
        psi : float
            Lower bound for eigenvalues
            
    Returns
       final_gmm : list
            
    '''
    

    components_range = compute_components_range(maxG)
    models = []
    
    print('components_range: ', components_range)
    for M in components_range:
        
        μ0, Σ0 =  compute_mu_C(DTR[:, LTR == 0])
        GMM_0 = LBG(DTR[:, LTR == 0], [(1.0, μ0, Σ0)] , M, variant, Δ, eigCons, psi)
        c0_l_densities = logpdf_GMM(DVAL, GMM_0)
        
        
        μ1, Σ1 = compute_mu_C(DTR[:, LTR == 1])
        GMM_1 = LBG(DTR[:, LTR == 1], [(1.0, μ1, Σ1)] , M, variant, Δ, eigCons, psi)
        c1_l_densities = logpdf_GMM(DVAL, GMM_1)
    
        LLRs = c1_l_densities - c0_l_densities
        actDCF, minDCF = predictions_and_DCF(π_target, 1.0, 1.0, LLRs, LVAL)
        print(f'{variant} GMM - Components: {M}')
        print(f'minDCF/actDCF: {round(minDCF, 7)} / {round(actDCF, 7)}')
        
        model = {
            'variant' : variant,
            'G' : M,
            'Δ' : Δ,
            'psi' : psi,
            'LLR': LLRs,
            'GMM_0': GMM_0,
            'GMM_1': GMM_1,
            
        }
        models.append(model)

        
    np.save(f'{variant} GMM', np.array(models))
    

def fit_density(X: np.ndarray, gmm: list, graph_description: str):
    '''
    
    Args:
        X : np.ndarray
            Input set of samples for computing log-densities
            
        gmm : list
            List of component parameters
    
    Returns:
        None
    '''
    
    plt.figure()
    plt.hist(X.ravel(), bins=30, density=True, edgecolor='black', alpha=0.8, linewidth=0.5, color=(134/255, 95/255, 240/255))
    XPlot = np.linspace(-10, 4, X.shape[1])
    plt.plot(XPlot.ravel(), np.exp(logpdf_GMM(mrow(XPlot), gmm).ravel()), color='orange')
    plt.ylim([0, 0.30])
    plt.title(f'{graph_description}')
    plt.show()
    
def choose_best_based_on_DCF(π: float,  Cfn: float, Cfp: float, LLRs_lists: list, LVAL: np.ndarray, verbose: bool = False) -> tuple[float, float, int]:
    '''
    

    Args:
    
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
        
        LLRs_lists : list
            Collection of log-likelihood ratios
            
        LVAL : np.ndarray
            Labels for the validation data
            
        verbose : bool, optional
            Boolean value used to check if print some comments or not. The default is False

    Returns:
    
        actDCF_for_lowestMinDCF : float
            Value of the ActualDCF associated to the lowest value of Mininum DCF
        
        lowest_minDCF : float
            Lowest value found for the Minimum DCF
            
        list_with_lowest_minDCF : int
            Index that indentifies the list between those received as input, that contains the lowest value of minDCF

    '''
  
    list_with_lowest_minDCF = 0
    lowest_minDCF = np.inf
    actDCF_for_lowestMinDCF = np.inf
    for idx, LLRs in enumerate(LLRs_lists):
        
        actDCF, minDCF = predictions_and_DCF(π, Cfn, Cfp, LLRs, LVAL)    
        if verbose:
            print('minDCF: ', minDCF)
            
        current_list_min = minDCF
        if current_list_min < lowest_minDCF or (current_list_min == lowest_minDCF and actDCF <  actDCF_for_lowestMinDCF):
            lowest_minDCF = current_list_min
            actDCF_for_lowestMinDCF = actDCF
            list_with_lowest_minDCF = idx
        

    return  actDCF_for_lowestMinDCF, lowest_minDCF, list_with_lowest_minDCF

    
def best_performing_candidate_for_method(target_π: float, Cfn: float, Cfp: float, mapping: dict, LVAL: np.ndarray) -> list:
    '''
    

    Args:
    
        target_π : float
            Target prior probability
        
        Cfn : float
            C0,1 we are predicting 0, so we are classifing the sample
            as false while the actual class is True. We can denote with C(0) 
            the expected cost for predicting the False Class for the test sample
        
        Cfp : float
            C1,0 we are predicting 1, so we are classifing the sample
            as true while the actual class is False. We can denote with C(1)
            the expected cost for predicting the True Class for the test sample
        
        mapping : dict
            Dictionary that maps integers to Models
            
        LVAL : np.ndarray
            Labels for the Validation Data

    Returns:
        
        best_candidates : list
            List of the best models found. Each item of the list is a dictionary that contains information about the model and its scores
        

    '''
    
    LR_llrs_list = []
    LR = np.load('Model/LR.npy', allow_pickle=True)
    LLRs_lists = [lr_list['LLR'] for lr_list in  LR]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = LR[idx_best_lr]
    best_lr['model'] = 'LR'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    LR_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the {best_lr["model"]} model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    
    LR_ZScore = np.load('Model/LR (ZScore normalization).npy', allow_pickle=True)
    LLRs_lists = [lr_list['LLR'] for lr_list in  LR_ZScore]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = LR_ZScore[idx_best_lr]
    best_lr['model'] = 'LR - ZScore'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    LR_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the LR ZScore model is {lowest_minDCF} obtained by:\n{best_lr}')
  
    
    
    LR_Centering = np.load('Model/LR (Centering preprocessing).npy', allow_pickle=True)
    LLRs_lists = [lr_list['LLR'] for lr_list in  LR_Centering]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = LR_Centering[idx_best_lr]
    best_lr['model'] = 'LR - Centering'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    LR_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the LR Centering model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    
    Quadratic_LR = np.load('Model/Quadratic LR.npy', allow_pickle=True)
    LLRs_lists = [lr_list['LLR'] for lr_list in  Quadratic_LR]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = Quadratic_LR[idx_best_lr]
    best_lr['model'] = 'Quadratic LR'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    LR_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the Quadratic LR model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    
    
    Weighted_LR = np.load('Model/Weighted LR.npy', allow_pickle=True)
    LLRs_lists = [lr_list['LLR'] for lr_list in  Weighted_LR]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = Weighted_LR[idx_best_lr]
    best_lr['model'] = 'Weighted LR'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    LR_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the Weighted LR model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    
    
    SVM_llrs_list = []
    Linear_SVM = np.load('Model/Linear SVM.npy', allow_pickle=True)
    LLRs_lists = [lr_list['score'].ravel() for lr_list in  Linear_SVM]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = Linear_SVM[idx_best_lr]
    best_lr['model'] = 'Linear SVM'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    SVM_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the Linear SVM model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    
    
    Poly_SVM = np.load('Model/polynomial kernel SVM.npy', allow_pickle=True)
    LLRs_lists = [lr_list['score'].ravel() for lr_list in  Poly_SVM]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = Poly_SVM[idx_best_lr]
    best_lr['model'] = 'Poly SVM'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    SVM_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the PolySVM model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    '''
    IMPORTANT NOTE:
        Since it has resulted, through this analysis (in particular through this function), that the best RBF kernel is found with gamma = e^(-2.0)
        to avoid the uploading of a project of more than 1GB, I decided to throwing away from the "Model" directory the files 
        Model/RBF kernel SVM gamma = e^-4.0.npy,  Model/RBF kernel SVM gamma = e^-2.0.npy, Model/RBF kernel SVM gamma = e^-1.0.npy.
        By running the SVM code it is possible to get them again and the analysis performed with this function will give exactly 
        the same results. Thank you. 
    
    RBF_SVM_4 = np.load('Model/RBF kernel SVM gamma = e^-4.0.npy', allow_pickle=True)
    LLRs_lists = [lr_list['score'].ravel() for lr_list in  RBF_SVM_4]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL, True)
    best_lr = RBF_SVM_4[idx_best_lr]
    best_lr['model'] = 'RBG SVM'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    SVM_llrs_list.append(best_lr)
    
    
    RBF_SVM_3 = np.load('Model/RBF kernel SVM gamma = e^-3.0.npy', allow_pickle=True)
    LLRs_lists = [lr_list['score'].ravel() for lr_list in  RBF_SVM_3]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL, True)
    best_lr = RBF_SVM_3[idx_best_lr]
    best_lr['model'] = 'RBG SVM'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    SVM_llrs_list.append(best_lr)
    
    
    
    RBF_SVM_1 = np.load('Model/RBF kernel SVM gamma = e^-1.0.npy', allow_pickle=True)
    LLRs_lists = [lr_list['score'].ravel() for lr_list in  RBF_SVM_1]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL, True)
    best_lr = RBF_SVM_1[idx_best_lr]
    best_lr['model'] = 'RBG SVM'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    SVM_llrs_list.append(best_lr)
    '''
    
    
    RBF_SVM_2 = np.load('Model/RBF kernel SVM gamma = e^-2.0.npy', allow_pickle=True)
    LLRs_lists = [lr_list['score'].ravel() for lr_list in  RBF_SVM_2]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL, True)
    best_lr = RBF_SVM_2[idx_best_lr]
    best_lr['model'] = 'RBG SVM'
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    SVM_llrs_list.append(best_lr)
    
    
 
    
    #print(f'Lowest minDCF for the RBF kernel SVM model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    
    GMM_llrs_list = []
    full_GMM = np.load('Model/full GMM.npy', allow_pickle=True)
    LLRs_lists = [lr_list['LLR'] for lr_list in full_GMM]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = full_GMM[idx_best_lr]
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    GMM_llrs_list.append(best_lr)
    
    
    #print(f'Lowest minDCF for the full GMM model is {lowest_minDCF} obtained by:\n{best_lr}')
    
    GMM_llrs_list = []
    diagonal_GMM = np.load('Model/diagonal GMM.npy', allow_pickle=True)
    LLRs_lists = [lr_list['LLR'] for lr_list in diagonal_GMM]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, LLRs_lists, LVAL)
    best_lr = diagonal_GMM[idx_best_lr]
    best_lr['minDCF'] = lowest_minDCF
    best_lr['actDCF'] = corresponding_actDCF
    GMM_llrs_list.append(best_lr)
    
    #print(f'Lowest minDCF for the diagonal GMM model is {lowest_minDCF} obtained by:\n{best_lr}')
    llrs_list = [model['LLR'] for model in LR_llrs_list]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, llrs_list, LVAL)
    best_LR_model = LR_llrs_list[idx_best_lr]
    print('\n\n\nBest LR model and configuration: ', best_LR_model)
    
    
    llrs_list = [model['score'].ravel() for model in SVM_llrs_list]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, llrs_list, LVAL)
    best_SVM_model = SVM_llrs_list[idx_best_lr]
    print('\n\n\nBest SVM model and configuration: ', best_SVM_model)
    
    
    llrs_list = [model['LLR'] for model in GMM_llrs_list]
    corresponding_actDCF, lowest_minDCF, idx_best_lr = choose_best_based_on_DCF(target_π, 1.0, 1.0, llrs_list, LVAL)
    best_GMM_model = GMM_llrs_list[idx_best_lr]
    print('\n\n\nBest GMM model and configuration: ', best_GMM_model)
    
    
    best_candidates = [best_LR_model]
    mapping[0] = best_LR_model['model'] + ' λ = 10^' + str(best_LR_model['l'])
    
    best_candidates.append(best_SVM_model)
    mapping[1] = best_SVM_model['model'] + ' γ = e^' + str(np.log(best_SVM_model['gamma'])) + '| C = 10^' + str(best_SVM_model['l'])
    
    best_candidates.append(best_GMM_model)
    mapping[2] = best_GMM_model['variant'] + ' G = ' + str(best_GMM_model['G']) + ' | Δ = ' + str(best_GMM_model['Δ']) + ' | ψ = ' + str(best_GMM_model['psi'])
    
    np.save('bestGMM', best_GMM_model)
    np.save('bestSVM', best_SVM_model)
    np.save('bestLR', best_LR_model)
    
    return best_candidates    
    
'''------------------------------------------------------'''

'''Calibration'''

def evaluate_calibration_model(target_π: float, Cfn: float, Cfp: float, LR_parameters: np.ndarray, cal_validation_scores: np.ndarray, LVAL: np.ndarray) -> tuple[float, float, np.ndarray]:
    
    '''
    Args:
        
        target_π : float
            prior probability for the considered application
        
        Cfn : float
              C0,1 we are predicting 0, so we are classifing the sample
              as false while the actual class is True. We can denote with C(0) 
              the expected cost for predicting the False Class for the test sample
    
        Cfp : float
              C1,0 we are predicting 1, so we are classifing the sample
              as true while the actual class is False. We can denote with C(1)
              the expected cost for predicting the True Class for the test sample
        
        LR_parameters : np.ndarray
            parameters estimated for LR 
    
            
        DVAL : np.ndarray
            Numpy array that contains the validation scores
        

        LVAL : numpy.ndarray
               Numpy array that contains the true labels of test samples
              
    Returns:
        
        (actDCF, minDCF, LLR) : tuple[float, float, np.ndarray]
    
    '''
    
    
   
    optimal_w = LR_parameters[0:-1]
    optimal_b = LR_parameters[-1]
   
    LPR = compute_LR_log_posterior_ratio(cal_validation_scores, optimal_w, optimal_b) # s(xt) = w^T * x_t + b
    LLR = recover_LLR_from_LPR(LPR, target_π)
    actDCF, minDCF = predictions_and_DCF(target_π, 1.0, 1.0, LLR, LVAL)
    return actDCF, minDCF, LLR



def split_k_fold(S: np.ndarray, L: np.ndarray, fold_out: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    '''
    Args:
        S : np.ndarray
            Input score to split with the k-fold
        
        fold_out : int
            Index of the fold that will be used as calibration validaiton set 
        
        k : int
            Desidered folds 
        
    Return:
        (STR, SVAL) : tuple[np.ndarray, np.ndarray]
            tuple of scores that will be used for training the calibration model and validate it
    
    '''
    
  
    STR = np.hstack([S[idx::k] for idx in range(k) if idx != fold_out])
    SVAL = S[fold_out::k]
    LTR = np.hstack([L[idx::k] for idx in range(k) if idx != fold_out])
    LVAL = L[fold_out::k]

    return STR, SVAL, LTR, LVAL


def K_fold(train_π: float, KFOLD: int, RS: np.ndarray, L: np.ndarray):
    
    '''
    Args:
        train_π : float
            Prior used for traning the calibration model
        
        KFOLD : int 
            Number of fold
            
        RS : np.ndarray
            Raw scores
            
        L : np.ndarray
            Set of labels related to the scores
    
    Return:
        
        SCAL, LCAL : tuple[np.ndarray, np.ndarray]
            Tuple of calibrated scores and related labels
        
    '''
  
    
    cal_scores = []
    cal_labels = []
   
    for jdx in range(KFOLD):
        STR, SVAL, LTR, LVAL = split_k_fold(RS, L, jdx, KFOLD)
        
        LR_parameters, _ , _ = trainPriorWeightedLogReg(mrow(STR), LTR, 0.0, train_π)
        _, _, cal_scores_k = evaluate_calibration_model(train_π, 1.0, 1.0, LR_parameters, mrow(SVAL), LVAL)
        cal_scores.append(cal_scores_k)
        cal_labels.append(LVAL)
    
    cal_scores = np.hstack([cs for cs in cal_scores])
    cal_labels = np.hstack([fold_labels for fold_labels in cal_labels])
    
    return cal_scores, cal_labels



def K_fold_fusion(target_π: float, KFOLD: int, RSs: np.ndarray, L: np.ndarray):
    
    '''
    Args:
        target_π : float
            Prior used for traning the calibration model
        
        KFOLD : int 
            Number of fold
            
        RSs : np.ndarray
            Numpy array that contains raw scores of different systems
            
        L : np.ndarray
            Set of labels related to the scores
    
    Return:
        
        SCAL, LCAL : tuple[np.ndarray, np.ndarray]
            Tuple of calibrated scores and related labels
        
    '''
  
    
    SCAL = []
    LCAL = []
   
    for jdx in range(KFOLD):
        STRs = []
        SVALs = []
        for rs in RSs:
            STR, SVAL, LTR, LVAL = split_k_fold(rs, L, jdx, KFOLD)
            STRs.append(STR)
            SVALs.append(SVAL)
        
        LR_parameters, _ , _ = trainPriorWeightedLogReg(np.vstack(STRs), LTR, 0.0, target_π)
        _, _, cal_scores = evaluate_calibration_model(target_π, 1.0, 1.0, LR_parameters, np.vstack(SVALs), LVAL)
        SCAL.append(cal_scores)
        LCAL.append(LVAL)
    
    SCAL = np.hstack([cal_scores for cal_scores in SCAL])
    LCAL = np.hstack([fold_labels for fold_labels in LCAL])
    
    return SCAL, LCAL

def shuffle_scores(model_scores: np.ndarray, L: np.ndarray, seed : int = 42, fusion: bool = False) -> tuple[np.ndarray, np.ndarray]:
    
    '''
    Arg:
        
        model_scores : np.ndarray
            Numpy array that contains typically the raw scores of a model before calibration
            
        L : np.ndarray 
            Scores related label (label i-th refers to the sample i-th for which the model provides the score s-th )
        
        seed : int
            random generator initializer 
            
    Return:
           shuffled_scores, shuffled_labels : tuple[np.ndarray, np.ndarray]
    
    '''
    np.random.seed(seed)
    
    if fusion:
        model_scores = np.vstack([model_scores])
        indices = np.random.permutation(model_scores.shape[1])
        shuffled_scores = model_scores[:, indices]
        shuffled_labels = L[indices]
        shuffled_scores = np.array([scores for scores in shuffled_scores])
    else:
        
        indices = np.random.permutation(len(model_scores))
        shuffled_scores = model_scores[indices]  
        shuffled_labels = L[indices]
        
    
    return np.array(shuffled_scores), np.array(shuffled_labels)

def train_calibration_model(shuffled_scores: np.ndarray, shuffled_labels: np.ndarray, target_π: float, fusion: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    '''
    

    Args:
      
        shuffled_scores : np.ndarray
            
        
        shuffled_labels : np.ndarray
              
            
        target_π : float
            Prior for the target application
           
            
        fusion : bool, optional
                 The default is False.

    Returns:
    
        SCAL : np.ndarray
           Calibrated scores
           
        LCAL : np.ndarray
          Labels related to the Calibrated scores
          
        best_train_π : float
          Best prior used for training  


    '''
    lowest_actDCF = np.inf
    best_train_π = target_π
    best_cal_scores = np.zeros((shuffled_scores.shape))
    corresponding_labels = np.ones((shuffled_labels.shape)) * -1 
    for train_π in np.arange(0.1, 0.6, 0.1):
        cal_scores, cal_labels = K_fold_fusion(train_π, 5, shuffled_scores, shuffled_labels) if fusion else K_fold(train_π, 5, shuffled_scores, shuffled_labels) 
        
        actDCF, minDCF = predictions_and_DCF(target_π, 1.0, 1.0, cal_scores, cal_labels)
        print(f'actDCF: {actDCF}, training_prior: {train_π}')
        if actDCF < lowest_actDCF:
            lowest_actDCF = actDCF 
            best_train_π = train_π
            best_cal_scores = cal_scores
            corresponding_labels = cal_labels
            
    '''
        Once identified the training prior that results in the calibration trasformation that gives the lowest actualDCF for the calibrated scores, 
        it is possible to return the calibrated scores with the corresponding labels and the prior used for training the calibration trasformation
    '''
    
    
    return best_cal_scores, corresponding_labels, best_train_π


def train_calibration_for_fusion(target_π: float, shuffle: bool, fused_scores: list, LVAL: np.ndarray):
    '''
    

    Args:
    
        target_π : float
            
          
        shuffle : bool
            
        
        fused_scores : list
          
        
        LVAL : np.ndarray
            

    Args:
    
        None

    '''

    if shuffle:
        shuffled_scores, shuffled_labels = shuffle_scores(fused_scores, LVAL, 0, True)
        cal_scores, cal_labels, best_train_π = train_calibration_model(shuffled_scores, shuffled_labels, target_π, True)
    else:
        cal_scores, cal_labels, best_train_π = train_calibration_model(np.array(fused_scores), LVAL, target_π, True)
       
    # --- Calibrated scores ----
    
    actDCF, minDCF = predictions_and_DCF(target_π, 1.0, 1.0, cal_scores, cal_labels)
    print(f'Best performing calibration trasformation for Fusion: calibration model trained on : {best_train_π} - Calibrated scores - K-fold - shuffle: {shuffle}--> actual DCF : {round(actDCF, 4)} - minimum DCF: {round(minDCF, 4)}')
    
    str_shuffle = 'shuffled scores' if shuffle else 'unshuffled scores'
    bayes_error_plot(cal_scores, cal_labels, 3, 0.001, f'Bayes error Fusion - Calibrated scores - K-fold - {str_shuffle}')
    
    
    # Train final calibration model with the best prior found with the k-fold procedure
    LR_parameters_final_calibrator, _ , _ = trainPriorWeightedLogReg(np.vstack(fused_scores), LVAL, 0.0, best_train_π)
    print('LR_parameters_final_calibrator: ', LR_parameters_final_calibrator)
    
    model = {
        'model': f'Fusion - {str_shuffle}',
        'SCAL': cal_scores,
        'LCAL': cal_labels,
        'best_train_π': best_train_π,
        'fused_scores': fused_scores if not shuffle else {'scores': shuffled_scores, 'shuffled_labels': shuffled_labels},
        'calibrator_parameters' : LR_parameters_final_calibrator
        
        }
    np.save(f"{model['model']}", model)
    
    
def evaluate_system_on_application(target_π: float, DTR: np.ndarray, LTR: np.ndarray,  DEVAL: np.ndarray, LEVAL: np.ndarray, L: np.ndarray,  system: np.ndarray) -> tuple[np]:
    
    '''
    Args:
        
        target_π : float
            Prior for the target application 
            
        LTR : np.ndarray
            Labels of the training data
        
        DEVAL : np.ndarray
            Evaluation data 
        
        LEVAL : np.ndarray
            Labels for the evaluation data
        
        L : np.ndarray
            Labels related to the score of the validation set
            
        system : np.ndarray
            Already trained system that it is used for computing the scores for the application
            
    Rerurns:
        eval_scores : np.ndarray
            Scores computed on the evaluation set
        
        cal_scores : np.ndarray
            Calibrated scores produced by the input system for the evaluation data
        
        
    '''
    
    if 'GMM' in system.item()['model']:
        GMM_0 = system.item()['GMM_0']
        GMM_1 = system.item()['GMM_1']
        c0_l_densities = logpdf_GMM(DEVAL, GMM_0)
        c1_l_densities = logpdf_GMM(DEVAL, GMM_1)
        eval_scores = c1_l_densities - c0_l_densities
        # LLRs for the evaluation set, are the 'eval_score'
        print('eval_scores: ', eval_scores.shape)
        

    
    elif 'SVM' in system.item()['model']:
        print('SVM')
        
        print('LTR: ', LTR.shape)
        kernel = retrieve_kernel(DTR, DEVAL, system.item()['kernel_name'], 0, 0, system.item()['gamma'], 1.0)
        eval_scores = compute_kernel_SVM_scores(system.item()['a'], LTR, DEVAL,  kernel).ravel() # SVM scores
        
        print('eval_scores: ', eval_scores.shape)
        
        
    elif 'LR' in system.item()['model']:
       
        LPR = compute_LR_log_posterior_ratio(expanded_feature_space(DEVAL), system.item()['w'], system.item()['b']) 
        π_emp = np.sum(LTR == 1) / LTR.size
        print('π_emp: ', π_emp)
        eval_scores = recover_LLR_from_LPR(LPR, π_emp)  # LR scores


    print(" system.item()['best_cal_train_π']: ",  system.item()['best_cal_train_π'])
    print(" system.item()['model']: ", system.item()['model'])
    
    if 'SVM' not in system.item()['model']:
        LR_parameters, _ , _ = trainPriorWeightedLogReg(mrow(system.item()['LLR']), L, 0.0, system.item()['best_cal_train_π']) 
    else:
        LR_parameters, _ , _ = trainPriorWeightedLogReg(mrow(system.item()['score'].ravel()), L, 0.0, system.item()['best_cal_train_π']) 
        print('LR_parameters: ', LR_parameters)
        
    actDCF, minDCF, cal_scores = evaluate_calibration_model(target_π, 1.0, 1.0, system.item()['calibrator_parameters'], mrow(eval_scores), LEVAL)
    print(f"{system.item()['model']} - Calibrated Scores - Evaluation set - actual DCF : {round(actDCF, 4)} - minimum DCF: {round(minDCF, 4)}\n\n")
    bayes_error_plot(cal_scores, LEVAL , 3, 0.001, f"Bayes error plot {system.item()['model']} - Evaluation set - Calibrated scores")
    
                         
    print('type cal_scores: ', type(cal_scores))
    print('shape cal_scores: ', cal_scores.shape)
    return eval_scores, cal_scores

def plot_actual_DCF_error_plots(cal_scores_list: list, LVAL: np.ndarray, prior_log_odds_range: int, mapping: dict, graph_description: str):
    '''
    
    
    Args:
        cal_scores_list : list
        
        LVAL : np.ndarray
            Numpy array that contains the true labels of test samples
            
        prior_log_odds_range : int
            Prior log odds is given by the log of effective prior / (1- effective prior)
            If the argument takes as value 3 for instance, the range will be [-3, 3].
        
        mapping : dict
            Dictionary that maps integers to Models
            
        graph_description : str
            Description for saving graph figure in local directory
    
    Returns:
        None
    '''
    plt.figure(figsize=(8, 6))
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
   
    
    
    for idx, (LLRs, color) in enumerate(zip(cal_scores_list, tab_colors[:len(cal_scores_list)])):
        y2 = []
        effPriorLogOdds = np.linspace(-prior_log_odds_range, prior_log_odds_range, 21)
        for p in effPriorLogOdds:
            effective_π = 1 / (1+np.exp(-p))
            predictions = binary_opt_bayes_decision(effective_π, 1, 1, LLRs)
            M = confusion_matrix(predictions, LVAL)
            DCF = compute_binary_DCF(effective_π, 1, 1, M)
            NDCF = binary_normalized_DCF(DCF, effective_π, 1, 1)
           
            y2.append(NDCF)
            
        dcf = np.array(y2)
        plt.plot(effPriorLogOdds, dcf, label=f'actual DCF {mapping[idx]}', color = color)
        
    plt.ylim([0, 1.1])
    plt.xlim([-prior_log_odds_range, prior_log_odds_range])
    plt.xlabel(r'$\log(\frac{\pi}{1 - \pi})$')
    plt.ylabel('DCF')
    plt.grid(True)
    plt.legend()
    plt.title(f'{graph_description}')
    plt.savefig(f'{graph_description}')
    plt.show()
    

def evaluate_GMMs_on_evaluation(target_π: float,  DEVAL: np.ndarray, LEVAL: np.ndarray,  gmm_model: np.ndarray):
    
    '''
    Args:
        
        target_π : float
            Prior for the target application 
        
        DEVAL : np.ndarray
            Evaluation data 
        
        LEVAL : np.ndarray
            Labels for the evaluation data
        
        gmm_model : np.ndarray
            Already trained system that it is used for computing the scores for the application
            
    Rerurns:
        None
        
    '''
    

    GMM_0 = gmm_model['GMM_0']
    GMM_1 = gmm_model['GMM_1']
    c0_l_densities = logpdf_GMM(DEVAL, GMM_0)
    c1_l_densities = logpdf_GMM(DEVAL, GMM_1)
    eval_scores = c1_l_densities - c0_l_densities
    # LLRs for the evaluation set, are the 'eval_score'
    actDCF, minDCF = predictions_and_DCF(target_π, 1.0, 1.0, eval_scores, LEVAL)
    print(f"{gmm_model['variant']} GMM, G = {gmm_model['G']}  -- evaluation set - Raw scores --> actual DCF : {round(actDCF, 4)} - minimum DCF: {round(minDCF, 4)}")
        



'''----------------------------------------------------'''
