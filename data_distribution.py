# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:05:53 2024

@author: pilli
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import utilities as ut

 

def data_distribution():
    D, L = ut.load()
    
    
    mu_fake = ut.dataset_mean(D[:, L == 0])
    mu_true = ut.dataset_mean(D[:, L == 1])
    print(f'Mean fake class: {mu_fake} \n Mean true class: {mu_true} ')
    
    var_fake = ut.mcol(D[:, L == 0].var(1))
    var_true = ut.mcol(D[:, L == 1].var(1))
    print(f'Var fake class: {var_fake} \nVar true class: {var_true}')
    
    
    for feature in range(6):
        ut.computation_and_plot(D, L, feature)
       
    
        