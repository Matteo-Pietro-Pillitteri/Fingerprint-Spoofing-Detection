# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:28:20 2024

@author: pilli
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut


def fitting_density(dataset, labels, feature, name):
   
    D0,D1 = ut.create_mask_binary(dataset, labels)
   
    m_true, C_true = ut.compute_mu_C(ut.mrow(D1[feature, :]))
    m_false, C_false = ut.compute_mu_C(ut.mrow(D0[feature, :]))
   
    plt.figure()
    
    plt.hist(D0[feature, :], bins = 10, density= True,  alpha=0.4, edgecolor='0.4', label='Fake Class')
    plt.hist(D1[feature, :], bins = 10, density= True,  alpha=0.4, edgecolor='0.4', label='True Class')
    XPlot = np.linspace(-4.5,4.5,3000)
    plt.plot(XPlot.ravel(), np.exp(ut.logpdf_GAU_ND(ut.mrow(XPlot), m_false, C_false)), color='mediumblue', label='density_fake')
    plt.plot(XPlot.ravel(), np.exp(ut.logpdf_GAU_ND(ut.mrow(XPlot), m_true, C_true)), color='darkorange', label='density_true')
    plt.legend()
    plt.title(f'{name}')
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig(f'{name}')
    plt.show()
    
 
def probability_densities_and_ml_estimates():
    D, L = ut.load()
    
    print(f'L shape: {L.shape}')
    print(f'D shape: {D.shape}')
      
    for feature in range(6):
        name = 'Density fit on feature num {f}'
        fitting_density(D, L, feature, name.format(f=feature))
       
    
    
    