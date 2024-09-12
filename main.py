# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:32:23 2024

@author: pilli
"""
import utilities as ut
import data_distribution as dd
import pca_and_lda as fr
import probability_densities_and_ml_estimates as mle
import generative_models as gm
import bayes_decisions_models as bdm
import logistic_regression_models as lr
import svm 
import gaussian_mixture_models as gmm
import calibration as cal

def main_menu():
    
    print("\n\nSelect a function:")
    print("1, Data distribution analysis")
    print("2, PCA and LDA analysis")
    print("3, Compute probability densities and ML estimates.")
    print("4, MVG and variants analysis")
    print("5, Bayes Decisions")
    print("6, Logistic Regression analysis")
    print("7, SVM analysis")
    print("8, GMM analysis")
    print("9, Scores calibration")
    print("0, Exit")
    
    choice = input("Function number: ")
    return choice

if __name__ == '__main__':
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            dd.data_distribution()
        elif choice == '2':
            fr.pca_and_lda()
        elif choice == '3':
            mle.probability_densities_and_ml_estimates()
        elif choice == '4':
            gm.generative_models()
        elif choice == '5':
            bdm.bayes_decisions()
        elif choice == '6':
            lr.logistic_regression_analysis()
        elif choice == '7':
            svm.svm_analysis()
        elif choice == '8':
            gmm.gmm_analysis()
        elif choice == '9':
            cal.calibration()
        elif choice == '0':
            print("Exit.")
            break
        else:
            print("Please, insert a valid option.\n\n")
            