# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:53:25 2020
Title: 
@author: Dr. Tian Guo
"""

import numpy as np 
import matplotlib.pyplot as plt 
from load_data import load_data 
from scipy import optimize

def descriptor(theta_d, E_M, r_M, g, IE): 
    E_C, r_C, Eea_C = 2.55, 170, 1.263 
    result = (IE * theta_d * g) * ((E_M  / r_M) + (E_C / r_C))
    return result  


def load_results(flag):
    _, _, [electronegativity, d_orbital_of_metal, 
           group, radius_pm, first_ionization_energy, DFT_data] = load_data(flag) 
    predicts = np.loadtxt("../sav/ML_results_"+flag+"_preditions.txt")
    
    desc_DFT, desc_ML = [], [] 
    V_ORR_ML = []
    index = 0
    for theta_d, E_M, r_M, g, IE in zip(d_orbital_of_metal, electronegativity, 
                                        radius_pm, group, first_ionization_energy): 
        if index <= 13: 
            desc_DFT.append(descriptor(theta_d, E_M, r_M, g, IE)) 
        else:
            desc_ML.append(descriptor(theta_d, E_M, r_M, g, IE)) 
            V_ORR_ML.append(predicts[index-14])
        index += 1
        
    return desc_DFT, DFT_data, desc_ML, V_ORR_ML
    

def fit_curve(flag): 
    
    def f_1(x, A, B):
        return A * x + B
    
    desc_DFT, DFT_data, desc_ML, V_ORR_ML = load_results(flag) 
    desc_DFT = np.array(desc_DFT) 
    results_DFT = np.stack([desc_DFT, DFT_data]) 
    desc_ML = np.array(desc_ML) 
    results_ML = np.stack([desc_ML, V_ORR_ML]) 
    results = np.hstack([results_DFT, results_ML]) 
    results = results.T[np.lexsort(results[::-1,:])].T
    
    if flag == "1V":
        num = -6
    else:
        num = -8 
        
    X_up = results[0, 0:num+1] 
    Y_up = results[1, 0:num+1] 
    X_down = results[0, num::] 
    Y_down = results[1, num::] 

    A_up, B_up = optimize.curve_fit(f_1, X_up, Y_up)[0] 
    x_up = np.arange(0, results[0,num], 1) 
    y_up = A_up * x_up + B_up
    
    A_down, B_down = optimize.curve_fit(f_1, X_down, Y_down)[0] 
    x_down = np.arange(results[0,num], results[0,-1], 1) 
    y_down = A_down * x_down + B_down
    return x_up, y_up, x_down, y_down


if __name__ == "__main__" : 
    flag = "2V" 
    desc_DFT, DFT_data, desc_ML, V_ORR_ML = load_results(flag) 
    x_up, y_up, x_down, y_down = fit_curve(flag) 
    
    fontsize = 20
    plt.figure(figsize=(10,9)) 
    plt.plot(desc_DFT, DFT_data, "s", markersize=10, color="b") 
    plt.plot(desc_ML, V_ORR_ML, "^", markersize=10, color="r") 
    plt.plot(x_up, y_up, "--", color="k") 
    plt.plot(x_down, y_down, "--", color="k") 
    plt.xlabel("Descriptor, $\phi$", fontsize=fontsize) 
    plt.ylabel("$\eta^{ORR}$", fontsize=fontsize+4)
    plt.ylim([-3.0, -0.2]) 
    plt.legend(["DFT calc", "ML pred"], fontsize=fontsize+4)
    plt.xticks(fontsize=fontsize) 
    plt.yticks(fontsize=fontsize) 
    plt.show()     
    
    desc = np.hstack([np.array(desc_DFT), np.array(desc_ML)])    
    np.savetxt("../sav/descriptors.txt", desc)
    # np.savetxt("../sav/fit_"+flag+"_up.txt", np.stack([x_up, y_up]).T)
    # np.savetxt("../sav/fit_"+flag+"_down.txt", np.stack([x_down, y_down]).T) 
    
    