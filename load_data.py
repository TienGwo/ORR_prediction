# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:33:20 2020
Title: 
@author: Dr. Tian Guo
"""

import numpy as np 
import xlrd 
import torch 

def load_data(flag, file="../sav/base_data.xlsx"):
    wb = xlrd.open_workbook(file) 
    sheet = wb.sheet_by_index(0) 
    
    num = 27 
    valence_electron = np.array([sheet.cell_value(loopi+1,1) for loopi in range(num)])              # 0
    electronegativity = np.array([sheet.cell_value(loopi+1,2) for loopi in range(num)])             # 1
    wo = np.array([sheet.cell_value(loopi+1,3) for loopi in range(num)])                            # 2 
    Electron_Affinity = np.array([sheet.cell_value(loopi+1,4) for loopi in range(num)])             # 3
    Electron_Affinity_of_Carbon = np.array([sheet.cell_value(loopi+1,5) for loopi in range(num)])   # 4
    d_orbital_of_metal = np.array([sheet.cell_value(loopi+1,6) for loopi in range(num)])            # 5
    period = np.array([sheet.cell_value(loopi+1,7) for loopi in range(num)])                        # 6
    group = np.array([sheet.cell_value(loopi+1,8) for loopi in range(num)])                         # 7
    radius_pm = np.array([sheet.cell_value(loopi+1,9) for loopi in range(num)])                     # 8
    first_ionization_energy = np.array([sheet.cell_value(loopi+1,10) for loopi in range(num)])      # 10
    V_ORR_1V = np.array([sheet.cell_value(loopi+1,11) for loopi in range(14)])                      # 11
    V_ORR_2V = np.array([sheet.cell_value(loopi+1,12) for loopi in range(14)])                      # 12
    
    X_data = np.stack((electronegativity, Electron_Affinity, d_orbital_of_metal, 
                       period, group, radius_pm, first_ionization_energy), axis=0).T
    X_data = torch.from_numpy(X_data) 
    
    if flag == "1V": 
        y_data = torch.from_numpy(V_ORR_1V) 
    else: 
        y_data = torch.from_numpy(V_ORR_2V) 
    
    return X_data, y_data, [d_orbital_of_metal, electronegativity, radius_pm, 
                            group, first_ionization_energy, y_data] 

    
def build_dataset(flag):
    X_data, y_data, _ = load_data(flag)
    
    train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] 
    if flag == "1V":
        test_list = [5, 10]  
    else: 
        test_list = [5, 11]  
    for idx in test_list: 
        train_list.remove(idx)
    
    predict_list = [14+loopi for loopi in range(13)]
    
    X_data_train, y_data_train = [], []
    X_data_test, y_data_test = [], [] 
    X_data_predict = [] 
    for index in range(27):  
        if index in train_list:
            X_data_train.append(X_data[index,:].reshape(1,1,7).float())
            y_data_train.append(y_data[index].reshape(1)) 
            
        if index in test_list: 
            X_data_test.append(X_data[index,:].reshape(1,1,7).float())
            y_data_test.append(y_data[index].reshape(1)) 
            
        if index in predict_list:
            X_data_predict.append(X_data[index,:].reshape(1,1,7).float()) 
            
    return X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict

