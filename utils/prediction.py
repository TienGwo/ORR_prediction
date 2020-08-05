# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:07:20 2020
Title: 
@author: Dr. Tian Guo
"""

import torch  
import numpy as np 
from utils import MyError, MyLoss, plot_training_rst
from model import Net 
from load_data import build_dataset, load_data  


# ========== data load ========== 
flag = "1V" 
X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict = build_dataset(flag) 
model = Net(flag) 
model.load_state_dict(torch.load("../sav/atomic_net_"+flag+".pkl")) 

predicts = [] 
model.eval() 
with torch.no_grad():
    for coord in X_data_predict:
        outputs = model(coord)             # calculate the outputs 
        predicts.append(outputs)

print(predicts) 


from results_analysis import descriptor 

_, _, [d_orbital_of_metal, electronegativity, 
       radius_pm, group, first_ionization_energy, DFT_data] = load_data(flag) 

desc_ML = []
for theta_d, E_M, r_M, g, IE in zip(d_orbital_of_metal, electronegativity, radius_pm, group, first_ionization_energy):
    desc_ML.append(descriptor(theta_d, E_M, r_M, g, IE))

