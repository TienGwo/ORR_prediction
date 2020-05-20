# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:43:37 2020
Title: 
@author: Dr. Tian Guo
"""

import torch  
import numpy as np 
from utils import MyError, MyLoss, plot_training_rst
from model import Net 
from load_data import build_dataset 

# ========== data load ========== 
flag = "2V" 
X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict = build_dataset(flag) 

# ========== Training parameters setting ========== 
epochs, init_lr = 180, 1e-3 
model = Net(flag) 
cal_loss, cal_error = MyLoss(), MyError() 
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=90, 
                                            gamma=1)

error_rec_train, error_rec_test = [], [] 
for epoch in range(epochs): 
    
    # ========== Training ========== 
    train_loss, train_error = 0, 0
    model.train() 
    index = 0
    for coord, labels in zip(X_data_train, y_data_train): 
        optimizer.zero_grad()                 # empty the paramster gradients 
        outputs = model(coord)                # calculate the outputs 
        loss = cal_loss(outputs, labels)      # calculate the loss
        error = cal_error(outputs, labels)    # calculate the error 
        loss.backward()                       # back propagation 
        optimizer.step()                      # update parameters 
        train_loss += loss.item()             # record the loss value in this batch 
        train_error += error.item()           # record the error value in this batch 
        index += 1 
        
    train_loss_mean, train_error_mean = (train_loss / index), (train_error / index)
    error_rec_train.append(train_error_mean) 
    scheduler.step()  

    # ========== Testing ========== 
    test_loss, test_error = 0, 0 
    model.eval() 
    with torch.no_grad(): 
        index = 0 
        for coord, labels in zip(X_data_test, y_data_test):
            outputs = model(coord)             # calculate the outputs 
            loss = cal_loss(outputs, labels)   # calculate the loss
            error = cal_error(outputs, labels) # calculate the error 
            test_loss += loss.item()           # record the loss value in this batch 
            test_error += error.item()         # record the error value in this batch 
            index += 1
            
    test_loss_mean, test_error_mean = (test_loss / index), (test_error / index)
    error_rec_test.append(test_error_mean)

# ========== prediction & final test ========== 
predicts = [] 
model.eval() 
with torch.no_grad():
    for coord in X_data_predict:
        outputs = model(coord)             # calculate the outputs 
        predicts.append(outputs)

test_rst = [] 
model.eval() 
with torch.no_grad():
    for coord in X_data_test:
        outputs = model(coord)             # calculate the outputs 
        test_rst.append(outputs) 

# ========== print & save results ==========
print(train_error_mean, test_error_mean) 
plot_training_rst(error_rec_train, error_rec_test) 
np.savetxt("../sav/ML_results_"+flag+".txt", np.array(predicts)) 
np.savetxt("../sav/ML_results_"+flag+"_test.txt", np.array(test_rst)) 
torch.save(model.state_dict(), "../sav/atomic_net_"+flag+".pkl")