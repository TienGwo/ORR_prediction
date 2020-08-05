# coding: utf-8 

import torch 
from load_data import build_dataset 
from utils import MyError, MyLoss 
from model import Net 

def train_func(flag): 
    ''' 
    This function is to train the network. 
    ''' 
    # ========== Training parameters setting ========== 
    X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict = build_dataset(flag) 
    epochs, init_lr = 120, 1e-3 
    model = Net(flag) 
    cal_loss, cal_error = torch.nn.MSELoss(), MyError() 
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr) 
    
    # ========== start training ========== 
    loss_rec_train, loss_rec_test = [], [] 
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
        loss_rec_train.append(train_loss_mean)
        error_rec_train.append(train_error_mean) 
        
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
        loss_rec_test.append(test_loss_mean) 
        error_rec_test.append(test_error_mean)
    
    return loss_rec_train, error_rec_train, loss_rec_test, error_rec_test, model