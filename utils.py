# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:37:03 2020
Title: 
@author: Dr. Tian Guo
"""

import torch  
import torch.nn.functional as F 
import matplotlib.pyplot as plt 

class MyLoss(torch.nn.Module): 
    def __init__(self):
        super(MyLoss, self).__init__() 
        
    def forward(self, pred, real): 
        return torch.mean(torch.abs(torch.sub(pred, real)))
    
    
class MyError(torch.nn.Module): 
    def __init__(self):
        super(MyError, self).__init__() 
        
    def forward(self, pred, real): 
        return torch.mean(torch.abs(torch.sub(pred, real) / real)) 
    
    
class Conv1d_same_padding(torch.nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=1, dilation=1):
        super(Conv1d_same_padding, self).__init__() 
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.conv1d = torch.nn.Conv1d(inplanes, planes, kernel_size, strides, bias=False) 
        torch.nn.init.xavier_uniform_(self.conv1d.weight)

    def forward(self, x): 
        input_rows = x.size(2)
        out_rows = (input_rows + self.strides - 1) // self.strides
        padding_rows = max(0, (out_rows - 1) * self.strides + (self.kernel_size - 1) * self.dilation + 1 - input_rows) 
        x = F.pad(x, pad=(0, padding_rows), mode="constant") 
        outputs = self.conv1d(x)        
        return outputs
    

def plot_training_rst(error_rec_train, error_rec_test):
    plt.figure(figsize=(10,9)) 
    plt.plot(error_rec_train, "-o", linewidth=3, markersize=8) 
    plt.plot(error_rec_test, "-o", linewidth=3, markersize=8) 
    plt.ylim([0,1]) 
    plt.legend(["Train error", "Test error"], fontsize=20) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Epoch", fontsize=18) 
    plt.ylabel("Error", fontsize=18) 
    plt.grid(which="both", linestyle="--", linewidth=2, axis='y')
    plt.show() 