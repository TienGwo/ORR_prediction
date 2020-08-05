# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:40:36 2020
Title: 
@author: Dr. Tian Guo
"""

import torch  
import torch.nn.functional as F 
from utils import Conv1d_same_padding 
from load_data import load_data

class Net(torch.nn.Module): 
    def __init__(self, flag): 
        super(Net, self).__init__() 
        X_data, _, _ = load_data("1V")
        self.conv_0 = Conv1d_same_padding( 4, 16, 1)
        self.conv_1 = Conv1d_same_padding(16, 16, 1) 
        self.conv_2 = Conv1d_same_padding(16, 16, 1) 
        self.conv_3 = Conv1d_same_padding(16, 16, 1) 
        self.conv_4 = Conv1d_same_padding(16,  8, 1) 
        self.dense_0 = torch.nn.Linear(40, 32, bias=False) 
        self.dense_1 = torch.nn.Linear(32, 16, bias=False) 
        self.dense_2 = torch.nn.Linear(16, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.dense_0.weight) 
        torch.nn.init.xavier_uniform_(self.dense_1.weight) 
        torch.nn.init.xavier_uniform_(self.dense_2.weight) 
        
        self.data = X_data 
        self.flag = flag 
        
    def forward(self, x): 
        if self.flag == "1V":
            label_0 = self.data[ 3,:].reshape(1,1,5).float()   # 1V
            label_1 = self.data[ 4,:].reshape(1,1,5).float()   # 1V
            label_2 = self.data[13,:].reshape(1,1,5).float()   # 1V 
        else:
            label_0 = self.data[ 3,:].reshape(1,1,5).float()   # 2V
            label_1 = self.data[11,:].reshape(1,1,5).float()   # 2V
            label_2 = self.data[13,:].reshape(1,1,5).float()   # 2V 
        net = torch.cat([x, label_0, label_1, label_2], dim=1) 
        
        net = F.relu(self.conv_0(net))  
        net = F.relu(self.conv_1(net))  
        net = F.relu(self.conv_2(net))  
        net = F.relu(self.conv_3(net))  
        net = F.relu(self.conv_4(net))  
        net = torch.flatten(net)  
        
        net = F.relu(self.dense_0(net)) 
        # net = self.dropout_0(net)
        net = F.relu(self.dense_1(net)) 
        # net = self.dropout_1(net)
        outputs = self.dense_2(net) 
        return outputs 


if __name__ == "__main__":
    model = Net("1V") 
    coord_tensor = torch.randn([1,1,5]) 
    result = model(coord_tensor) 
    print(result.shape) 