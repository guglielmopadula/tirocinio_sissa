#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn


class LBR(nn.Module):
    def __init__(self,in_features,out_features,drop_prob):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.Sigmoid()
        self.dropout=nn.Dropout(drop_prob)
    
    def forward(self,x):
        x=self.lin(x)
        x=self.batch(x)
        x=self.relu(x)
        x=self.dropout(x)
        return x


