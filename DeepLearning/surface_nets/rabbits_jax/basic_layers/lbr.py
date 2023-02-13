#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from flax import linen as nn

class LBR(nn.Module):
    in_features: None
    out_features: None
    drop_prob:None

    def setup(self):
        super().__init__()
        self.lin=nn.Dense(self.batchout_features)
        self.batch=nn.BatchNorm()
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(self.drop_prob)
    
    def __call__(self,x):
        return self.dropout(self.relu(self.batch(self.lin(x))))


