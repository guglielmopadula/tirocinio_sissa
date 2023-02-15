#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from flax import linen as nn

class LBR(nn.Module):
    out_features: None
    
    @nn.compact
    def __call__(self,x):
        x=nn.Dense(self.out_features)(x)
        x=nn.relu(x)
        return x

