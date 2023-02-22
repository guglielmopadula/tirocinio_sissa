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
    def __call__(self,x, training):
        x=nn.Dense(self.out_features)(x)
        x=nn.relu(x)
        x=nn.LayerNorm()(x)
        x = nn.Dropout(rate=0.01, deterministic=not training)(x)
        return x

