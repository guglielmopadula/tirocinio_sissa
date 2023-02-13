#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""
from flax import linen as nn
import jax

import jax.numpy as jnp
class Barycentre():
    def __init__(self,batch_size,barycenter):
        self.barycenter=barycenter
        self.batch_size=batch_size
    @jax.jit
    def apply(self,x):
        x=x.reshape(x.shape[0],-1,3)
        return x-jnp.expand_dims(jnp.mean(x,axis=1),1).repeat(1,x.shape[1],1)+jnp.expand_dims(jnp.expand_dims(self.barycenter)).repeat(x.shape[0],x.shape[1],1)


