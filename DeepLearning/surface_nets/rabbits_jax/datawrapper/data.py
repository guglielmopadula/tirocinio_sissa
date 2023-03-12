#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:28:55 2023

@author: cyberguli
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import pickle
from utils.pca import PCA
import meshio

class Data():
    
    def get_size(self):
        return self.red_dim

    def __init__(
        self,batch_size,num_train,num_test,string,red_dim):
        super().__init__()
        self.num_train=num_train
        self.red_dim=red_dim
        self.num_test=num_test
        self.batch_size=batch_size
        self.pca=PCA(self.red_dim)
        self.num_samples=self.num_train+self.num_test
        key = random.PRNGKey(0)
        tmp=meshio.read(string.format(0)).points
        self.barycenter=jnp.array(np.mean(tmp,axis=0))
        self.data=np.zeros((self.num_samples,tmp.reshape(-1).shape[0]))
        for i in range(self.num_samples):
            self.data[i]=meshio.read(string.format(i)).points.reshape(-1)
        self.data=jnp.array(self.data)
        self.pca.fit(self.data)
        indices = random.permutation(key,self.num_samples)
        self.data_train=self.data[indices[:self.num_train]]
        self.data_test=self.data[indices[self.num_train:]]
        self.size=self.data.shape[1]
        

    def train_dataloader(self,i):
        return self.data_train[i*self.batch_size:self.batch_size*(i+1),:]
    def test_dataloader(self,i):
        return self.data_test[i*self.batch_size:self.batch_size*(i+1),:]    
