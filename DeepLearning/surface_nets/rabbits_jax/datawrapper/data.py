#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:28:55 2023

@author: cyberguli
"""
import meshio
import numpy as np
import copy
from basic_layers.PCA import PCA
import jax
import jax.numpy as jnp
from jax import random

def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    barycenter=np.mean(points,axis=0)
    return points,barycenter


class Data():
    def get_size(self):
        temp_interior,_,_=getinfo(self.string.format(0))
        return ((1,temp_interior.shape[0],3))
    
    def get_reduced_size(self):
        return self.reduced_dimension

    def __init__(
        self,batch_size,num_train,num_test,reduced_dimension,string):
        super().__init__()
        self.batch_size=batch_size
        self.num_train=num_train
        self.num_test=num_test
        self.reduced_dimension=reduced_dimension
        self.string=string
        self.num_samples=self.num_test+self.num_train
        tmp,self.barycenter=getinfo(string.format(0))
        self.barycenter=jnp.array(self.barycenter)
        self.data=np.zeros((self.num_samples,*tmp.shape))
        for i in range(0,self.num_samples):
            if i%100==0:
                print(i)
            self.data[i]=getinfo(self.string.format(i))[0]
        self.data=jnp.array(self.data)    
        self.pca=PCA(self.reduced_dimension)
        self.pca.fit(self.data.reshape(self.num_samples,-1))
        key = random.PRNGKey(0)
        indices = random.permutation(key,self.num_samples)
        self.data_train=self.data[indices[:self.num_train]]
        self.data_test=self.data[indices[self.num_train:]]
        

    def train_dataloader(self,i):
        return self.data_train[i*self.batch_size:self.batch_size*(i+1)]
    def test_dataloader(self,i):
        return self.data_test[i*self.batch_size:self.batch_size*(i+1)]    
