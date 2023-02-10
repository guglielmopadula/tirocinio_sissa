#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:28:55 2023

@author: cyberguli
"""
import torch
import meshio
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import copy
from models.basic_layers.PCA import PCA
from torch.utils.data import random_split

def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    barycenter=np.mean(points,axis=0)
    return torch.tensor(points),torch.tensor(barycenter)


class Data(LightningDataModule):
    def get_size(self):
        temp_interior,_,_,_,_=getinfo(self.string.format(0))
        return ((1,temp_interior.shape[0],3))
    
    def get_reduced_size(self):
        return self.reduced_dimension

    def __init__(
        self,batch_size,num_workers,num_train,num_test,reduced_dimension,string,use_cuda):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.num_train=num_train
        self.num_workers = num_workers
        self.num_test=num_test
        self.reduced_dimension=reduced_dimension
        self.string=string
        self.num_samples=self.num_test+self.num_train
        tmp,self.barycenter=getinfo(string.format(0))
        self.data=torch.zeros(self.num_samples,*tmp.shape)
        for i in range(0,self.num_samples):
            if i%100==0:
                print(i)
            self.data[i],_=getinfo(self.string.format(i))
        self.pca=PCA(self.reduced_dimension)
        if use_cuda:
            self.pca.fit(self.data.reshape(self.num_samples,-1).cuda())
            self.barycenter=self.barycenter.cuda()
        else:
            self.pca.fit(self.data.reshape(self.num_samples,-1))

        self.data_train,self.data_test = random_split(self.data, [self.num_train,self.num_test])    

    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
