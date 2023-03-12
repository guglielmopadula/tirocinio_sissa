#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:34:37 2023

@author: cyberguli
"""
import torch
class PCA():
    def __init__(self,reduced_dim):
        self._reduced_dim=reduced_dim
        
    def fit(self,matrix):
        self._n=matrix.shape[0]
        self._p=matrix.shape[1]
        mean=torch.mean(matrix,dim=0)
        self._mean_matrix=torch.mm(torch.ones(self._n,1,device=matrix.device),mean.reshape(1,self._p))
        X=matrix-self._mean_matrix
        Cov=torch.matmul(X.t(),X)/self._n
        self._V,S,_=torch.linalg.svd(Cov)
        self._V=self._V[:,:self._reduced_dim]
        
    def transform(self,matrix):
        return torch.matmul(matrix-self._mean_matrix[:matrix.shape[0],:],self._V)
    
    def inverse_transform(self,matrix):
        return torch.matmul(matrix,self._V.t())+self._mean_matrix[:matrix.shape[0],:]
    
    def to(self,device):
        self._V=self._V.to(device)
        self._mean_matrix=self._mean_matrix.to(device)