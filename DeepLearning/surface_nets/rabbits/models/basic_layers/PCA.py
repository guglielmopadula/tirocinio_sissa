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
        self.mean=torch.mean(matrix,dim=0).to(matrix.device)
        self._mean_matrix=torch.mm(torch.ones(self._n,1).to(matrix.device),self.mean.reshape(1,self._p)).to(matrix.device)
        X=matrix-self._mean_matrix
        _,S,tmp=torch.linalg.svd(X,full_matrices=False)
        self._V=tmp.T
        self._V=self._V[:,:self._reduced_dim].to(matrix.device)
        
    def transform(self,matrix):
        return torch.matmul(matrix-self._mean_matrix[:matrix.shape[0],:],self._V)
    
    def inverse_transform(self,matrix):
        return torch.matmul(matrix,self._V.t())+self._mean_matrix[:matrix.shape[0],:]    