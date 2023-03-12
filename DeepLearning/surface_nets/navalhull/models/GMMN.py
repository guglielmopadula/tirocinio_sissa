#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:41:30 2023

@author: cyberguli
"""

from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.lr import LR
from models.basic_layers.volumenormalizer import VolumeNormalizer
from models.basic_layers.smoother import Smoother
from models.losses.losses import torch_mmd
import torch
import numpy as np

class GMMN(LightningModule):
    
    
    
    class Generator(nn.Module):

        def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,local_indices_1,local_indices_2,drop_prob,reduced_data_shape):
            super().__init__()
            self.data_shape=data_shape
            self.reduced_data_shape=reduced_data_shape
            self.pca=pca
            self.drop_prob=drop_prob
            self.newtriangles_zero=newtriangles_zero
            self.vertices_face_x=vertices_face_x
            self.edge_matrix=edge_matrix
            self.vertices_face_xy=vertices_face_xy
            self.temp_zero=temp_zero
            self.k=k
            self.local_indices_1=local_indices_1
            self.local_indices_2=local_indices_2
            self.fc_interior_1 = LR(latent_dim, hidden_dim)
            self.fc_interior_2 = LR(hidden_dim, hidden_dim)
            self.fc_interior_3 = LR(hidden_dim, hidden_dim)
            self.fc_interior_4 = LR(hidden_dim, hidden_dim)
            self.fc_interior_5 = LR(hidden_dim, hidden_dim)
            self.fc_interior_6 = LR(hidden_dim, hidden_dim)
            self.fc_interior_7 = LR(hidden_dim, hidden_dim)
            self.fc_interior_8 = LR(hidden_dim, hidden_dim)
            self.fc_interior_9 = LR(hidden_dim, hidden_dim)
            self.fc_interior_10 = LR(hidden_dim, hidden_dim)
            self.fc_interior_11 = nn.Linear(hidden_dim, self.reduced_data_shape)
            self.smoother=Smoother(edge_matrix=self.edge_matrix, k=self.k,temp_zero=self.temp_zero, local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
            self.vol_norm=VolumeNormalizer(temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,vertices_face_x=self.vertices_face_x,vertices_face_xy=self.vertices_face_xy,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
            self.relu = nn.ReLU()

        def forward(self,z):
            tmp=self.fc_interior_11(self.fc_interior_10(self.fc_interior_9(self.fc_interior_8(self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z)))))))))))
            x=self.pca.inverse_transform(tmp)
            result_interior=x[:,:np.prod(self.data_shape[0])]
            result_boundary=x[:,np.prod(self.data_shape[0]):]
            result_interior=self.smoother(result_interior,result_boundary)
            result_interior,result_boundary=self.vol_norm(result_interior,result_boundary)
            result_interior=result_interior.reshape(result_interior.shape[0],-1)
            result_boundary=result_boundary.reshape(result_interior.shape[0],-1)
            return torch.concat((result_interior,result_boundary),axis=1)
        

    
    def __init__(self,data_shape,reduced_data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,latent_dim,batch_size,drop_prob,hidden_dim: int= 500,**kwargs):
        super().__init__()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca=pca
        self.drop_prob=drop_prob
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.edge_matrix=edge_matrix
        self.reduced_data_shape=reduced_data_shape
        self.k=k
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.vertices_face_x=vertices_face_x
        self.vertices_face_xy=vertices_face_xy
        self.data_shape = data_shape
        self.generator = self.Generator(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca=self.pca,edge_matrix=self.edge_matrix,vertices_face_x=self.vertices_face_x,vertices_face_xy=self.vertices_face_xy,k=self.k,drop_prob=self.drop_prob,reduced_data_shape=self.reduced_data_shape)


            
    def training_step(self, batch, batch_idx):
        alpha=torch.tensor(0.99999999999)
        x=batch
        z_p_1=torch.randn(len(x), self.latent_dim).type_as(x)
        batch_p_1=self.generator(z_p_1)
        #loss=100*torch.linalg.norm(torch.var(x,axis=0)-torch.var(batch_p_1,axis=0))+0.001*torch.linalg.norm(torch.mean(x,axis=0)-torch.mean(batch_p_1,axis=0))+0.001*torch.linalg.norm(torch.cov(x.T)-torch.cov(batch_p_1.T))
        print(torch.sum(torch.var(batch_p_1,axis=0)))
        loss=100*torch.linalg.norm(torch.var(x,axis=0)-torch.var(batch_p_1,axis=0))+0.001*torch.linalg.norm(torch.mean(x,axis=0)-torch.mean(batch_p_1,axis=0))+(1-alpha)*torch_mmd(x,batch_p_1)
        self.log("train_generator_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x=batch
        z_p_1=torch.randn(len(x), self.latent_dim).type_as(x)
        batch_p_1=self.generator(z_p_1)
        loss=torch.linalg.norm(torch.mean(x,axis=0)-torch.mean(batch_p_1,axis=0))+torch.linalg.norm(torch.cov(x.T)-torch.cov(batch_p_1.T))
        self.log("train_generator_loss", loss)
        return loss

        

    def configure_optimizers(self): 
        optimizer_gen = torch.optim.AdamW(self.generator.parameters(), lr=1e-4) #0.0002
        return [optimizer_gen], []

    def sample_mesh(self,mean=None,var=None):
        device=self.generator.pca._V.device
        self=self.to(device)
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim)

        if var==None:
            var_1=torch.ones(1,self.latent_dim)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim)+mean_1
        z=z.to(device)
        tmp=self.generator(z)
        return tmp
     
