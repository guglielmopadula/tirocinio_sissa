#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:41:30 2023

@author: cyberguli
"""

from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.encoder import Encoder_base
from models.basic_layers.decoder import Decoder_base
from models.losses.losses import L2_loss
import torch


class BEGAN(LightningModule):
    
    
    class Generator(nn.Module):

        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2,drop_prob):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k,drop_prob=drop_prob)

        def forward(self,x,y):
            return self.decoder_base(x,y)
        
    class Discriminator(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2,drop_prob):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2,drop_prob=drop_prob)
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k,drop_prob=drop_prob)
             
        def forward(self,x,y):
            x_hat,y_hat=self.decoder_base(*self.encoder_base(x,y))
            return x_hat,y_hat
         
    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,latent_dim_1,latent_dim_2,batch_size,drop_prob,hidden_dim: int= 300,**kwargs):
        super().__init__()
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.drop_prob=drop_prob
        self.edge_matrix=edge_matrix
        self.k=k
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.discriminator = self.Discriminator(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k,drop_prob=self.drop_prob)
        self.generator = self.Generator(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k,drop_prob=self.drop_prob)


        
    def forward(self, x):
        x_hat=self.discriminator(x)
        return x_hat.reshape(x.shape)
    
    def disc_loss(self, x,y):
        x_hat,y_hat=self.discriminator(x,y)
        loss=0.5*L2_loss(x, x_hat.reshape(x.shape)).mean()+0.5*L2_loss(y, y_hat.reshape(y.shape)).mean()
        return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx ):
        x,y=batch
        z_p_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_p_2=torch.randn(len(y), self.latent_dim_2).type_as(y)


        batch_p_1,batch_p_2=self.generator(z_p_1,z_p_2)
        
        

        
        if optimizer_idx==0:
            loss=self.disc_loss(batch_p_1,batch_p_2)
            self.log("train_generator_loss", loss)
            return loss
        

        if optimizer_idx==1:
            gamma=torch.tensor(0.5)
            k=torch.tensor(0.)
            lambda_k = torch.tensor(0.001)
            z_d_1,z_d_2=self.discriminator.encoder_base(x,y)
            z_d_1=z_d_1.reshape(len(x),self.latent_dim_1)
            z_d_2=z_d_2.reshape(len(y),self.latent_dim_2)
            batch_d_1,batch_d_2=self.generator(z_d_1,z_d_2)

            tmp=self.disc_loss(x,y)
            loss_disc=tmp-k*self.disc_loss(batch_d_1,batch_d_2)
            loss_gen=self.disc_loss(batch_p_1,batch_p_2)
            self.log("train_discriminagtor_loss", loss_disc)
            k = torch.min(torch.max( k + lambda_k * torch.mean(gamma * tmp - loss_gen), torch.tensor(0)), torch.tensor(1))
            return loss_disc
        
    def validation_step(self, batch, batch_idx):
        x,y=batch
        self.log("val_began_loss", self.disc_loss(x,y))
        return self.disc_loss(x,y)

        
    def test_step(self, batch, batch_idx):
        x,y=batch
        self.log("test_began_loss", self.disc_loss(x,y))
        return self.disc_loss(x,y)
        

    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer_gen = torch.optim.AdamW(self.generator.parameters(), lr=0.00002) 
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=0.00005) 
        return [optimizer_gen,optimizer_disc], []

    def sample_mesh(self):
        device=self.generator.decoder_base.pca_1._V.device
        self=self.to(device)
        mean_1=torch.zeros(1,self.latent_dim_1)
        mean_2=torch.zeros(1,self.latent_dim_2)
        var_1=torch.ones(1,self.latent_dim_1)
        var_2=torch.ones(1,self.latent_dim_2)
        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        w=w.to(device)
        z=z.to(device)
        temp_interior,temp_boundary=self.generator(z,w)
        return temp_interior,temp_boundary
