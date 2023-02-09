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

        def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,local_indices_1,local_indices_2,drop_prob,reduced_data_shape):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca=pca,edge_matrix=edge_matrix,vertices_face_x=vertices_face_x,vertices_face_xy=vertices_face_xy,k=k,drop_prob=drop_prob,reduced_data_shape=reduced_data_shape)

        def forward(self,x):
            return self.decoder_base(x)
        
    class Discriminator(nn.Module):

        def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,local_indices_1,local_indices_2,drop_prob,reduced_data_shape):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca=pca,edge_matrix=edge_matrix,vertices_face_x=vertices_face_x,vertices_face_xy=vertices_face_xy,k=k,drop_prob=drop_prob,reduced_data_shape=reduced_data_shape)
            self.encoder_base=Encoder_base(latent_dim=latent_dim,hidden_dim=hidden_dim, reduced_data_shape=reduced_data_shape,pca=pca,drop_prob=drop_prob)

        def forward(self,x):
            return self.decoder_base(self.encoder_base(x))

    
    def __init__(self,data_shape,reduced_data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,latent_dim,batch_size,drop_prob,hidden_dim: int= 300,**kwargs):
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
        self.discriminator = self.Discriminator(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca=self.pca,edge_matrix=self.edge_matrix,vertices_face_x=self.vertices_face_x,vertices_face_xy=self.vertices_face_xy,k=self.k,drop_prob=self.drop_prob,reduced_data_shape=self.reduced_data_shape)


        
    def forward(self, x):
        x_hat=self.discriminator(x)
        return x_hat.reshape(x.shape)
    
    def disc_loss(self, x):
        x_hat=self.discriminator(x)
        loss=L2_loss(x, x_hat.reshape(x.shape)).mean()
        return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx ):
        x=batch
        z_p_1=torch.randn(len(x), self.latent_dim).type_as(x)
        batch_p_1=self.generator(z_p_1)
        
        if optimizer_idx==0:
            loss=self.disc_loss(batch_p_1)
            self.log("train_generator_loss", loss)
            return loss
        

        if optimizer_idx==1:
            gamma=torch.tensor(0.5)
            k=torch.tensor(0.)
            lambda_k = torch.tensor(0.001)
            z_d_1=self.discriminator.encoder_base(x)
            z_d_1=z_d_1.reshape(len(x),self.latent_dim)
            batch_d_1=self.generator(z_d_1)
            tmp=self.disc_loss(x)
            loss_disc=tmp-k*self.disc_loss(batch_d_1)
            loss_gen=self.disc_loss(batch_p_1)
            self.log("train_discriminagtor_loss", loss_disc)
            k = torch.min(torch.max( k + lambda_k * torch.mean(gamma * tmp - loss_gen), torch.tensor(0)), torch.tensor(1))
            return loss_disc
    
    def validation_step(self, batch, batch_idx):
        x=batch
        z=self.sample_mesh().reshape(1,-1)
        loss=torch.min(torch.linalg.norm((x-z),axis=1))/torch.linalg.norm(x)
        self.log("val_rec", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x=batch
        self.log("test_began_loss", self.disc_loss(x))
        return self.disc_loss(x)
        

    def configure_optimizers(self): 
        optimizer_gen = torch.optim.AdamW(self.generator.parameters(), lr=1e-2) #0.0002
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=2e-2) #0.0006
        return [optimizer_gen,optimizer_disc], []

    def sample_mesh(self,mean=None,var=None):
        device=self.generator.decoder_base.pca._V.device
        self=self.to(device)
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim)

        if var==None:
            var_1=torch.ones(1,self.latent_dim)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim)+mean_1
        z=z.to(device)
        tmp=self.generator(z)
        return tmp
     
