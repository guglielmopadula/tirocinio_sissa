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

        def __init__(self, latent_dim, hidden_dim, batch_size,data_shape,pca,drop_prob,barycenter):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,pca=pca,batch_size=batch_size,drop_prob=drop_prob,barycenter=barycenter)

        def forward(self,x):
            return self.decoder_base(x)
        
    class Discriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dim, batch_size,data_shape,pca,drop_prob,barycenter):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,pca=pca,drop_prob=drop_prob,batch_size=batch_size)
            self.decoder_base=Decoder_base(latent_dim=latent_dim, hidden_dim=hidden_dim, data_shape=data_shape,pca=pca,batch_size=batch_size,drop_prob=drop_prob,barycenter=barycenter)
             
        def forward(self,x):
            x_hat=self.decoder_base(self.encoder_base(x))
            return x_hat
         
    def __init__(self,data_shape,pca,latent_dim,batch_size,drop_prob,barycenter,hidden_dim: int= 300,**kwargs):
        super().__init__()
        self.pca=pca
        self.barycenter=barycenter
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.data_shape = data_shape
        self.generator = self.Generator(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,pca=self.pca,drop_prob=self.drop_prob,batch_size=self.batch_size,barycenter=self.barycenter)
        self.discriminator = self.Discriminator(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,pca=self.pca,drop_prob=self.drop_prob,batch_size=self.batch_size,barycenter=self.barycenter)
        self.automatic_optimization=False


        
    def forward(self, x):
        x_hat=self.discriminator(x)
        return x_hat.reshape(x.shape)
    
    def disc_loss(self, x):
        x_hat=self.discriminator(x)
        loss=L2_loss(x, x_hat.reshape(x.shape)).mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        x=batch
        z_p_1=torch.randn(len(x), self.latent_dim).type_as(x)

        z_d_1=self.discriminator.encoder_base(x)
        z_d_1=z_d_1.reshape(len(x),self.latent_dim)
        

        batch_p_1=self.generator(z_p_1)
        batch_d_1=self.generator(z_d_1)
        
        gamma=1
        k=0
        lambda_k = 0.001
        
        loss=self.disc_loss(batch_p_1)
        g_opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(g_opt, gradient_clip_val=0.1)
        g_opt.step()        
        

        loss_disc=self.disc_loss(x)-k*self.disc_loss(batch_d_1)
        loss_gen=self.disc_loss(batch_p_1.detach())
        diff = torch.mean(gamma * self.disc_loss(batch) - loss_gen)
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)
        d_opt.zero_grad()
        self.manual_backward(loss_disc)
        self.clip_gradients(d_opt, gradient_clip_val=0.1)
        d_opt.step()        

        
        return loss_disc
        
    def validation_step(self, batch, batch_idx):
        x=batch
        self.log("val_began_loss", self.disc_loss(x))
        return self.disc_loss(x)

        
    def test_step(self, batch, batch_idx):
        x=batch
        self.log("test_began_loss", self.disc_loss(x))
        return self.disc_loss(x)
        

    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer_gen = torch.optim.AdamW(self.generator.parameters(), lr=0.00000002) 
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0000001) 
        return [optimizer_gen,optimizer_disc], []

    def sample_mesh(self,mean=None,var=None):
        device=self.generator.decoder_base.pca._V.device
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)

        if var==None:
            var=torch.ones(1,self.latent_dim)

        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        z=z.to(device)
        temp_interior=self.generator(z)
        return temp_interior,z
     
