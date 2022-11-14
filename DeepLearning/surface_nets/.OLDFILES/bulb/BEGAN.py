#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:52:19 2022

@author: cyberguli
"""

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data import DataLoader
import meshio
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import random_split
import os.path as osp
from torch_geometric.datasets import Planetoid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math


ae_hyp=0.999

NUMBER_SAMPLES=200
STRING="bulbo_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 200
NUM_WORKERS = int(os.cpu_count() / 2)

def getinfo(stl):
    mesh=meshio.read(stl)
    points=torch.tensor(mesh.points.astype(np.float32))
    triangles=torch.tensor(mesh.cells_dict['triangle'].astype(np.int64))
    return points,triangles


class Data(LightningDataModule):
    def get_size(self):
        temp,self.M=getinfo(STRING.format(0))
        return (1,temp.shape[0],temp.shape[1])

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        num_samples: int = NUMBER_SAMPLES,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples=num_samples


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage=="fit":
            self.data=torch.zeros(self.num_samples,self.get_size()[1],self.get_size()[2])
            for i in range(0,self.num_samples):
                if i%100==0:
                    print(i)
                self.data[i],_=getinfo(STRING.format(i))
            # Assign train/val datasets for use in dataloaders
            self.data_train, self.data_val,self.data_test = random_split(self.data, [math.floor(0.5*self.num_samples), math.floor(0.3*self.num_samples),self.num_samples-math.floor(0.5*self.num_samples)-math.floor(0.3*self.num_samples)])

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)

class LSL(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=torch.nn.utils.parametrizations.spectral_norm(nn.Linear(in_features, out_features))
        self.relu=nn.LeakyReLU()
    
    def forward(self,x):
        return self.relu(self.lin(x))

class LBR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.batch(self.lin(x)))

class VolumeNormalizer(nn.Module):
    def __init__(self,M,data_shape):
        super().__init__()
        self.M=M
        self.data_shape=data_shape
    def forward(self, x):
        temp=x.shape
        x=x.reshape(-1,self.data_shape[1],self.data_shape[2])
        x=x/((x[:,self.M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
        return x.reshape(temp)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,M):
        super().__init__()
        self.data_shape=data_shape
        self.M=M
        self.fc1 = LBR(latent_dim, hidden_dim)
        self.fc2 = LBR(hidden_dim, hidden_dim)
        self.fc3 = LBR(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.fc5=VolumeNormalizer(self.M,self.data_shape)
    def forward(self, z):
        #result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))
        result=self.fc4(self.fc3(self.fc2(self.fc1(z))))
        result=self.fc5(result)
        result=result.view(result.size(0),-1)
        return result
    
    

class Encoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.latent_dim=latent_dim
        self.fc1 = LBR(int(np.prod(self.data_shape)),hidden_dim)
        self.fc21 = LBR(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
        self.batch_mu=nn.BatchNorm1d(self.latent_dim)
        self.fc32 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        #sigma=self.fc32(sigma)
        mu=self.batch_mu(mu)
        mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        return mu


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape,M):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = Decoder(latent_dim, hidden_dim,data_shape,M)
        
    def forward(self,x):
        result=self.fc1(x)
        return result



class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape,M):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = Encoder(latent_dim, hidden_dim,data_shape)
        self.fc2 = Decoder(latent_dim, hidden_dim,data_shape,M)
        
    def forward(self,x):
        x=x.reshape(-1,int(np.prod(self.data_shape)))
        result=self.fc1(x)
        result=self.fc2(result)
        return result


class BEGAN(LightningModule):
    
    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = 8,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.discriminator = Discriminator(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
        self.generator = Generator(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim,M=self.M)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        
    def forward(self, x):
        x_hat=self.discriminator(x)
        return x_hat.reshape(x.shape)
    
    def ae_loss(self, x):
        loss=F.mse_loss(x, self.discriminator(x).reshape(x.shape), reduction="none")
        loss=loss.mean()
        return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx ):
        z_p=torch.randn(len(batch), self.hparams.latent_dim).type_as(batch)
        batch_p=self.generator(z_p)
        gamma=0.5
        k=0
        lambda_k = 0.001
        
        if optimizer_idx==0:
            loss=self.ae_loss(batch_p)
            self.log("train_generator_loss", loss)
            return loss
        

        if optimizer_idx==1:    
            loss_disc=self.ae_loss(batch)-k*self.ae_loss(batch)
            loss_gen=self.ae_loss(batch_p)
            self.log("train_discriminagtor_loss", loss_disc)
            diff = torch.mean(gamma * loss_disc - loss_gen)
            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)
            return loss_disc
        
            
                
    
    def validation_step(self, batch, batch_idx):
        self.log("val_vaegam_loss", self.ae_loss(batch))
        return self.ae_loss(batch)

        
    def test_step(self, batch, batch_idx):
        self.log("test_vaegan_loss", self.ae_loss(batch))
        return self.ae_loss(batch)
        

    

    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.02) #0.02
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.050) #0.050

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return [optimizer_gen,optimizer_disc], []
        #return {"optimizer": [optimizer_ae,optimizer_disc], "lr_scheduler": [scheduler_ae,scheduler_disc], "monitor": ["train_loss","train_loss"]}

    def sample_mesh(self):
        z = torch.randn(1,self.latent_dim)
        temp=self.generator(z)
        return temp

if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1)
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = BEGAN(data.get_size(),data.M)
trainer.fit(model, data)
trainer.validate(datamodule=data)
trainer.test(datamodule=data)
model.eval()
temp = model.sample_mesh()
meshio.write_points_cells('test.stl',temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy().tolist(),[("triangle", data.M)])
error=0
for i in range(100):
    temp = model.sample_mesh()
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (prior) and data is", error)
meshio.write_points_cells('test.stl',temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy().tolist(),[("triangle", data.M)])
