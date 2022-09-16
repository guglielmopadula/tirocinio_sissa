#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data import DataLoader
from stl import mesh
from torch.utils.data import Dataset
import os
import torch
import stl
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pyro
from collections import OrderedDict
import pyro.distributions as dist
import torch.nn.functional as F
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from ordered_set import OrderedSet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split,TensorDataset
import pyro.poutine as poutine
from argparse import ArgumentParser
import itertools


device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math


ae_hyp=0.999

NUMBER_SAMPLES=100
STRING="bulbo_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 100
NUM_WORKERS = int(os.cpu_count() / 2)

def getinfo(stl):
    your_mesh = mesh.Mesh.from_file(stl)
    myList = list(OrderedSet(tuple(map(tuple,your_mesh.vectors.reshape(np.prod(your_mesh.vectors.shape)//3,3)))))
    array=your_mesh.vectors
    topo=np.zeros((np.prod(your_mesh.vectors.shape)//9,3))
    for i in range(np.prod(your_mesh.vectors.shape)//9):
        for j in range(3):
            topo[i,j]=myList.index(tuple(array[i,j].tolist()))
    return torch.tensor(myList),torch.tensor(topo, dtype=torch.int64)

    
def applytopology(V,M):
    Q=torch.zeros((M.shape[0],3,3),device=device)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Q[i,j]=V[M[i,j].item()]
    return Q

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


class VolumeNormalizer(nn.Module):
    def __init__(self,M):
        super().__init__()
        self.M=M
    def forward(self, x):
        temp=x.shape
        x=x.reshape(x.shape[0],-1,3)
        x=x/((x[:,self.M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
        return x.reshape(temp)
    
    def forward_single(self,x):
        temp=x.shape
        x=x.reshape(1,-1,3)
        x=x/((x[:,self.M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(1,-1,3)
        return x.reshape(temp)







class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,M):
        super().__init__()
        self.data_shape=data_shape
        self.M=M
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.fc5=VolumeNormalizer(self.M)
        self.relu = nn.ReLU()

    def forward(self, z):
        print(type(z))
        result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))
        result=self.fc5(result)
        result=result.view(result.size(0),-1)
        return result
    
    

class Encoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = nn.Linear(int(np.prod(self.data_shape)),hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
        self.batch=nn.BatchNorm1d(1)
        self.fc32 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        mu=self.batch(mu)
        mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        return mu


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1=nn.Linear(latent_dim,2)
        self.relu=nn.LeakyReLU()
        self.fc2=nn.Linear(2,2)
        self.fc3=nn.Linear(2,1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,z):
        result=self.sigmoid(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))
        return result


        
class AAE(LightningModule):
    
    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = 1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.decoder = Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
        self.encoder = Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.discriminator=Discriminator(latent_dim=self.hparams.latent_dim)
        
    def forward(self, x):
        print(x.shape)
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat.reshape(x.shape)
    
    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss
    
    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy(y_hat, y)


    def training_step(self, batch, batch_idx, optimizer_idx ):
        z_enc=self.encoder(batch)
        z=torch.randn(len(batch), self.hparams.latent_dim)
        ones=torch.ones(len(batch))
        zeros=torch.zeros(len(batch))

        
        if optimizer_idx==0:
            batch_hat=self.decoder(z_enc).reshape(batch.shape)
            ae_loss = ae_hyp*self.ae_loss(batch_hat,batch)+(1-ae_hyp)*self.adversarial_loss(self.discriminator(z).reshape(ones.shape), ones)
            self.log("train_ae_loss", ae_loss)
            return ae_loss
        
        if optimizer_idx==1:
            real_loss = self.adversarial_loss(self.discriminator(z).reshape(ones.shape), ones)
            fake_loss = self.adversarial_loss(self.discriminator(z_enc).reshape(zeros.shape), zeros)
            tot_loss= (real_loss+fake_loss)/2
            self.log("train_ae_loss", tot_loss)
            return tot_loss
            
            
    
    def validation_step(self, batch, batch_idx):
        z=torch.randn(len(batch), self.hparams.latent_dim)
        z_enc=self.encoder(batch)
        ones=torch.ones(len(batch))
        zeros=torch.zeros(len(batch))
        batch_hat=self.decoder(z_enc).reshape(batch.shape)
        ae_loss = self.ae_loss(batch_hat,batch)
        self.log("val_ae_loss", ae_loss)
        return ae_loss
        
    
    def test_step(self, batch, batch_idx):
        z=torch.randn(len(batch), self.hparams.latent_dim)
        z_enc=self.encoder(batch)
        ones=torch.ones(len(batch))
        zeros=torch.zeros(len(batch))
        batch_hat=self.decoder(z_enc).reshape(batch.shape)
        ae_loss = self.ae_loss(batch_hat,batch)
        self.log("test_ae_loss", ae_loss)
        return ae_loss
        

    

    def configure_optimizers(self):
        optimizer_ae = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-3)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler_ae = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return [optimizer_ae,optimizer_disc], []
        #return {"optimizer": [optimizer_ae,optimizer_disc], "lr_scheduler": [scheduler_ae,scheduler_disc], "monitor": ["train_loss","train_loss"]}

    def sample_mesh(self,mean,var):
        z = torch.sqrt(var)*torch.randn(1,1)+mean
        temp=self.decoder(z)
        temp=applytopology(temp.reshape(-1,3), self.M)
        return temp

if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1)
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = AAE(data.get_size(),data.M)
trainer.fit(model, data)
trainedlatent=model.encoder.forward(data.data_train[0:len(data.data_train)]).cpu().detach().numpy()
testedlatent=model.encoder.forward(data.data_test[0:len(data.data_test)]).cpu().detach().numpy()
validatedlatent=model.encoder.forward(data.data_val[0:len(data.data_val)]).cpu().detach().numpy()
trainer.validate(datamodule=data)
trainer.test(datamodule=data)
mean=torch.mean(torch.tensor(testedlatent))
var=torch.var(torch.tensor(testedlatent))

temp = model.sample_mesh(mean,var)
newmesh = np.zeros(len(temp), dtype=mesh.Mesh.dtype)
newmesh['vectors'] = temp.cpu().detach().numpy().copy()
mymesh = mesh.Mesh(newmesh.copy())
mymesh.save('test.stl', mode=stl.Mode.ASCII)

#vae.load_state_dict(torch.load("cube.pt"))
