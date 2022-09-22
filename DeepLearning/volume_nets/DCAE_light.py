#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data import DataLoader
import meshio
from torch.utils.data import Dataset
import os
import torch
import stl
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pyro
from pyevtk.hl import unstructuredGridToVTK
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
from pytorch_lightning.callbacks import TQDMProgressBar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math


NUMBER_SAMPLES=200
STRING="bulbo_{}.pt"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 200
NUM_WORKERS = int(os.cpu_count() / 2)


def getinfo(pt):
    temp=torch.load(pt)
    return temp


class Data(LightningDataModule):
    def get_size(self):
        temp=getinfo(STRING.format(0))
        return (1,temp.shape[0],temp.shape[1],temp.shape[2])

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
            self.data=torch.zeros(self.num_samples,self.get_size()[1],self.get_size()[2],self.get_size()[3])
            for i in range(0,self.num_samples):
                if i%100==0:
                    print(i)
                self.data[i]=getinfo(STRING.format(i))
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


class CBR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.conv=nn.Conv3d(1, 1, in_features-out_features+1, bias=False)
        self.batch=nn.BatchNorm3d(1)
        self.relu=nn.Tanh()
    
    def forward(self,x):
        return self.relu(self.batch(self.conv(x)))

class DBR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.conv=nn.ConvTranspose3d(1, 1, out_features-in_features+1, bias=False)
        self.batch=nn.BatchNorm3d(1)
        self.relu=nn.Sigmoid()
    
    def forward(self,x):
        return self.relu(self.batch(self.conv(x)))

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = DBR(latent_dim, 10)
        self.fc2 = DBR(10, 20)
        self.fc3=nn.ConvTranspose3d(1, 1, 11)


    def forward(self, z):
        z=z.reshape((-1,1,1,1,1))
        z=self.fc1(z)
        z=self.fc2(z)
        z=self.fc3(z)
        return z
    
    

class Encoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = CBR(self.data_shape[1],20)
        self.fc2 = CBR(20, 10)
        self.fc3 = CBR(10, 1)



    def forward(self, x):
        x=x.reshape((-1,1,self.data_shape[1],self.data_shape[2],self.data_shape[3]))
        x=self.fc3(self.fc2(self.fc1(x)))
        x=x/torch.linalg.norm(x)*torch.tanh(torch.linalg.norm(x))*(2/math.pi)
        return x

        
class AE(LightningModule):
    
    def __init__(self,data_shape,hidden_dim: int= 300,latent_dim: int = 1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.decoder = Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape)
        self.encoder = Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        
    def forward(self, x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat.reshape(x.shape).reshape(x.shape)
    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        
        loss = self.ae_loss(batch_hat,batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        loss = self.ae_loss(batch_hat,batch)
        self.log("validation_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        loss = self.ae_loss(batch_hat,batch)
        self.log("test_loss", loss)
        return loss

    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def sample_mesh(self,mean,var):
        z = torch.sqrt(var)*torch.randn(1,1)+mean
        temp=self.decoder(z)
        return temp

if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1,callbacks=[TQDMProgressBar(refresh_rate=10)])
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = AE(data.get_size())
trainer.fit(model, data)
trainedlatent=model.encoder.forward(data.data_train[0:len(data.data_train)]).cpu().detach().numpy()
testedlatent=model.encoder.forward(data.data_test[0:len(data.data_test)]).cpu().detach().numpy()
validatedlatent=model.encoder.forward(data.data_val[0:len(data.data_val)]).cpu().detach().numpy()
alllatent=model.encoder.forward(data.data[0:len(data.data)]).cpu().detach().numpy()

trainer.validate(datamodule=data)
trainer.test(datamodule=data)
mean=torch.mean(torch.tensor(alllatent))
var=torch.var(torch.tensor(alllatent))
model.eval()
temp= model.sample_mesh(mean,var)
threshold=torch.tensor(0.)
for i in range(len(data.data)):
    fitted=model.decoder(model.encoder(data.data[i]).to(model.device)).reshape(-1)
    num=(data.data[i].shape.numel()-torch.sum(data.data[i])).long()
    fitted,_=torch.sort(fitted)
    print(fitted[num])
    threshold=threshold+fitted[num]
threshold=threshold/len(data.data)
error=torch.tensor(0.)
for i in range(100):
    #temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))).type_as(error).reshape(1,-1)
    temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))>threshold).type_as(error).reshape(1,-1)
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (prior) and data is", error)
error=torch.tensor(0.)
for i in range(100):
    temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))>threshold).type_as(error).reshape(1,-1)
    #temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))).type_as(error).reshape(1,-1)
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (posterior) and data is", error)
#ae.load_state_dict(torch.load("cube.pt"))



