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
from pytorch_lightning.callbacks import TQDMProgressBar,RichProgressBar

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

gamma=0.98

def getinfo(pt):
    temp=torch.load(pt)
    return temp

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
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.conv=nn.Conv3d(in_channels, out_channels, kernel_size,stride,bias=False)
        self.batch=nn.BatchNorm3d(out_channels)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.batch(self.conv(x)))

class DBS(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.conv=nn.ConvTranspose3d(in_channels, out_channels, kernel_size,stride, bias=False)
        self.batch=nn.BatchNorm3d(out_channels)
        self.sigmoid=nn.Sigmoid()
        self.step=StraightThroughEstimator()
    
    def forward(self,x):
        return self.sigmoid(self.batch(self.conv(x)))
    
class LBR(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.lin=nn.Linear(input_dim, output_dim)
        self.batch=nn.BatchNorm1d(output_dim)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.batch(self.lin(x)))  
    




class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.lin1= LBR(1,4)
        self.lin2=LBR(4,16)
        self.lin3=LBR(16,64)
        self.lin4=LBR(64,256)
        self.lin5=LBR(256,1024)
        self.fc1 = DBS(1024, 256,2,2)
        self.fc2 = DBS(256, 64,2,2)
        self.fc3 = DBS(64, 16,2,2)
        self.fc4 = DBS(16, 4,2,2)
        self.fc5 = DBS(4, 1,2,2)
        self.straight=StraightThroughEstimator()


    def forward(self, z):
        z=z.reshape((-1,1))
        z=self.lin5(self.lin4(self.lin3(self.lin2(self.lin1(z)))))
        z=z.reshape(-1,1024,1,1,1)
        z=self.fc1(z)
        z=self.fc2(z)
        z=self.fc3(z)
        z=self.fc4(z)
        z=self.fc5(z)
        #z=self.straight(z)
        return z
    
    

class Encoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = CBR(1,4,2,2)
        self.fc2 = CBR(4,16,2,2)
        self.fc3 = CBR(16,64,2,2)
        self.fc4 = CBR(64,256,2,2)
        self.fc5 = CBR(256,1024,2,2)
        self.lin1= LBR(1024,256)
        self.lin2=LBR(256,64)
        self.lin3=LBR(64,16)
        self.lin4=LBR(16,4)
        self.lin5=LBR(4,1)

        
    def forward(self, x):
        x=x.reshape((-1,1,self.data_shape[1],self.data_shape[2],self.data_shape[3]))
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        x=self.fc5(x)
        x=x.reshape(-1,1024)
        x=self.lin5(self.lin4(self.lin3(self.lin2(self.lin1(x)))))
        return x

        
class AE(LightningModule):
    
    def __init__(self,data_shape,hidden_dim: int= 300,latent_dim: int = 1,lr: float = 0.02,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
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
    
    def ae_loss(self,target, output):
        target=target.reshape(-1)
        output=output.reshape(-1)
        lossfunction=nn.BCELoss()
        #loss = -(gamma * target * torch.clamp(torch.log(output),-100,100) + 2.0 * (1-gamma) * torch.clamp(torch.log(1.0 - output),-100,100))
        loss=lossfunction(target,output)
        loss=loss.mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        #batch_res=-1+3*batch
        #batch_hat_res=0.1+0.9*batch_hat
        #loss = self.ae_loss(batch_res   ,batch_hat_res)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-05)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def sample_mesh(self,mean,var):
        z = torch.sqrt(var)*torch.randn(1)+mean
        temp=self.decoder(z)
        return temp

def plot(tensor):
    tensor=tensor.reshape((32,32,32))
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(tensor.detach().numpy(), facecolors='black', edgecolor='k')
    plt.show()
        


if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1,callbacks=[RichProgressBar(refresh_rate=10)])
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1,callbacks=[RichProgressBar(refresh_rate=10)])
data=Data()
model = AE(data.get_size())
trainer.fit(model, data)
trainedlatent=model.encoder.forward(data.data_train[0:len(data.data_train)]).cpu().detach().numpy()
testedlatent=model.encoder.forward(data.data_test[0:len(data.data_test)]).cpu().detach().numpy()
validatedlatent=model.encoder.forward(data.data_val[0:len(data.data_val)]).cpu().detach().numpy()
alllatent=model.encoder.forward(data.data[0:len(data.data)]).cpu().detach().numpy().reshape(-1,1)

trainer.validate(datamodule=data)
trainer.test(datamodule=data)
mean=torch.mean(torch.tensor(trainedlatent),dim=0)
var=torch.var(torch.tensor(trainedlatent),dim=0)
model.eval()
temp= model.sample_mesh(mean,var).reshape(32,32,32)
threshold=torch.tensor(0.)
for i in range(len(data.data)):
    fitted=model.decoder(model.encoder(data.data[i]).to(model.device)).reshape(-1)
    num=(data.data[i].shape.numel()-torch.sum(data.data[i])).long()
    fitted,_=torch.sort(fitted)
    threshold=threshold+fitted[num]
threshold=threshold/len(data.data)
error=torch.tensor(0.)
for i in range(100):
    temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))).type_as(error).reshape(1,-1)
    #temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))>threshold).type_as(error).reshape(1,-1)
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (prior) and data is", error)
error=torch.tensor(0.)
for i in range(100):
    #temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))>threshold).type_as(error).reshape(1,-1)
    temp = (model.sample_mesh(torch.tensor(0),torch.tensor(1))).type_as(error).reshape(1,-1)
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (posterior) and data is", error)
#ae.load_state_dict(torch.load("cube.pt"))



