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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math


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




class LBR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.batch(self.lin(x)))





class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,M):
        super().__init__()
        self.data_shape=data_shape
        self.M=M
        self.fc1 = LBR(latent_dim, hidden_dim)
        self.fc2 = LBR(hidden_dim, hidden_dim)
        self.fc3 = LBR(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.fc5=VolumeNormalizer(self.M)
        self.relu = nn.ReLU()

    def forward(self, z):
        result=self.fc4(self.fc3(self.fc2(self.relu(self.fc1(z)))))
        result=self.fc5(result)
        result=result.view(result.size(0),-1)
        return result
    
    

class Encoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = LBR(int(np.prod(self.data_shape)),hidden_dim)
        self.fc21 = LBR(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
        self.batch=nn.BatchNorm1d(1)
        self.fc32 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        mu=self.batch(mu)
        #mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        return mu





        
class AE(LightningModule):
    
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
        self.log("train_ae_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        loss = self.ae_loss(batch_hat,batch)
        self.log("validation_ae_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        loss = self.ae_loss(batch_hat,batch)
        self.log("test_ae_loss", loss)
        return loss

    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return {"optimizer": optimizer}

    def sample_mesh(self,mean,var):
        z = torch.sqrt(var)*torch.randn(1,1)+mean
        temp=self.decoder(z)
        return temp

if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1)
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = AE(data.get_size(),data.M)
trainer.fit(model, data)
trainedlatent=model.encoder.forward(data.data_train[0:len(data.data_train)]).cpu().detach().numpy()
testedlatent=model.encoder.forward(data.data_test[0:len(data.data_test)]).cpu().detach().numpy()
validatedlatent=model.encoder.forward(data.data_val[0:len(data.data_val)]).cpu().detach().numpy()
trainer.validate(datamodule=data)
trainer.test(datamodule=data)
mean=torch.mean(torch.tensor(trainedlatent))
var=torch.var(torch.tensor(trainedlatent))
model.eval()
temp = model.sample_mesh(mean,var)
meshio.write_points_cells('test.stl',temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy().tolist(),[("triangle", data.M)])
error=0
for i in range(100):
    temp = model.sample_mesh(torch.tensor(0),torch.tensor(1))
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (prior) and data is", error)
error=0
for i in range(100):
    temp = model.sample_mesh(mean,var)
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (posterior) and data is", error)
#ae.load_state_dict(torch.load("cube.pt"))

