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
from torch_geometric.nn import ChebConv,GraphNorm,GCNConv,FeaStConv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math
graphs=torch.load("graphs.pt")
Ss=torch.load("Ss.pt")


K=6
NUMBER_SAMPLES=500
STRING="bulbo_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 500
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




class CBDR(nn.Module):
    def __init__(self,in_channels,out_channels,i):
        super().__init__()
        self.cheb=ChebConv(in_channels,out_channels,K)
        self.i=i
        self.batch=GraphNorm(out_channels)
        self.elu=nn.ELU()
    def forward(self,x):
        temp=self.cheb(x,graphs[self.i])
        #temp=self.batch(temp)
        temp=torch.matmul(Ss[self.i],temp)
        temp=self.elu(temp)
        return temp

class CBLB_E(nn.Module):
    def __init__(self,in_channels,out_channels,i):
        super().__init__()
        self.cheb=ChebConv(in_channels,out_channels,K)
        self.i=i
        self.batch=GraphNorm(out_channels)
        self.lin=nn.Linear(int((torch.max(graphs[i]+1)).item()),1)
        self.elu=nn.ELU()
    def forward(self,x):
        temp=self.cheb(x,graphs[self.i])
        #temp=self.batch(temp)
        temp=temp.reshape(-1,torch.max(graphs[self.i]+1))
        temp=self.lin(temp)
        temp=temp/torch.linalg.norm(temp)*torch.tanh(torch.linalg.norm(temp))*(2/math.pi)
        return temp
    
class DBLB_D(nn.Module):
    def __init__(self,in_channels,out_channels,i):
        super().__init__()
        self.cheb=ChebConv(in_channels,out_channels,K)
        self.i=i
        self.batch=GraphNorm(out_channels)
        self.lin=nn.Linear(1,int((torch.max(graphs[i]+1)).item()))
        self.elu=nn.ELU()
    def forward(self,x):
        temp=self.lin(x).reshape(-1,torch.max(graphs[self.i]+1),1)
        temp=self.cheb(temp,graphs[self.i])
        #temp=self.batch(temp)
        temp=self.elu(temp)
        return temp

class DBDR(nn.Module):
    def __init__(self,in_channels,out_channels,i):
        super().__init__()
        self.cheb=ChebConv(in_channels,out_channels,K)
        self.i=i
        #self.batch=GraphNorm(out_channels)
        self.elu=nn.ELU()
    def forward(self,x):
        temp=torch.matmul(Ss[self.i].T,x)
        temp=self.cheb(temp,graphs[self.i])
        #temp=self.batch(temp)
        temp=self.elu(temp)
        return temp 



class Encoder(nn.Module):
    def __init__(self,latent_dim, hidden_dim, data_shape):
        super().__init__()
        self.fc1=CBDR(3,3,0)
        self.fc2=CBDR(3,2,1)
        self.fc3=CBDR(2,4,2)
        self.fc4=CBDR(4,8,3)
        self.fc5=CBDR(8,16,4)
        self.fc6=CBDR(16,32,5)
        self.fc7=CBDR(32,64,6)
        self.fc8=CBDR(64,128,7)
        self.fc9=CBLB_E(128,1,8)
    def forward(self,x):
        x=x.reshape(x.size(0),-1,3)
        temp=self.fc8(self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x))))))))
        temp=self.fc9(temp)
        return temp

class Decoder(nn.Module):
    def __init__(self,latent_dim, hidden_dim, data_shape,M):
        super().__init__()
        self.fc1=DBLB_D(1,128,8)
        self.fc2=DBDR(128,64,7)
        self.fc3=DBDR(64,32,6)
        self.fc4=DBDR(32,16,5)
        self.fc5=DBDR(16,8,4)
        self.fc6=DBDR(8,4,3)
        self.fc7=DBDR(4,2,2)
        self.fc8=DBDR(2,3,1)
        self.fc9=DBDR(3,3,0)
        self.M=M
        self.vol=VolumeNormalizer(M)

    def forward(self,z):    
        temp=self.fc8(self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(z))))))))
        temp=self.fc9(temp)
        x=self.vol(temp)
        return x






        
class AE(LightningModule):
    
    def __init__(self,data_shape,M,hidden_dim: int=300,latent_dim: int = 1,lr: float = 0.01,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.lr=lr
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",patience=16,factor=0.16,min_lr=5e-02)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_ae_loss"}

    def sample_mesh(self,mean,var):
        z = torch.sqrt(var)*torch.randn(1,1)+mean
        z=z.reshape(1,-1,1)
        temp=self.decoder(z)
        return temp

if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500)
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = AE(data.get_size(),data.M)
trainer.tune(model,data)
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
    temp = model.sample_mesh(torch.tensor(0),torch.tensor(1)).reshape(-1)
    true=data.data.reshape(-1,temp.shape[0])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (prior) and data is", error)
error=0
for i in range(100):
    temp = model.sample_mesh(mean,var).reshape(-1)
    true=data.data.reshape(-1,temp.shape[0])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (posterior) and data is", error)
#ae.load_state_dict(torch.load("cube.pt"))

