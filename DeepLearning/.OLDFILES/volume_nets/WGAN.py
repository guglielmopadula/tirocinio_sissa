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
from scipy.spatial import Delaunay
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torch.utils.data import random_split


device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math


ae_hyp=0.999

NUMBER_SAMPLES=500
STRING="bulbo_{}.vtk"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 500
NUM_WORKERS = int(os.cpu_count() / 2)


def getinfo(vtk):
    mesh=meshio.read(vtk)
    points=torch.tensor(mesh.points.astype(np.float64))
    F=torch.tensor(mesh.cells_dict['tetra'].astype(np.int64))
    triangles=torch.tensor(mesh.cells_dict['triangle'].astype(np.int64))
    return points,F,triangles


class Data(LightningDataModule):
    def get_size(self):
        temp,self.M,self.triangles=getinfo(STRING.format(0))
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
                self.data[i],_,_=getinfo(STRING.format(i))
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
    def __init__(self,M,data_shape):
        super().__init__()
        self.M=M
        self.data_shape=data_shape
    def forward(self, x):
        temp=x.shape
        x=x.reshape(-1,self.data_shape[1],self.data_shape[2])
        xmatrix=x[:,self.M]
        ones=torch.ones(xmatrix.shape[0],xmatrix.shape[1],xmatrix.shape[2],1).type_as(x)
        x=x/((torch.cat((x[:,self.M],ones),3).det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
        return x.reshape(temp)
    
    


class LSL(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=torch.nn.utils.parametrizations.spectral_norm(nn.Linear(in_features, out_features))
        self.relu=nn.LeakyReLU()
    
    def forward(self,x):
        return self.relu(self.lin(x))

class LSR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=torch.nn.utils.parametrizations.spectral_norm(nn.Linear(in_features, out_features))
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.lin(x))


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,M):
        super().__init__()
        self.data_shape=data_shape
        self.M=M
        self.fc1 = LSR(latent_dim, hidden_dim)
        self.fc2 = LSR(hidden_dim, hidden_dim)
        self.fc3 = LSR(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.fc5=VolumeNormalizer(self.M,self.data_shape)

    def forward(self, z):
        result=self.fc1(z)
        result=self.fc2(result)
        result=self.fc3(result)
        result=self.fc4(result)
        result=self.fc5(result)
        result=result.view(result.size(0),-1)
        return result
    


class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = LSL(int(np.prod(self.data_shape)),hidden_dim)
        self.fc2 = LSL(hidden_dim, hidden_dim)
        self.fc3=LSL(hidden_dim,2)
        self.fc4=nn.Linear(2,1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        x=x.reshape(-1,int(np.prod(self.data_shape)))
        result=self.fc1(x)
        result=self.fc2(result)
        result=self.fc3(result)
        result=self.fc4(result)
        #result=self.sigmoid(self.fc4(result))
        return result


        
class GAN(LightningModule):
    
    def __init__(self,data_shape,M,hidden_dim: int= 500,latent_dim: int = 5,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.generator = Generator(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.discriminator=Discriminator(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hparams.hidden_dim)
        self.lr=lr
    def forward(self):
        z=torch.randn(self.hparams.latent_dim)
        x_hat=self.generator(z)
        return x_hat
    
    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss
    
    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy(y_hat)
    
    def wasserstain_loss(self,y_hat):
        return -torch.mean(y_hat)

    

    def training_step(self, batch, batch_idx, optimizer_idx ):
        z=torch.randn(len(batch), self.hparams.latent_dim).type_as(batch)
        ones=torch.ones(len(batch)).type_as(batch)
        zeros=-torch.ones(len(batch)).type_as(batch)
        batch_hat=self.generator(z)
        
        if optimizer_idx==0:
            #g_loss = self.adversarial_loss(self.discriminator(batch_hat).reshape(ones.shape), ones)
            g_loss=self.wasserstain_loss(self.discriminator(batch_hat))
            self.log("gan_gen_train_loss", g_loss)
            return g_loss
        
        if optimizer_idx==1:
            real_loss = self.wasserstain_loss(self.discriminator(batch).reshape(ones.shape))
            fake_loss = -self.wasserstain_loss(self.discriminator(batch_hat).reshape(zeros.shape))
            tot_loss= (real_loss+fake_loss)/2
            self.log("gan_disc_train_loss", tot_loss)
            return tot_loss
            
            
    
    def validation_step(self, batch, batch_idx):
        None        
    
    def test_step(self, batch, batch_idx):
        None
             
    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002,betas=(0.5,0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.005,betas=(0.5,0.999)  )
        #scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50,gamma=0.01)
        #scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50,gamma=0.01)

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return [optimizer_g,optimizer_d]# [scheduler_g,scheduler_d]
        #return {"optimizer": [optimizer_ae,optimizer_disc], "lr_scheduler": [scheduler_ae,scheduler_disc], "monitor": ["train_loss","train_loss"]}

    def sample_mesh(self):
        z = torch.randn(1,self.latent_dim)
        temp=self.generator(z)
        return temp

if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1,auto_lr_find=True,callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],gradient_clip_val=0.5)
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = GAN(data.get_size(),data.M)
trainer.fit(model, data)
model.eval()
temp = model.sample_mesh()



mesh=meshio.write_points_cells('test.vtk',temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy().tolist(),[("triangle", data.triangles),("tetra", data.M)])
#vae.load_state_dict(torch.load("cube.pt"))
temp=temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy()
tri = Delaunay(temp)
meshio.write_points_cells('test_del.vtk',temp,[("tetra", tri.simplices)])
error=0
for i in range(100):
    temp = model.sample_mesh()
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
print("Average distance between sample (prior) and data is", error)
error=0

