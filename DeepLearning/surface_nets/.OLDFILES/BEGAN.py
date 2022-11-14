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
from xitorch.optimize import rootfinder
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


def poly(y,a,b,c,d):
    return a*y**3+b*y**2+c*y+d

NUMBER_SAMPLES=500
STRING="hull_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 500
NUM_WORKERS = int(os.cpu_count() / 2)


def mysqrt(x):
    return torch.sqrt(torch.max(x,torch.zeros_like(x)))

def myacos(x):
    return torch.acos(torch.min(0.9999*torch.ones_like(x),torch.max(x,-0.9999*torch.ones_like(x))))

def myacosh(x):
    return torch.acosh(torch.max(1.00001*torch.ones_like(x),x))

def multi_cubic(a, b, c, d):
    p=(3*a*c-b**2)/(3*a**2)
    q=(2*b**3-9*a*b*c+27*a**2*d)/(27*a**3)
    temp1=(p>=0).int()*(-2*torch.sqrt(torch.abs(p)/3)*torch.sinh(1/3*torch.asinh(3*q/(2*torch.abs(p))*torch.sqrt(3/torch.abs(p)))))
    temp2=(torch.logical_and(p<0,(4*p**3+27*q**2)>0).int()*(-2*torch.abs(q)/q)*torch.sqrt(torch.abs(p)/3)*torch.cosh(1/3*myacosh(3*torch.abs(q)/(2*torch.abs(p))*torch.sqrt(3/torch.abs(p)))))
    temp3=torch.logical_and(p<0,(4*p**3+27*q**2)<0).int()*2*mysqrt(torch.abs(p)/3)*torch.max(torch.stack((torch.cos(1/3*myacos(3*q/(2*p)*torch.sqrt(3/torch.abs(p)))-2*torch.pi*0/3),torch.cos(1/3*myacos(3*q/(2*p)*torch.sqrt(3/torch.abs(p)))-2*torch.pi*1/3),torch.cos(1/3*myacos(3*q/(2*p)*torch.sqrt(3/torch.abs(p)))-2*torch.pi*2/3))))
    return temp1+temp2+temp3-b/(3*a)

'''
def multi_cubic(a, b, c, d):
    y=torch.zeros(len(a))
    p=(3*a*c-b**2)/(3*a**2)
    q=(2*b**3-9*a*b*c+27*a**2*d)/(27*a**3)

    for i in range(len(a)):
        if p[i]>=0:
            y[i]=(-2*torch.sqrt(p[i]/3)*torch.sinh(1/3*torch.asinh(3*q[i]/(2*p[i])*torch.sqrt(3/p[i]))))
        if torch.logical_and(p[i]<0,(4*p[i]**3+27*q[i]**2)>0):
            y[i]=(-2*torch.abs(q[i])/q[i])*torch.sqrt(torch.abs(p[i])/3)*torch.cosh(1/3*torch.acosh(3*torch.abs(q[i])/(2*torch.abs(p[i]))*torch.sqrt(3/torch.abs(p[i]))))
        if torch.logical_and(p[i]<0,(4*p[i]**3+27*q[i]**2)<0):
            y[i]=2*torch.sqrt(torch.abs(p[i])/3)*torch.max(torch.stack((torch.cos(1/3*torch.acos(3*q[i]/(2*p[i])*torch.sqrt(3/torch.abs(p[i])))-2*torch.pi*0/3),torch.cos(1/3*torch.acos(3*q[i]/(2*p[i])*torch.sqrt(3/torch.abs(p[i])))-2*torch.pi*1/3),torch.cos(1/3*torch.acos(3*q[i]/(2*p[i])*torch.sqrt(3/torch.abs(p[i])))-2*torch.pi*2/3))))
    return y-b/(3*a)
'''
    
def getinfo(stl,flag):
    mesh=meshio.read(stl)
    points_old=torch.tensor(mesh.points.astype(np.float32))
    points=points_old[points_old[:,2]>0]
    points_zero=points_old[points_old[:,2]>=0]
    if flag==True:
        newmesh_indices_global=np.arange(len(mesh.points))[mesh.points[:,2]>0].tolist()
        triangles=torch.tensor(mesh.cells_dict['triangle'].astype(np.int64))
        triangles=triangles.long()
        newtriangles=[]
        for T in triangles:
            if T[0] in newmesh_indices_global and T[1] in newmesh_indices_global and T[2] in newmesh_indices_global:
                newtriangles.append([newmesh_indices_global.index(T[0]),newmesh_indices_global.index(T[1]),newmesh_indices_global.index(T[2])])
        newmesh_indices_global_zero=np.arange(len(mesh.points))[mesh.points[:,2]>=0].tolist()
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])
        newmesh_indices_local=np.arange(len(points_zero))[points_zero[:,2]>0].tolist()
        newtriangles_local_3=[]
        newtriangles_local_2=[]
        newtriangles_local_1=[]
        for T in newtriangles_zero:
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==3:
                newtriangles_local_3.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==2:
                newtriangles_local_2.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==1:
                newtriangles_local_1.append([T[0],T[1],T[2]])


    else:
        triangles=0
        newtriangles=0
        newmesh_indices_local=0
        newtriangles_zero=0
        newtriangles_local_1=0
        newtriangles_local_2=0
        newtriangles_local_3=0
        
    return points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3


def volume_tetra(M):
    return abs(np.linalg.det(M))/6


def volume(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume

class Data(LightningDataModule):
    def get_size(self):
        temp,self.temp_zero,self.oldmesh,self.local_indices,self.oldM,_,self.M1,self.M2,self.M3=getinfo(STRING.format(0),True)
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
                self.data[i],_,_,_,_,_,_,_,_=getinfo(STRING.format(i),False)
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




_,points_zero,_,_,_,newtriangles_zero,_,_,_=getinfo(STRING.format(0),True)
volume_const=volume(points_zero[newtriangles_zero])


class VolumeNormalizer(nn.Module):
    def __init__(self,temp_zero,M1,M2,M3,local_indices):
        super().__init__()
        self.M1=M1
        self.M2=M2
        self.M3=M3
        self.temp_zero=temp_zero
        self.local_indices=local_indices

    def forward(self, x):
        temp_shape=x.shape
        temp=self.temp_zero.clone()
        x=x.reshape(x.shape[0],-1,3)
        temp=temp.repeat(x.shape[0],1,1)
        temp[:,self.local_indices,:]=x
        a=((temp[:,self.M3].det().abs().sum(1)/6))
        b=((temp[:,self.M2].det().abs().sum(1)/6))
        c=((temp[:,self.M1].det().abs().sum(1)/6))
        d=-volume_const
        k=multi_cubic(a, b, c, d)
        x=x*(k).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
        return x.reshape(temp_shape)
    
    def forward_single(self,x):
        temp=x.shape
        x=x.reshape(1,-1,3)
        x=x/((x[:,self.M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(1,-1,3)
        x=x*volume_const
        return x.reshape(temp)

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
        self.relu=nn.LeakyReLU()
    
    def forward(self,x):
        return self.relu(self.batch(self.lin(x)))

    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,M1,M2,M3,local_indices):
        super().__init__()
        self.data_shape=data_shape
        self.M1=M1
        self.M2=M2
        self.M3=M3
        self.local_indices=local_indices
        self.temp_zero=temp_zero
        self.fc1 = LBR(latent_dim, hidden_dim)
        self.fc2 = LBR(hidden_dim, hidden_dim)
        self.fc3 = LBR(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.fc5=VolumeNormalizer(self.temp_zero,self.M1,self.M2,self.M3,self.local_indices)
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
        self.batch_mu=nn.BatchNorm1d(self.latent_dim,affine=False,track_running_stats=False)
        self.fc32 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        #sigma=self.fc32(sigma)
        mu=self.batch_mu(mu)
        #mu=(mu-torch.mean(mu,dim=0))/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        return mu


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape,temp_zero,M1,M2,M3,local_indices):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = Decoder(latent_dim, hidden_dim,data_shape,temp_zero,M1,M2,M3,local_indices)
        
    def forward(self,x):
        result=self.fc1(x)
        return result



class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape,temp_zero,M1,M2,M3,local_indices):
        super().__init__()
        self.data_shape=data_shape
        self.fc1 = Encoder(latent_dim, hidden_dim,data_shape)
        self.fc2 = Decoder(latent_dim, hidden_dim,data_shape,temp_zero,M1,M2,M3,local_indices)
        
    def forward(self,x):
        x=x.reshape(-1,int(np.prod(self.data_shape)))
        result=self.fc1(x)
        result=self.fc2(result)
        return result


class BEGAN(LightningModule):
    
    def __init__(self,data_shape,temp_zero,local_indices,M1,M2,M3,hidden_dim: int= 300,latent_dim: int = 3,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M1=M1
        self.M2=M2
        self.M3=M3
        self.local_indices=local_indices
        self.temp_zero=temp_zero
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.discriminator = Discriminator(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M1=self.M1,M2=self.M2,M3=self.M3,local_indices=self.local_indices,temp_zero=self.temp_zero)
        self.generator = Generator(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim,M1=self.M1,M2=self.M2,M3=self.M3,local_indices=self.local_indices,temp_zero=self.temp_zero)
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
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)
            return loss_disc
        
            
                
    
    def validation_step(self, batch, batch_idx):
        self.log("val_began_loss", self.ae_loss(batch))
        return self.ae_loss(batch)

        
    def test_step(self, batch, batch_idx):
        self.log("test_began_loss", self.ae_loss(batch))
        return self.ae_loss(batch)
        

    

    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.02) #0.02
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.05) #0.050

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
    trainer=Trainer(max_epochs=500,log_every_n_steps=1,detect_anomaly=True)
data=Data()
model = BEGAN(data.get_size(),data.temp_zero,data.local_indices,data.M1,data.M2,data.M3)
trainer.fit(model, data)
trainer.validate(datamodule=data)
trainer.test(datamodule=data)
model.eval()
temp = model.sample_mesh()
oldmesh=data.oldmesh.clone().numpy()
oldmesh[oldmesh[:,2]>0]=temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy()
meshio.write_points_cells('test.stl',oldmesh,[("triangle", data.oldM)])
temparr=torch.zeros(100,*tuple(temp.shape))
error=0
for i in range(100):
    temp = model.sample_mesh()
    oldmesh=data.oldmesh.clone().numpy()
    oldmesh[oldmesh[:,2]>0]=temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy()
    print(volume(oldmesh[data.oldM]))
    temparr[i]=temp
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
variance=torch.sum(torch.var(temparr,dim=0))
print("Average distance between sample (prior) and data is", error)
print("Variance from prior is", variance)

