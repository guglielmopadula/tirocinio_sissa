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
LOGGING=1

def poly(y,a,b,c,d):
    return a*y**3+b*y**2+c*y+d

NUMBER_SAMPLES=500
STRING="hull_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 64
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


class LBR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.batch(self.lin(x)))


def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum()

def kl_divergence( z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl

def L2_loss(x_hat, x):
    loss=F.mse_loss(x, x_hat, reduction="none")
    loss=loss.mean()
    return loss


"""
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,M1,M2,M3,local_indices):
        super().__init__()
        self.data_shape=data_shape
        self.M1=M1
        self.M2=M2
        self.M3=M3
        self.temp_zero=temp_zero
        self.local_indices=local_indices
        self.fc1 = LBR(latent_dim, hidden_dim)
        self.fc2 = LBR(hidden_dim, hidden_dim)
        self.fc3 = LBR(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(np.prod(self.data_shape)))
        self.fc5=VolumeNormalizer(self.temp_zero,self.M1,self.M2,self.M3,self.local_indices)
        self.relu = nn.ReLU()

    def forward(self, z):
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
        #self.batch_mu=nn.BatchNorm1d(self.latent_dim)
        self.batch_mu=nn.BatchNorm1d(self.latent_dim,affine=False,track_running_stats=False)
        self.batch_sigma=nn.BatchNorm1d(self.latent_dim)
        self.fc32 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        sigma=self.batch_sigma(self.fc22(hidden))
        sigma=1+sigma/torch.linalg.norm(sigma)*torch.tanh(torch.linalg.norm(sigma))*(1/(math.pi*8))
        mu=self.batch_mu(mu)
        #mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        return mu,sigma





        
class VAE(LightningModule):
    
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
        self.decoder = Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M1=self.M1,M2=self.M2,M3=self.M3,local_indices=self.local_indices,temp_zero=self.temp_zero)
        self.encoder = Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
    def forward(self, x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat.reshape(x.shape).reshape(x.shape)
    
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum()

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    

    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss


    def training_step(self, batch, batch_idx):
        
        # encode x to get the mu and variance parameters
        mu,sigma = self.encoder(batch)
        
        
        # sample z from q
        q = torch.distributions.Normal(mu, sigma)
        z_sampled = q.rsample()

        # decoded
        batch_hat = self.decoder(z_sampled).reshape(batch.shape)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(batch_hat, self.log_scale, batch)

        # kl
        kl = self.kl_divergence(z_sampled, mu, sigma)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        
        self.log("train_vae_loss", elbo)
        return elbo
    
    def validation_step(self, batch, batch_idx):
         
        # encode x to get the mu and variance parameters
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        self.log("val_vae_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)
    
    def test_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        self.log("test_vae_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return {"optimizer": optimizer}

    def sample_mesh(self,mean,var):
        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        temp=self.decoder(z)
        return temp

"""

class Decoder_base(nn.Module):
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
        self.relu = nn.ReLU()

    def forward(self, z):
        result=self.fc4(self.fc3(self.fc2(self.fc1(z))))
        result=self.fc5(result)
        result=result.view(result.size(0),-1)
        return result
    
    

class Encoder_base(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.latent_dim=latent_dim
        self.fc1 = LBR(int(np.prod(self.data_shape)),hidden_dim)
        self.fc21 = LBR(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
        self.batch_mu=nn.BatchNorm1d(self.latent_dim,affine=False,track_running_stats=False)

    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        mu=self.batch_mu(mu)
        return mu

class Variance_estimator(nn.Module):
    def __init__(self,latent_dim,hidden_dim,data_shape):
        super().__init__()
        self.latent_dim=latent_dim
        self.fc22 = nn.Linear(latent_dim, latent_dim)
        self.batch_sigma=nn.BatchNorm1d(self.latent_dim)

    
    def forward(self,mu):
        sigma=self.batch_sigma(self.fc22(mu))
        sigma=torch.exp(sigma)
        return sigma

class VAE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim, hidden_dim, data_shape)
            self.variance_estimator=Variance_estimator(latent_dim, hidden_dim, data_shape)
            
        def forward(self,x):
            mu=self.encoder_base(x)
            sigma=self.variance_estimator(mu)
            return mu,sigma
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim, data_shape,temp_zero,M1,M2,M3,local_indices):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape, temp_zero,M1,M2,M3,local_indices)

        def forward(self,x):
            return self.decoder_base(x)

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
        self.decoder = self.Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M1=self.M1,M2=self.M2,M3=self.M3,local_indices=self.local_indices,temp_zero=self.temp_zero)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
    def forward(self, x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat.reshape(x.shape).reshape(x.shape)
    
    def training_step(self, batch, batch_idx):
        
        # encode x to get the mu and variance parameters
        mu,sigma = self.encoder(batch)
        
        
        # sample z from q
        q = torch.distributions.Normal(mu, sigma)
        z_sampled = q.rsample()

        # decoded
        batch_hat = self.decoder(z_sampled).reshape(batch.shape)

        # reconstruction loss
        recon_loss = gaussian_likelihood(batch_hat, self.log_scale, batch)

        # kl
        kl = kl_divergence(z_sampled, mu, sigma)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        if LOGGING:
            self.log("train_vae_loss", L2_loss(batch,batch_hat))
        return elbo
    
    
    def get_latent(self,data):
        return self.encoder.forward(data)[0]

    
    def validation_step(self, batch, batch_idx):
         
        # encode x to get the mu and variance parameters
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        if LOGGING:
            self.log("val_vae_loss", L2_loss(batch,batch_hat))
        return L2_loss(batch,batch_hat)
    
    def test_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        if LOGGING:
            self.log("test_vae_loss", L2_loss(batch,batch_hat))
        return L2_loss(batch,batch_hat)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}
    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)
        if var==None:
            var=torch.ones(1,self.latent_dim)
        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        temp=self.decoder(z)
        return temp
    



if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1)
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = VAE(data.get_size(),data.temp_zero,data.local_indices,data.M1,data.M2,data.M3)
trainer.fit(model, data)
trainedlatent=model.encoder.forward(data.data_train[0:len(data.data_train)])[0].cpu().detach().numpy()
testedlatent=model.encoder.forward(data.data_test[0:len(data.data_test)])[0].cpu().detach().numpy()
validatedlatent=model.encoder.forward(data.data_val[0:len(data.data_val)])[0].cpu().detach().numpy()
alllatent=model.encoder.forward(data.data[0:len(data.data[:])])[0].cpu().detach().numpy()

trainer.validate(datamodule=data)
trainer.test(datamodule=data)
mean=torch.mean(torch.tensor(alllatent))
var=torch.var(torch.tensor(alllatent))
model.eval()
temp = model.sample_mesh(mean,var)
oldmesh=data.oldmesh.clone().numpy()
oldmesh[oldmesh[:,2]>0]=temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy()
meshio.write_points_cells('test.stl',oldmesh,[("triangle", data.oldM)])
error=0
temparr=torch.zeros(500,*tuple(temp.shape))
for i in range(500):
    temp = model.sample_mesh(torch.tensor(0),torch.tensor(1))
    temparr[i]=temp
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
variance=torch.sum(torch.var(temparr,dim=0))
print("Average distance between sample (prior) and data is", error)
print("Variance from prior is", variance)
error=0
for i in range(500):
    temp = model.sample_mesh(mean,var)
    temparr[i]=temp
    true=data.data.reshape(-1,temp.shape[1])
    error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
variance=torch.sum(torch.var(temparr,dim=0))
print("Average distance between sample (posterior) and data is", error)
print("Variance from posterior is", variance)

#vae.load_state_dict(torch.load("cube.pt"))

#vae.load_state_dict(torch.load("cube.pt"))
