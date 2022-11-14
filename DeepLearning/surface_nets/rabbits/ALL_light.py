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
import logging
import itertools



device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math


NUMBER_SAMPLES=500
STRING="hull_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 64
NUM_WORKERS = int(os.cpu_count() / 2)
LATENT_DIM=3
LOGGING=0
MAX_EPOCHS=500


def volume_tetra(M):
    return abs(torch.linalg.det(M))/6


def volume(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume

def getinfo(stl,flag):
    mesh=meshio.read(stl)
    points=torch.tensor(mesh.points.astype(np.float32)).float()
    if flag==True:
        triangles=torch.tensor(mesh.cells_dict['triangle'].astype(np.int64)).long()


    else:
        triangles=0
        
    return points,triangles
    

class Data(LightningDataModule):
    def get_size(self):
        temp,self.M=getinfo(STRING.format(0),True)
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
        self.setupped=0


    ###UNCOMMENT FOR MULTICPU training
    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        if self.setupped==0:
            self.data=torch.zeros(self.num_samples,self.get_size()[1],self.get_size()[2])
            for i in range(0,self.num_samples):
                if i%100==0:
                    print(i)
                self.data[i],_,=getinfo(STRING.format(i),False)
            # Assign train/val datasets for use in dataloaders
            self.data_train, self.data_val,self.data_test = random_split(self.data, [math.floor(0.5*self.num_samples), math.floor(0.3*self.num_samples),self.num_samples-math.floor(0.5*self.num_samples)-math.floor(0.3*self.num_samples)])    
            self.setupped=1
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




points,triangles=getinfo(STRING.format(0),True)
volume_const=volume(points[triangles])


class VolumeNormalizer(nn.Module):
    def __init__(self,M):
        super().__init__()
        self.M=M
        
    def forward(self, x):
        temp_shape=x.shape
    
        x=x.reshape(x.shape[0],-1,3)
        vol=((x[:,self.M].det().abs().sum(1)/6)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)    
        x=x*(volume_const/vol)**0.5
        return x.reshape(temp_shape)
    

def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum()

def kl_divergence(z, mu, std):
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


class LBR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.batch(self.lin(x)))


class LSL(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=torch.nn.utils.parametrizations.spectral_norm(nn.Linear(in_features, out_features))
        self.relu=nn.LeakyReLU()
    
    def forward(self,x):
        return self.relu(self.lin(x))

class Decoder_base(nn.Module):
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

        
class Discriminator_base(nn.Module):
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
        result=self.sigmoid(self.fc4(result))
        return result


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

        def __init__(self, latent_dim, hidden_dim, data_shape,M):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape, M)

        def forward(self,x):
            return self.decoder_base(x)

    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = LATENT_DIM,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        return {"optimizer": optimizer}
    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)
        if var==None:
            var=torch.ones(1,self.latent_dim)
        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        temp=self.decoder(z)
        return temp
    

class AE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim, hidden_dim, data_shape)
            
        def forward(self,x):
            x_hat=self.encoder_base(x)
            return x_hat
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim,data_shape,M):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape, M)

        def forward(self,x):
            return self.decoder_base(x)
    
    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = LATENT_DIM,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M1=M
        self.latent_dim=latent_dim
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)
        
    def forward(self, x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat.reshape(x.shape).reshape(x.shape)

    def training_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        loss = L2_loss(batch_hat,batch)
        if LOGGING:
            self.log("train_ae_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        loss = L2_loss(batch_hat,batch)
        if LOGGING:
            self.log("validation_ae_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        z=self.encoder(batch)
        batch_hat=self.decoder(z).reshape(batch.shape)
        loss = L2_loss(batch_hat,batch)
        if LOGGING:
            self.log("test_ae_loss", loss)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
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
    
    
class BEGAN(LightningModule):
    
    
    class Generator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,M):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape,M)
            
        def forward(self,x):
            x_hat=self.decoder_base(x)
            return x_hat
        
    class Discriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,M):
            super().__init__()
            self.encoder=Encoder_base(latent_dim, hidden_dim, data_shape)
            self.decoder=Decoder_base(latent_dim, hidden_dim, data_shape, M)
             
        def forward(self,x):
            x_hat=self.decoder(self.encoder(x))
            return x_hat
         
    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = LATENT_DIM,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.discriminator = self.Discriminator(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
        self.generator = self.Generator(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim,M=self.M)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        
    def forward(self, x):
        x_hat=self.discriminator(x)
        return x_hat.reshape(x.shape)
    
    def disc_loss(self, x):
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
            loss=self.disc_loss(batch_p)
            if LOGGING:
                self.log("train_generator_loss", loss)
            return loss
        

        if optimizer_idx==1:    
            loss_disc=self.disc_loss(batch)-k*self.disc_loss(batch)
            loss_gen=self.disc_loss(batch_p)
            if LOGGING:
                self.log("train_discriminagtor_loss", loss_disc)
            diff = torch.mean(gamma * loss_disc - loss_gen)
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)
            return loss_disc
        
            
                
    
    def validation_step(self, batch, batch_idx):
        if LOGGING:
            self.log("val_began_loss", self.disc_loss(batch))
        return self.disc_loss(batch)

        
    def test_step(self, batch, batch_idx):
        if LOGGING:
            self.log("test_began_loss", self.disc_loss(batch))
        return self.disc_loss(batch)
        

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

class GAN(LightningModule):
    
    class Generator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape,M):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape,M)
            
        def forward(self,x):
            x_hat=self.decoder_base(x)
            return x_hat
        
    class Discriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape):
            super().__init__()
            self.discriminator=Discriminator_base(latent_dim, hidden_dim, data_shape)
             
        def forward(self,x):
            x_hat=self.discriminator(x)
            return x_hat

    
    def __init__(self,data_shape,M,hidden_dim: int= 400,latent_dim: int = LATENT_DIM,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.generator = self.Generator(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
        self.discriminator=self.Discriminator(data_shape=self.data_shape, latent_dim=self.latent_dim,hidden_dim=self.hparams.hidden_dim)
        
    def forward(self):
        z=torch.randn(self.hparams.latent_dim)
        x_hat=self.generator(z)
        return x_hat
    
    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss
    
    def adversarial_loss(self,y_hat,y):
        return F.binary_cross_entropy(y_hat, y)


    def training_step(self, batch, batch_idx, optimizer_idx ):
        z=torch.randn(len(batch), self.hparams.latent_dim).type_as(batch)
        batch_hat=self.generator(z)
        
        if optimizer_idx==0:
            g_loss = -torch.log(self.discriminator(batch_hat)).sum()
            if LOGGING:
                self.log("gan_gen_train_loss", g_loss)
            return g_loss
        
        if optimizer_idx==1:
            real_loss = -torch.log(self.discriminator(batch)).sum()
            fake_loss = torch.log(self.discriminator(batch_hat)).sum()
            tot_loss= (real_loss+fake_loss)/2
            if LOGGING:
                self.log("gan_disc_train_loss", tot_loss)
            return tot_loss
            
            
    
    def validation_step(self, batch, batch_idx):
        z = torch.randn(1,self.latent_dim).type_as(batch)
        generated=self.generator(z)
        true=batch.reshape(-1,generated.shape[1])
        loss=torch.min(torch.norm(generated-true,dim=1))
        if LOGGING:
            self.log("gan_val_loss", loss)
        return loss
        
    
    def test_step(self, batch, batch_idx):
        z = torch.randn(1,self.latent_dim).type_as(batch)
        generated=self.generator(z)
        true=batch.reshape(-1,generated.shape[1])
        loss=torch.min(torch.norm(generated-true,dim=1))
        if LOGGING:
            self.log("gan_test_loss", loss)
        return loss

        
    def configure_optimizers(self):#
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.02) #0.02
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002) #0.0002

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return [optimizer_g,optimizer_d], []
        #return {"optimizer": [optimizer_ae,optimizer_disc], "lr_scheduler": [scheduler_ae,scheduler_disc], "monitor": ["train_loss","train_loss"]}

    def sample_mesh(self):
        z = torch.randn(1,self.latent_dim)
        temp=self.generator(z)
        return temp


class AAE(LightningModule):
    
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim, hidden_dim,data_shape)
            
        def forward(self,x):
            x_hat=self.encoder_base(x)
            return x_hat

    class Decoder(nn.Module):

        def __init__(self, latent_dim, hidden_dim,data_shape,M):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape, M)

        def forward(self,x):
            return self.decoder_base(x)
    
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape):
            super().__init__()
            self.discriminator=Discriminator_base(latent_dim, hidden_dim, latent_dim)
             
        def forward(self,x):
            x_hat=self.discriminator(x)
            return x_hat



    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = LATENT_DIM,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE, ae_hyp=0.999,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.ae_hyp=ae_hyp
        self.M=M
        self.latent_dim=latent_dim
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim,data_shape=self.data_shape,M=self.M)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)
        self.discriminator=self.Discriminator(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)
        
    def forward(self, x):
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
        z=torch.randn(len(batch), self.hparams.latent_dim).type_as(batch)
        ones=torch.ones(len(batch)).type_as(batch)
        zeros=torch.zeros(len(batch)).type_as(batch)

        
        if optimizer_idx==0:
            batch_hat=self.decoder(z_enc).reshape(batch.shape)
            ae_loss = self.ae_hyp*self.ae_loss(batch_hat,batch)+(1-self.ae_hyp)*self.adversarial_loss(self.discriminator(z_enc).reshape(ones.shape), ones)
            if LOGGING:
                self.log("train_ae_loss", ae_loss)
            return ae_loss
        
        if optimizer_idx==1:
            real_loss = self.adversarial_loss(self.discriminator(z).reshape(ones.shape), ones)
            fake_loss = self.adversarial_loss(self.discriminator(z_enc).reshape(zeros.shape), zeros)
            tot_loss= (real_loss+fake_loss)/2
            if LOGGING:
                self.log("train_aee_loss", tot_loss)
            return tot_loss
            
    def validation_step(self, batch, batch_idx):
        z=torch.randn(len(batch), self.hparams.latent_dim)
        z_enc=self.encoder(batch)
        ones=torch.ones(len(batch))
        zeros=torch.zeros(len(batch))
        batch_hat=self.decoder(z_enc).reshape(batch.shape)
        ae_loss = self.ae_loss(batch_hat,batch)
        if LOGGING:
            self.log("val_aee_loss", ae_loss)
        return ae_loss
        
    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def test_step(self, batch, batch_idx):
        z=torch.randn(len(batch), self.hparams.latent_dim)
        z_enc=self.encoder(batch)
        ones=torch.ones(len(batch))
        zeros=torch.zeros(len(batch))
        batch_hat=self.decoder(z_enc).reshape(batch.shape)
        ae_loss = self.ae_loss(batch_hat,batch)
        if LOGGING:
            self.log("test_aee_loss", ae_loss)
        return ae_loss

    def configure_optimizers(self):
        optimizer_ae = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-3)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return [optimizer_ae,optimizer_disc], []
    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)
        if var==None:
            var=torch.ones(1,self.latent_dim)
        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        temp=self.decoder(z)
        return temp
    
class VAEGAN(LightningModule):
    
    
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

        def __init__(self, latent_dim, hidden_dim,data_shape,M):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim, hidden_dim, data_shape,M)

        def forward(self,x):
            return self.decoder_base(x)
    
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim, hidden_dim,data_shape):
            super().__init__()
            self.discriminator=Discriminator_base(latent_dim, hidden_dim, data_shape)
             
        def forward(self,x):
            x_hat=self.discriminator(x)
            return x_hat

    
    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = 1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE, ae_hyp=0.999,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        self.ae_hyp=ae_hyp
        self.latent_dim=latent_dim
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim,data_shape=self.data_shape,M=self.M)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim ,hidden_dim=self.hparams.hidden_dim)
        self.discriminator=self.Discriminator(hidden_dim=self.hparams.hidden_dim,data_shape=self.data_shape,latent_dim=self.latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        
    def forward(self, x):
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
        mu,sigma=self.encoder(batch)
        q = torch.distributions.Normal(mu, sigma)
        z_sampled = q.rsample().type_as(batch)
        batch=batch.reshape(-1,np.prod(self.data_shape))
        disl=self.discriminator(batch)
        batch_hat = self.decoder(z_sampled).reshape(batch.shape)
        disl_hat=self.discriminator(batch_hat.reshape(-1,np.prod(self.data_shape)))
        z_p=torch.randn(len(batch), self.hparams.latent_dim).type_as(batch)
        batch_p=self.decoder(z_p)
        ones=torch.ones(len(batch)).type_as(batch)
        zeros=torch.zeros(len(batch)).type_as(batch)
        # reconstruction loss
        recon_loss = gaussian_likelihood(disl_hat, self.log_scale, disl)

        # kl
        kl = kl_divergence(z_sampled, mu, sigma)

        
        if optimizer_idx==0:
            loss=kl-recon_loss
            if LOGGING:
                self.log("train_encoder_loss", loss.mean())
            return loss.mean()
        

        if optimizer_idx==1:    
            ae_loss = -self.ae_hyp*recon_loss-0.5*(1-self.ae_hyp)*torch.log(self.discriminator(batch_hat)).sum()-0.5*(1-self.ae_hyp)*torch.log(self.discriminator(batch_p)).sum()
            if LOGGING:
                self.log("train_decoder_loss", ae_loss)
            return ae_loss
        
        if optimizer_idx==2:
            real_loss = -torch.log(self.discriminator(batch)).sum()
            fake_loss_1 = 0.5*torch.log(self.discriminator(batch_hat)).sum()
            fake_loss_2 =0.5*torch.log(self.discriminator(batch_p)).sum()
            tot_loss= (real_loss+fake_loss_1+fake_loss_2)/2
            if LOGGING:
                self.log("train_discriminator_loss", tot_loss)
            return tot_loss
            
                
    
    def validation_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        if LOGGING:
            self.log("val_vaegam_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)

        
    def test_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        if LOGGING:
            self.log("test_vaegan_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)
    
        
    def get_latent(self,data):
        return self.encoder.forward(data)[0]


    

    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer_enc=torch.optim.Adam(self.encoder.parameters(), lr=0.02)#0.02
        optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=0.02) #0.02
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0050) #0.050

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return [optimizer_enc,optimizer_dec,optimizer_disc], []
        #return {"optimizer": [optimizer_ae,optimizer_disc], "lr_scheduler": [scheduler_ae,scheduler_disc], "monitor": ["train_loss","train_loss"]}

    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)
        if var==None:
            var=torch.ones(1,self.latent_dim)
        z = torch.sqrt(var)*torch.randn(1,self.latent_dim)+mean
        temp=self.decoder(z)
        return temp

data=Data()

d=dict

d={
  VAE: "VAE",
  VAEGAN: "VAEGAN",
  AE: "AE",
  BEGAN: "BEGAN",
  GAN: "GAN",
  AAE: "AAE"
   }

for wrapper, name in d.items():
    torch.manual_seed(0)
    if not LOGGING:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
   
    if LOGGING:
        if AVAIL_GPUS:
            trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,log_every_n_steps=1)
        else:
            trainer=Trainer(max_epochs=MAX_EPOCHS,log_every_n_steps=1)
    else:
        if AVAIL_GPUS:
            trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
        else:
            trainer=Trainer(max_epochs=MAX_EPOCHS,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
    
    model=wrapper(data.get_size(),data.M)
    #LITTLE HACK
    trainer.fit(model, data)
    trainer.validate(datamodule=data)
    trainer.test(datamodule=data)

    model.eval()
    temp = model.sample_mesh().detach()
    temp=temp.reshape(1,-1,3)
    meshio.write_points_cells('test_'+name+'.stl',temp.reshape(-1,3).numpy(),[("triangle", data.M)])
    error=0
    temparr=torch.zeros(NUMBER_SAMPLES,*tuple(temp.shape))
    vol=torch.zeros(NUMBER_SAMPLES,0)
    for i in range(NUMBER_SAMPLES):
        temp = model.sample_mesh().reshape(1,-1,3).detach()
        temparr[i]=temp
        true=data.data.reshape(data.num_samples,-1)
        temp=temp.reshape(1,-1)
        error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
        vol[i]=volume(temp.reshape(-1,3)[data.M])

    variance=torch.sum(torch.var(temparr,dim=0))
    variance_vol=torch.sum(torch.var(vol,dim=0))
    print("Average distance between sample (prior) and data of ",  name, " is", error.item())
    print("Variance from prior of ", name, "is", variance.item())
    print("Volume variance prior of ", name, "is", variance_vol.item())

    if name!="GAN" and name!="BEGAN":
        error=0
        latent=model.get_latent(data.data[:])
        mean=torch.mean(latent)
        var=torch.var(latent)
        for i in range(NUMBER_SAMPLES):
            temp = model.sample_mesh(mean,var).reshape(1,-1,3).detach()
            temparr[i]=temp
            true=data.data.reshape(data.num_samples,-1)
            temp=temp.reshape(1,-1)
            error=error+torch.min(torch.norm(temp-true,dim=1))/torch.norm(temp)/100
            vol[i]=volume(temp.reshape(-1,3)[data.M])
        variance_vol=torch.sum(torch.var(vol,dim=0))
        variance=torch.sum(torch.var(temparr,dim=0))
        print("Average distance between sample (posterior) and data of ", name, " is", error.item())
        print("Variance from posterior of ", name, "  is", variance.item())
        print("Volume variance posterior of ", name, "is", variance_vol.item())


temparr=temparr.reshape(NUMBER_SAMPLES,-1,3)
for i in range(NUMBER_SAMPLES):
    temp=data.data[i]
    temp=temp.reshape(-1,3)
    temparr[i]=temp[temp[:,2]>0]

variance=torch.sum(torch.var(temparr,dim=0))
print("Variance of data is", variance.item())

