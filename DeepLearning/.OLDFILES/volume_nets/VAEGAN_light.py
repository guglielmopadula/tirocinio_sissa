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


ae_hyp=0.999

NUMBER_SAMPLES=200
STRING="bulbo_{}.vtk"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 200
NUM_WORKERS = int(os.cpu_count() / 2)

def getinfo(vtk):
    mesh=meshio.read(vtk)
    points=torch.tensor(mesh.points.astype(np.float32))
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
        ones=torch.ones(xmatrix.shape[0],xmatrix.shape[1],xmatrix.shape[2],1).type_as(xmatrix)
        x=x/((torch.cat((x[:,self.M],ones),3).det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
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
        self.fc1 = LBR(int(np.prod(self.data_shape)),hidden_dim)
        self.fc21 = LBR(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
        self.batch_mu=nn.BatchNorm1d(1)
        self.batch_sigma=nn.BatchNorm1d(1)
        self.fc32 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(x.size(0),-1)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        sigma=self.batch_sigma(self.fc22(hidden))
        #sigma=self.fc32(sigma)
        sigma=1+sigma/torch.linalg.norm(sigma)*torch.tanh(torch.linalg.norm(sigma))*(1/(math.pi*8))
        mu=self.batch_mu(mu)
        mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        return mu,sigma


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
        result=self.sigmoid(self.fc4(result))
        return result


class VAEGAN(LightningModule):
    
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
        self.discriminator=Discriminator(hidden_dim=self.hparams.hidden_dim,data_shape=self.data_shape,latent_dim=self.latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        
    def forward(self, x):
        print(x.shape)
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat.reshape(x.shape)
    
    def ae_loss(self, x_hat, x):
        loss=F.mse_loss(x, x_hat, reduction="none")
        loss=loss.mean()
        return loss
    
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
        recon_loss = self.gaussian_likelihood(disl_hat, self.log_scale, disl)

        # kl
        kl = self.kl_divergence(z_sampled, mu, sigma)

        
        if optimizer_idx==0:
            loss=kl-recon_loss
            self.log("train_encoder_loss", loss.mean())
            return loss.mean()
        

        if optimizer_idx==1:    
            ae_loss = -ae_hyp*recon_loss+0.5*(1-ae_hyp)*self.adversarial_loss(self.discriminator(batch_hat).reshape(ones.shape), ones)+0.5*(1-ae_hyp)*self.adversarial_loss(self.discriminator(batch_p).reshape(ones.shape), ones)
            self.log("train_decoder_loss", ae_loss)
            return ae_loss
        
        if optimizer_idx==2:
            real_loss = self.adversarial_loss(self.discriminator(batch).reshape(ones.shape), ones)
            fake_loss_1 = 0.5*self.adversarial_loss(self.discriminator(batch_hat).reshape(zeros.shape), zeros)
            fake_loss_2 = 0.5*self.adversarial_loss(self.discriminator(batch_p).reshape(zeros.shape), zeros)
            tot_loss= (real_loss+fake_loss_1+fake_loss_2)/2
            self.log("train_discriminator_loss", tot_loss)
            return tot_loss
            
            
    
    def validation_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        self.log("val_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)

        
    def test_step(self, batch, batch_idx):
        mu,sigma = self.encoder(batch)
        batch_hat=self.decoder(mu).reshape(batch.shape)
        self.log("val_loss", self.ae_loss(batch,batch_hat))
        return self.ae_loss(batch,batch_hat)
        

    

    def configure_optimizers(self):
        optimizer_enc=torch.optim.Adam(self.encoder.parameters(), lr=0.02)
        optimizer_dec = torch.optim.Adam(self.decoder.parameters(), lr=0.02)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.05)

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        return [optimizer_enc,optimizer_dec,optimizer_disc], []
        #return {"optimizer": [optimizer_ae,optimizer_disc], "lr_scheduler": [scheduler_ae,scheduler_disc], "monitor": ["train_loss","train_loss"]}

    def sample_mesh(self,mean,var):
        z = torch.randn(1,1)*torch.sqrt(var)+mean
        temp=self.decoder(z)
        return temp

if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1)
else:
    trainer=Trainer(max_epochs=500,log_every_n_steps=1)
data=Data()
model = VAEGAN(data.get_size(),data.M)
trainer.fit(model, data)
trainedlatent=model.encoder.forward(data.data_train[0:len(data.data_train)])[0].cpu().detach().numpy()
testedlatent=model.encoder.forward(data.data_test[0:len(data.data_test)])[0].cpu().detach().numpy()
validatedlatent=model.encoder.forward(data.data_val[0:len(data.data_val)])[0].cpu().detach().numpy()
trainer.validate(datamodule=data)
trainer.test(datamodule=data)
mean=torch.mean(torch.tensor(trainedlatent))
var=torch.var(torch.tensor(trainedlatent))
model.eval()
temp = model.sample_mesh(torch.tensor(0),torch.tensor(1))
meshio.write_points_cells('test.vtk',temp.reshape(model.data_shape[1],model.data_shape[2]).detach().numpy().tolist(),[("triangle", data.triangles),("tetra", data.M)])
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
