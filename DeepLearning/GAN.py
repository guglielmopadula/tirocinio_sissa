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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math


NUMBER_SAMPLES=100
STRING="bulbo_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
BATCH_SIZE = 50 
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
        self.data=torch.zeros(self.num_samples,self.get_size()[1],self.get_size()[2])
        for i in range(0,self.num_samples):
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





class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape,M):
        super().__init__()  
        self.data_shape=data_shape
        self.M=M
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(data_shape))),
            VolumeNormalizer(self.M))   


    def forward(self, z):
        x = self.model(z)
        x = x.view(x.size(0), *self.data_shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(data_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x=x.view(x.size(0),-1)
        validity = self.model(x)
        return validity







        
class GAN(LightningModule):
    
    def __init__(self,data_shape,M,hidden_dim: int= 300,latent_dim: int = 100,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.M=M
        # networks
        self.data_shape = data_shape
        self.generator = Generator(latent_dim=self.hparams.latent_dim,hidden_dim=self.hparams.hidden_dim ,data_shape=self.data_shape,M=self.M)
        self.discriminator = Discriminator(data_shape=self.data_shape, hidden_dim=self.hparams.hidden_dim)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        self.g_losses=[]
        self.d_losses=[]
        
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        meshes = batch

        # sample noise
        z = torch.randn(meshes.shape[0], self.hparams.latent_dim)
        z = z.type_as(meshes)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_meshes = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(meshes.size(0), 1)
            valid = valid.type_as(meshes)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.g_losses.append(g_loss.clone().cpu().item())
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(meshes.size(0), 1)
            valid = valid.type_as(meshes)

            real_loss = self.adversarial_loss(self.discriminator(meshes), valid)

            # how well can it label as fake?
            fake = torch.zeros(meshes.size(0), 1)
            fake = fake.type_as(meshes)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.d_losses.append(d_loss.clone().cpu().item())
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def sample_mesh(self):
        z = torch.randn(5, self.hparams.latent_dim)
        a=self.generator.forward(z)[0]
        return a.reshape(self.data_shape[1],self.data_shape[2])
    


if AVAIL_GPUS:
    trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=1000,log_every_n_steps=1)
else:
    trainer=Trainer(max_epochs=1000,log_every_n_steps=1)
data=Data()
model = GAN(data.get_size(),data.M)
trainer.fit(model, data)
plt.plot([i for i in range(len(model.g_losses))],model.g_losses,label="generator")
plt.plot([i for i in range(len(model.d_losses))],model.d_losses,label="discriminator")
plt.legend(loc="upper left")
plt.show()

newpoints=model.sample_mesh()
temp=applytopology(newpoints, data.M)
newmesh = np.zeros(len(temp), dtype=mesh.Mesh.dtype)
newmesh['vectors'] = temp.cpu().detach().numpy().copy()
mymesh = mesh.Mesh(newmesh.copy())
mymesh.save('test.stl', mode=stl.Mode.ASCII)

#vae.load_state_dict(torch.load("cube.pt"))


