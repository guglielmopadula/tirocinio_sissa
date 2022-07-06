#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data import DataLoader
from stl import mesh
from torch.utils.data import Dataset
import os
import torch
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math
number_samples=100


AVAIL_GPUS = min(0, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

def getinfo(stl):
    your_mesh = mesh.Mesh.from_file(stl)
    myList = list(OrderedSet(tuple(map(tuple,your_mesh.vectors.reshape(3618,3)))))
    K=len(your_mesh)
    array=your_mesh.vectors
    topo=np.zeros((1206,3))
    for i in range(1206):
        for j in range(3):
            topo[i,j]=myList.index(tuple(array[i,j].tolist()))
    N=9*K
    return torch.tensor(array.copy()).reshape(10854),torch.tensor(myList),N,len(myList)*3,torch.tensor(topo, dtype=torch.int64)

    
def applytopology(V,M):
    Q=torch.zeros((M.shape[0],3,3),device=device)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Q[i,j]=V[M[i,j].item()]
    return Q


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, int(np.prod(data_shape)))
        self.relu = nn.ReLU()

    def forward(self, z):
        result=self.fc4(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))
        return result

class Discriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim,data_shape):
        super().__init__()
        self.fc1 = nn.Linear(int(np.prod(data_shape)),hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu=nn.ReLU()
        self.fc5 = nn.Sigmoid()

    def forward(self, x):
        x=x.reshape(-1,N)
        prob=self.fc5(self.relu(self.fc4(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))))
        return prob





data=torch.zeros(number_samples,605,3)
for i in range(0,number_samples):
    meshes,data[i],N,K,M=getinfo("bulbo_{}.stl".format(i))
data=data.to(device)
data.reshape(number_samples,1815)
my_dataset = TensorDataset(data)
my_dataloader = DataLoader(my_dataset) 










        
class GAN(nn.Module):
    def __init__(
     self,
     N,
     hidden_dim: int= 300,
     latent_dim: int = 100,
     lr: float = 0.0002,
     b1: float = 0.5,
     b2: float = 0.999,
     batch_size: int = BATCH_SIZE,
     **kwargs
 ):
     super().__init__()
     self.save_hyperparameters()

     # networks
     self.data_shape = (N, 3)
     self.generator = Generator(latent_dim=self.hparams.latent_dim, data_shape=self.data_shape)
     self.discriminator = Discriminator(data_shape=self.data_shape, hidden_dim=self.hparams.hidden_dim)

     self.validation_z = torch.randn(8, self.hparams.latent_dim)

     self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        
     def forward(self, z):
         return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        meshes, _ = batch

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
        z_loc = torch.zeros(1,self.z_dim,device=device)
        z_scale = torch.ones(1,self.z_dim,device=device)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale))
        a=self.generator.forward(z)
        return a.reshape(12,3,3)
    

'''   
def train(vae,datatraintorch,datatesttorch,epochs=10000):
    pyro.clear_param_store()
    elbotrain=[]
    elbotest=[]
    errortest=[]
    adam_args = {"lr": 0.0001}
    optimizer = Adam(adam_args)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    for epoch in range(epochs):
        if epoch%1000==0:
            print(epoch)
        elbotest.append(svi.evaluate_loss(datatesttorch))
        temp=(1/(24*len(datatesttorch)))*(((vae.apply_vae(datatesttorch)-datatesttorch.reshape(-1,108))**2).sum())
        print(temp)
        errortest.append(temp.clone().detach().cpu())
        elbotrain.append(svi.step(datatraintorch))
    return elbotrain,elbotest,errortest
'''


model = GAN(*my_dataloader.size())
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=5, progress_bar_refresh_rate=20)
trainer.fit(model, my_dataloader)


#vae.load_state_dict(torch.load("cube.pt"))


