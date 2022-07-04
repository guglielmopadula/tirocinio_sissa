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
import pyro.distributions as dist
import torch.nn.functional as F
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from ordered_set import OrderedSet
import pyro.poutine as poutine
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)
import math




def getinfo(stl):
    your_mesh = mesh.Mesh.from_file(stl)
    myList = list(OrderedSet(tuple(map(tuple,your_mesh.vectors.reshape(36,3)))))
    K=len(your_mesh)
    array=your_mesh.vectors
    topo=np.zeros((12,3))
    for i in range(12):
        for j in range(3):
            topo[i,j]=myList.index(tuple(array[i,j].tolist()))
    N=9*K
    return torch.tensor(array.copy()).reshape(108),torch.tensor(myList),N,len(myList)*3,torch.tensor(topo, dtype=torch.int64)
    
def applytopology(V,M):
    Q=torch.zeros((M.shape[0],3,3),device=device)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Q[i,j]=V[M[i,j].item()]
    return Q




data=torch.zeros(10000,108)
for i in range(0,10000):
    data[i],points,N,K,M=getinfo("parallelepiped_{}.stl".format(i))
    data=data.to(device)

datatrain=data[:data.shape[0]//2]
datatest=data[data.shape[0]//2:]
    





class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, N)
        self.relu = nn.ReLU()

    def forward(self, z):
        result=self.fc4(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))
        return result

class Discriminator(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(N,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu=nn.ReLU()
        self.fc5 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(-1,N)
        hidden=self.fc1(x)
        prob=self.fc5(self.relu(self.fc4(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))))
        return prob

        
class GAN(nn.Module):
    def __init__(self, z_dim=2, hidden_dim=50, use_cuda=False):
        super().__init__()
        self.generator = Generator(z_dim, hidden_dim)
        self.discriminator = Discriminator(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda=use_cuda
        self.z_dim = z_dim
        

    def train(self,datatrain,epochs=5000):
        G_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.9, 0.999))
        D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))
        mean = torch.zeros(datatrain.shape[0],self.z_dim,device=device)
        std = torch.ones(datatrain.shape[0],self.z_dim,device=device)
        fake_labels=torch.ones(datatrain.shape[0])
        true_labels=torch.zeros(datatrain.shape[0])
        all_labels=torch.cat((true_labels,fake_labels),0)
        loss = nn.BCELoss()
        g_train_error=[]
        d_train_error=[]
        for i in range(epochs):
            ###Generator training
            z = pyro.sample("latent", dist.Normal(mean, std))
            generated_mesh=self.generator(z)
            generated_labels=self.discriminator(generated_mesh)
            G_optimizer.zero_grad()
            G_loss=loss(generated_labels.squeeze(),true_labels)
            g_train_error.append(G_loss.clone().detach().tolist())
            G_loss.backward()
            G_optimizer.step()
            ####Discriminator Training
            generated_mesh=self.generator(z)
            all_mesh=torch.concat((datatrain,generated_mesh.detach()),0)
            generated_labels=self.discriminator(all_mesh)
            D_optimizer.zero_grad()
            D_loss=loss(generated_labels.squeeze(),all_labels)
            d_train_error.append(D_loss.clone().detach().tolist())
            D_loss.backward()
            D_optimizer.step()
            if i%1000==0:
                print(i)
        fig, axs = plt.subplots(2)

        axs[0].plot([i for i in range(len(g_train_error))],g_train_error)
        axs[1].plot([i for i in range(len(d_train_error))],d_train_error)

            
        
        
    
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


gan = GAN(use_cuda=use_cuda)

gan.train(datatrain)
temp=gan.sample_mesh()
print(temp)
print("test volume is:",8*temp[0,0,0]*temp[0,0,1]*temp[0,0,2])


#vae.load_state_dict(torch.load("cube.pt"))


