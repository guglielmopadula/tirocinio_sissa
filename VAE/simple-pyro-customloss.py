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
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO,TraceMeanField_ELBO
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
    return torch.tensor(array.copy()),torch.tensor(myList),N,len(myList)*3,torch.tensor(topo, dtype=torch.int64)
    
def applytopology(V,M):
    Q=torch.zeros((M.shape[0],3,3),device=device)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Q[i,j]=V[M[i,j].item()]
    return Q



'''
data=[]
M=0
N=0
K=0
for i in range(1,1000):
    meshs,points,N,K,M=getinfo("parallelepiped_{}.stl".format(i))
    if device!='cpu':
        meshs=meshs.to(device)
    data.append(meshs)

if device!='cpu':
    M=M.to(device)
    
datatrain=data[1:len(data)//3]
datatest=data[len(data)//3:]
datatraintorch=torch.zeros(len(datatrain),datatrain[0].shape[0],datatrain[0].shape[1],datatrain[0].shape[2],dtype=datatrain[0].dtype, device=device)
datatesttorch=torch.zeros(len(datatest),datatest[0].shape[0],datatest[0].shape[1],datatest[0].shape[2],dtype=datatest[0].dtype, device=device)
for i in range(len(datatrain)):
    datatraintorch[i:]=datatrain[i]
for i in range(len(datatest)):
    datatesttorch[i:]=datatest[i]
'''



data=[]
M=0
N=0
K=0
for i in range(1,10000):
    meshs,points,N,K,M=getinfo("parallelepiped_{}.stl".format(i))
    if device!='cpu':
        meshs=meshs.to(device)
    data.append(points)

if device!='cpu':
    M=M.to(device)
    
datatrain=data[0:len(data)//2]
datatest=data[len(data)//2:]
datatraintorch=torch.zeros(len(datatrain),datatrain[0].shape[0],datatrain[0].shape[1],dtype=datatrain[0].dtype, device=device)
datatesttorch=torch.zeros(len(datatest),datatest[0].shape[0],datatest[0].shape[1],dtype=datatest[0].dtype, device=device)
for i in range(len(datatrain)):
    datatraintorch[i:]=datatrain[i]
for i in range(len(datatest)):
    datatesttorch[i:]=datatest[i]
N=24



class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, N)
        self.tanh = nn.Tanh()

    def forward(self, z):
        result=self.fc4(self.fc3(self.fc2(self.fc1(z))))
        return result

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(N,hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc41=nn.Tanh()
        self.fc32 = nn.Sigmoid()



    def forward(self, x):
        x=x.reshape(-1,N)
        hidden=self.fc1(x)
        mu=self.fc41(self.fc31(self.fc21(hidden)))
        sigma=1/(2*math.exp(1))*torch.exp(self.fc32(self.fc22(hidden)))
        return mu,sigma

        
class VAE(nn.Module):
    def __init__(self, z_dim=2, hidden_dim=30, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda=use_cuda
        self.z_dim = z_dim
        
    def model(self,x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            x_hat = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
        
            # return the loc so we can visualize it later
            return x_hat

    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    
    def apply_vae_verbose(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        print("scale is",z_scale,"mean is", z_loc)
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
    
    def apply_vae(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


    def sample_mesh(self):
        z_loc = torch.zeros(self.z_dim,device=device)
        z_scale = torch.ones(self.z_dim,device=device)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale))
        a=self.decoder.forward(z_loc)
        return a.reshape(8,3)
    
def mseloss(vae,x):
    loss=1/(x.shape[0]*24)*(((x.reshape(-1,24)-vae.apply_vae(x))**2).sum())
    return loss
def train(vae,datatraintorch,datatesttorch,epochs=10000):
    pyro.clear_param_store()
    elbotrain=[]
    elbotest=[]
    errortest=[]
    optimizer = torch.optim.Adam(vae.parameters(),lr=0.0001)
    kl = TraceMeanField_ELBO().differentiable_loss
    for epoch in range(epochs):
        globaltrain=0
        globaltest=0

        if epoch%1000==0:
            print(epoch)
        for i in range(10):
            losstrainstep = kl(vae.model, vae.guide, datatraintorch[i*100:(i+1)*100])/datatraintorch[i*100:(i+1)*100].shape[0]+mseloss(vae,datatraintorch[i*100:(i+1)*100])
            losstrainstep.backward()
            lossteststep = kl(vae.model, vae.guide, datatesttorch[i*100:(i+1)*100])/datatesttorch[i*100:(i+1)*100].shape[0]+mseloss(vae,datatesttorch[i*100:(i+1)*100])
            optimizer.step()
            optimizer.zero_grad()
            globaltrain=globaltrain+losstrainstep
            globaltest=globaltrain+lossteststep
            
        elbotest.append(globaltest/10)
        temp=(1/(24*100))*(((vae.apply_vae(datatesttorch[0:100])-datatesttorch[0:100].reshape(-1,24))**2).sum())
        errortest.append(temp.clone().detach().cpu())
        elbotrain.append(globaltrain/10)
    return elbotrain,elbotest,errortest



vae = VAE(use_cuda=use_cuda)



#vae.load_state_dict(torch.load("cube.pt"))
elbotrain,elbotest,errortest = train(vae,datatraintorch, datatesttorch)


fig, axs = plt.subplots(2)

axs[0].plot([i for i in range(len(elbotrain))],elbotrain)
axs[0].plot([i for i in range(len(elbotest))],elbotest)
axs[1].plot([i for i in range(len(errortest))],errortest)



temp=vae.sample_mesh()
a=applytopology(temp,M).cpu().detach().numpy()
cube = mesh.Mesh(np.zeros(12, dtype=mesh.Mesh.dtype))
cube.vectors=a
cube.save('test.stl')
cube= mesh.Mesh.from_file('test.stl')


print(datatesttorch[30])
print(vae.apply_vae_verbose(datatesttorch[30]).reshape(8,3))
print(datatesttorch[31])
print(vae.apply_vae_verbose(datatesttorch[31]).reshape(8,3))

