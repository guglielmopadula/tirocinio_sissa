#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:47:35 2022

@author: cyberguli
"""

import numpy as np
from torch.utils.data import DataLoader
from stl import mesh
import stl
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pyro
import torch.nn.functional as F
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam
from ordered_set import OrderedSet
import pyro.poutine as poutine
import torch.distributions.constraints as constraints
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
torch.manual_seed(0)
import math
from torch_geometric.nn import GAE, VGAE, GCNConv, ASAPooling,SAGEConv,FeaStConv,ChebConv

def getinfo(stl):
    your_mesh = mesh.Mesh.from_file(stl)
    myList = list(OrderedSet(tuple(map(tuple,your_mesh.vectors.reshape(np.prod(your_mesh.vectors.shape)//3,3)))))
    array=your_mesh.vectors
    topo=np.zeros((np.prod(your_mesh.vectors.shape)//9,3),dtype=int)
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

def rescale(x,M):
    temp=x.shape
    x=x.reshape(x.shape[0],-1,3)
    return (x/((torch.abs(torch.det(x[:,M])).sum(1).reshape(x.shape[0],1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape)/6)**1/3)).reshape(temp)

def calcvolume(x,M):
    return x[M].det().abs().sum()/6

def generate_graph(topo):
    list1=[]
    list2=[]
    for i in range(len(topo)):
        for j in range(len(topo[i])):
            for k in range(j+1,len(topo[i])):
                list1.append(topo[i][j].item())
                list2.append(topo[i][k].item())
                list1.append(topo[i][k].item())
                list2.append(topo[i][j].item())
    return torch.tensor([list1,list2],dtype=torch.long)


data=[]
for i in range(100):
    if i%100==0:
        print(i)
    points,M=getinfo("bulbo_{}.stl".format(i))
    if device!='cpu':
        points=points.to(device)
    data.append(points)
    
    
meshsize=torch.numel(points)
if device!='cpu':
    M=M.to(device)
    
datatrain=data[:len(data)//2]
datatest=data[len(data)//2:]
datatraintorch=torch.zeros(len(datatrain),datatrain[0].shape[0],datatrain[0].shape[1],dtype=datatrain[0].dtype, device=device)
datatesttorch=torch.zeros(len(datatest),datatest[0].shape[0],datatest[0].shape[1],dtype=datatest[0].dtype, device=device)
for i in range(len(datatrain)):
    datatraintorch[i:]=datatrain[i]
for i in range(len(datatest)):
    datatesttorch[i:]=datatest[i]

edge_index=generate_graph(M)
indices=torch.argsort(edge_index[1])
edge_index=edge_index[:,indices]


class VolumeNormalizer(nn.Module):
    def __init__(self,M):
        super().__init__()
        self.M=M
        '''
    def forward(self, x):
        temp=x.shape
        x=x.reshape(x.shape[0],-1,3)
        x=x/((x[:,self.M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
        return x.reshape(temp)
    '''   
    def forward(self,x):
        temp=x.shape
        x=x.reshape(1,-1,3)
        x=x/((x[:,self.M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(1,-1,3)
        return x.reshape(temp)

def create_permutation(perm,max_el):
    a=torch.arange(max_el+1)
    t=0
    for i in perm:
        a[i]=t
        a[t]=i
        t=t+1
    return a


class Unpooling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.conv1=ChebConv(in_channels,out_channels,6)
        self.conv2=ChebConv(in_channels,out_channels,6)
        self.linear=nn.Linear(in_channels,in_channels)

    
    def forward(self,x,perm,edge_index):
        x=torch.cat([x,torch.zeros(torch.max(edge_index)-len(x)+1,self.in_channels)],0)
        perm_vec=create_permutation(perm,torch.max(edge_index).item())
        x=x[perm_vec,:]
        x=self.linear(self.conv2(self.conv1(x,edge_index),edge_index))
        return x

        

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=ChebConv(3, 3,6)
        self.relu=nn.ReLU()
        self.fc2=ChebConv(3, 1,6)
        self.fc3=ASAPooling(1,30)


    def forward(self, x,edge_index):
        x=x.reshape(meshsize//3,3)
        temp=self.fc1(x,edge_index)
        temp=self.relu(temp)
        temp=self.fc2(temp,edge_index)
        z,new_edge_index,weights,_,perm=self.fc3(temp,edge_index)
        z=torch.tanh(z)
        return z,new_edge_index,perm

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=ChebConv(1, 2,6)
        self.relu=nn.ReLU()
        self.unpool=Unpooling(1,1)
        self.fc2=ChebConv(2,3,6)
        self.fc3=ChebConv(3,3,6)
        self.vol=VolumeNormalizer(M)

    def forward(self,z,perm):
        temp=self.unpool(z,perm,edge_index)
        temp=self.fc1(temp,edge_index)
        temp=self.relu(temp)
        temp=self.fc2(temp,edge_index)
        x=self.fc3(temp,edge_index)
        #x=self.vol(x)
        
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=Decoder()
        self.encoder=Encoder()
    
    def forward(self,x,edge_index):
        z,new_edge_index,perm=self.encoder(x,edge_index)
        x_hat=self.decoder(z,perm)
        return x_hat,z,perm
    
vae=VAE()

def dependent_sample(prob,n):
    lst=[]
    for i in range(n):
        dist=torch.distributions.categorical.Categorical(prob)
        temp=dist.sample()
        lst.append(temp)
        prob = prob[np.r_[:temp, (temp+1):len(prob)]]
    return lst
        


    
def train(vae,epochs):
    opt = torch.optim.Adam(vae.parameters(),lr=0.01)
    errortrain=[]
    errortest=[]
    prob=torch.zeros(len(datatraintorch[0]))
    for _ in range(epochs):
        losstrain=0
        losstest=0
        opt.zero_grad()
        zlist=0
        for i in range(len(datatraintorch)):
            x_hat,z,perm=vae(datatraintorch[i],edge_index)
            prob[perm]+=1
            temp=torch.zeros(len(datatraintorch[0]))
            temp[perm]=z.reshape(-1)
            losstrain=losstrain+torch.norm(datatraintorch[i]-x_hat)/torch.norm(datatraintorch[i])
            losstrain=losstrain/len(datatraintorch)
            if i==0:
                zlist=temp.reshape(1,-1)
            else:
                zlist=torch.cat((zlist,temp.reshape(1,-1)))
        losstrain=losstrain
        losstrain.backward()
        opt.step()
        for i in range(len(datatesttorch)):
            x_hat,z,perm=vae(datatesttorch[i],edge_index)
            losstest=losstest+torch.norm(datatesttorch[i]-x_hat)/torch.norm(datatesttorch[i])
            losstest=losstest/len(datatesttorch)
        errortrain.append(losstrain.clone().detach().item())
        errortest.append(losstest.clone().detach().item())
    prob=prob/torch.sum(prob)
    return errortrain,errortest,prob,torch.mean(zlist,dim=0),torch.var(zlist,dim=0)



errortrain,errortest,prob,mean,var=train(vae,100)
hidden_dim=30
hidden_channels=1

def sample_mesh(prob,mean,var,vae):
    indices=dependent_sample(prob, hidden_dim)
    lst=[]
    for i in indices:
        temp=torch.distributions.normal.Normal(0,1).sample()*torch.sqrt(var[i])+mean[i]
        lst.append(temp.item())
    temp=torch.tensor(lst).reshape(-1,hidden_channels)
    x=vae.decoder(temp,indices)        
    mesh=applytopology(x.reshape(meshsize//3,3),M)
    return mesh

    
    return x

plt.plot([i for i in range(len(errortrain))],errortrain,'g')
plt.plot([i for i in range(len(errortest))],errortest,'r')


temp=sample_mesh(prob,mean,var,vae)
newmesh = np.zeros(len(temp), dtype=mesh.Mesh.dtype)
newmesh['vectors'] = temp.cpu().detach().numpy().copy()
mymesh = mesh.Mesh(newmesh.copy())
mymesh.save('test.stl', mode=stl.Mode.ASCII)


    
    

'''        

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, meshsize)
        self.fc5=VolumeNormalizer(M)
        self.relu = nn.ReLU()

    def forward(self, z):
        result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))       
        #result=self.fc5(result)
        return result
    
    def sample(self,z):
        result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))
        result=self.fc5.forward_single(result)
        return result

    
        
class VAE(nn.Module):
    def __init__(self, z_dim=1, hidden_dim=300, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda=use_cuda
        self.z_dim = z_dim
        
    @poutine.scale(scale=1.0/datatraintorch.shape[0])
    def model(self,x):
        pyro.module("decoder", self.decoder)
        mu = pyro.param("mu",torch.zeros( self.z_dim, dtype=x.dtype, device=x.device))
        sigma = pyro.param("sigma",torch.ones(self.z_dim, dtype=x.dtype, device=x.device),constraint=constraints.positive)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z_scale=torch.cat(x.shape[0]*[mu], 0).reshape(-1,self.z_dim)
            z_loc=torch.cat(x.shape[0]*[sigma], 0).reshape(-1,self.z_dim)
            z = pyro.sample("latent", dist.Normal(z_scale, z_loc).to_event(1))
            # decode the latent code z
            x_hat = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Normal(x_hat, (1e-7)*torch.ones(x_hat.shape, dtype=x.dtype, device=x.device), validate_args=False).to_event(1),
                obs=x.reshape(-1, meshsize),
            )
            # return the loc so we can visualize it later
            return x_hat

    @poutine.scale(scale=1.0/datatraintorch.shape[0])   
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
    
    def apply_vae_point(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    def apply_vae_mesh(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        loc_img=applytopology(loc_img.reshape(meshsize//3,3),M)
        return loc_img


    def sample_mesh(self):
        mu = pyro.param("mu")
        sigma = pyro.param("sigma") 
        z = pyro.sample("latent", dist.Normal(mu, sigma))
        a=self.decoder.sample(z)
        mesh=applytopology(a.reshape(meshsize//3,3),M)
        return mesh
    

    
def train(vae,datatraintorch,datatesttorch,epochs=5000):
    pyro.clear_param_store()
    elbotrain=[]
    elbotest=[]
    errortest=[]
    adam_args = {"lr": 0.001}
    optimizer = Adam(adam_args)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    for epoch in range(epochs):
        print(epoch)
        if epoch%1000==0:
            print(epoch)
        elbotest.append(svi.evaluate_loss(datatesttorch))
        temp=(1/(meshsize*len(datatesttorch)))*(((vae.apply_vae_point(datatesttorch)-datatesttorch.reshape(len(datatesttorch),meshsize))**2).sum())
        print(temp)
        errortest.append(temp.clone().detach().cpu())
        elbotrain.append(svi.step(datatraintorch))
    return elbotrain,elbotest,errortest



vae = VAE(use_cuda=use_cuda)



#vae.load_state_dict(torch.load("cube.pt"))
elbotrain,elbotest,errortest = train(vae,datatraintorch, datatesttorch)


fig, axs = plt.subplots(2)

axs[0].plot([i for i in range(len(elbotrain))],elbotrain)
axs[0].plot([i for i in range(len(elbotest))],elbotest)
axs[1].plot([i for i in range(len(errortest))],errortest)


lst=vae.encoder(datatesttorch)[0].clone().detach().cpu().numpy()
    
lst=np.array(lst)

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

def rescale(x):
    temp=x.shape
    x=x.reshape(x.shape[0],-1,3)
    return (x/((torch.abs(torch.det(x[:,M])).sum(1).reshape(x.shape[0],1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape)/6)**1/3)).reshape(temp)

def calcvolume(x):
    return x[M].det().abs().sum()/6

data=[]
for i in range(100):
    if i%100==0:
        print(i)
    points,M=getinfo("bulbo_{}.stl".format(i))
    if device!='cpu':
        points=points.to(device)
    data.append(points)
    
    
meshsize=torch.numel(points)
if device!='cpu':
    M=M.to(device)
    
datatrain=data[:len(data)//2]
datatest=data[len(data)//2:]
datatraintorch=torch.zeros(len(datatrain),datatrain[0].shape[0],datatrain[0].shape[1],dtype=datatrain[0].dtype, device=device)
datatesttorch=torch.zeros(len(datatest),datatest[0].shape[0],datatest[0].shape[1],dtype=datatest[0].dtype, device=device)
for i in range(len(datatrain)):
    datatraintorch[i:]=datatrain[i]
for i in range(len(datatest)):
    datatesttorch[i:]=datatest[i]

class VolumeNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        temp=x.shape
        x=x.reshape(x.shape[0],-1,3)
        x=x/((x[:,M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(x.shape[0],-1,3)
        return x.reshape(temp)
    
    def forward_single(self,x):
        temp=x.shape
        x=x.reshape(1,-1,3)
        x=x/((x[:,M].det().abs().sum(1)/6)**(1/3)).reshape(-1,1).expand(x.shape[0],x.numel()//x.shape[0]).reshape(1,-1,3)
        return x.reshape(temp)

        

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, meshsize)
        self.fc5=VolumeNormalizer()
        self.relu = nn.ReLU()

    def forward(self, z):
        result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))       
        #result=self.fc5(result)
        return result
    
    def sample(self,z):
        result=self.fc4(self.relu(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(z))))))))
        result=self.fc5.forward_single(result)
        return result

    

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(meshsize,hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc41=nn.Tanh()
        self.fc32 = nn.Sigmoid()
        self.batch=nn.BatchNorm1d(1)



    def forward(self, x):
        x=x.reshape(-1,meshsize)
        hidden=self.fc1(x)
        mu=self.fc31(self.fc21(hidden))
        mu=self.batch(mu)
        mu=mu/torch.linalg.norm(mu)*torch.tanh(torch.linalg.norm(mu))*(2/math.pi)
        sigma=torch.exp(self.fc32(self.fc22(hidden)))-0.9
        return mu,sigma

        
class VAE(nn.Module):
    def __init__(self, z_dim=1, hidden_dim=300, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda=use_cuda
        self.z_dim = z_dim
        
    @poutine.scale(scale=1.0/datatraintorch.shape[0])
    def model(self,x):
        pyro.module("decoder", self.decoder)
        mu = pyro.param("mu",torch.zeros( self.z_dim, dtype=x.dtype, device=x.device))
        sigma = pyro.param("sigma",torch.ones(self.z_dim, dtype=x.dtype, device=x.device),constraint=constraints.positive)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z_scale=torch.cat(x.shape[0]*[mu], 0).reshape(-1,self.z_dim)
            z_loc=torch.cat(x.shape[0]*[sigma], 0).reshape(-1,self.z_dim)
            z = pyro.sample("latent", dist.Normal(z_scale, z_loc).to_event(1))
            # decode the latent code z
            x_hat = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Normal(x_hat, (1e-7)*torch.ones(x_hat.shape, dtype=x.dtype, device=x.device), validate_args=False).to_event(1),
                obs=x.reshape(-1, meshsize),
            )
            # return the loc so we can visualize it later
            return x_hat

    @poutine.scale(scale=1.0/datatraintorch.shape[0])   
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
    
    def apply_vae_point(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    def apply_vae_mesh(self,x):
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        loc_img=applytopology(loc_img.reshape(meshsize//3,3),M)
        return loc_img


    def sample_mesh(self):
        mu = pyro.param("mu")
        sigma = pyro.param("sigma") 
        z = pyro.sample("latent", dist.Normal(mu, sigma))
        a=self.decoder.sample(z)
        mesh=applytopology(a.reshape(meshsize//3,3),M)
        return mesh
    

    
def train(vae,datatraintorch,datatesttorch,epochs=5000):
    pyro.clear_param_store()
    elbotrain=[]
    elbotest=[]
    errortest=[]
    adam_args = {"lr": 0.001}
    optimizer = Adam(adam_args)
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    for epoch in range(epochs):
        print(epoch)
        if epoch%1000==0:
            print(epoch)
        elbotest.append(svi.evaluate_loss(datatesttorch))
        temp=(1/(meshsize*len(datatesttorch)))*(((vae.apply_vae_point(datatesttorch)-datatesttorch.reshape(len(datatesttorch),meshsize))**2).sum())
        print(temp)
        errortest.append(temp.clone().detach().cpu())
        elbotrain.append(svi.step(datatraintorch))
    return elbotrain,elbotest,errortest



vae = VAE(use_cuda=use_cuda)



#vae.load_state_dict(torch.load("cube.pt"))
elbotrain,elbotest,errortest = train(vae,datatraintorch, datatesttorch)


fig, axs = plt.subplots(2)

axs[0].plot([i for i in range(len(elbotrain))],elbotrain)
axs[0].plot([i for i in range(len(elbotest))],elbotest)
axs[1].plot([i for i in range(len(errortest))],errortest)


lst=vae.encoder(datatesttorch)[0].clone().detach().cpu().numpy()
    
lst=np.array(lst)
'''