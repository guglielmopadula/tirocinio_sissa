#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:47:35 2022

@author: cyberguli
"""

import numpy as np
from stl import mesh
import stl
import meshio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
torch.manual_seed(0)
from torch_geometric.nn import ChebConv
import math
import torch.nn.functional as F


def getinfo(stl):
    mesh=meshio.read(stl)
    points=torch.tensor(mesh.points.astype(np.float32))
    triangles=torch.tensor(mesh.cells_dict['triangle'].astype(np.int64))
    return points,triangles
    
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

graphs=torch.load("graphs.pt")
Ss=torch.load("Ss.pt")



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


        

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=ChebConv(3,3,6)
        self.fc2=ChebConv(3,2,6)
        self.fc3=ChebConv(2,4,6)
        self.fc4=ChebConv(4,8,6)
        self.fc5=ChebConv(8,16,6)
        self.fc6=ChebConv(16,1,6)
        self.elu=nn.ELU()
    def forward(self,x):
        x=x.reshape(meshsize//3,3)
        temp=self.fc1(x,graphs[0])
        temp=torch.matmul(Ss[0],temp)
        temp=self.elu(temp)
        temp=self.fc2(temp,graphs[1])
        temp=torch.matmul(Ss[1],temp)
        temp=self.elu(temp)
        temp=self.fc3(temp,graphs[2])
        temp=torch.matmul(Ss[2],temp)
        temp=self.elu(temp)
        temp=self.fc4(temp,graphs[3])
        temp=torch.matmul(Ss[3],temp)
        temp=self.elu(temp)
        temp=self.fc5(temp,graphs[4])
        temp=torch.matmul(Ss[4],temp)
        temp=self.elu(temp)
        temp=self.fc6(temp,graphs[5])
        temp=temp/torch.linalg.norm(temp)*torch.tanh(torch.linalg.norm(temp))*(2/math.pi)

        return temp

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=ChebConv(1,16,6)
        self.fc2=ChebConv(16,8,6)
        self.fc3=ChebConv(8,4,6)
        self.fc4=ChebConv(4,2,6)
        self.fc5=ChebConv(2,3,6)
        self.fc6=ChebConv(3,3,6)
        self.elu=nn.ELU()

        self.vol=VolumeNormalizer(M)

    def forward(self,z):
        temp=self.fc1(z,graphs[5])
        temp=torch.matmul(Ss[4].T,temp)
        temp=self.elu(temp)
        temp=self.fc2(temp,graphs[4])
        temp=torch.matmul(Ss[3].T,temp)
        temp=self.elu(temp)
        temp=self.fc3(temp,graphs[3])
        temp=torch.matmul(Ss[2].T,temp)
        temp=self.elu(temp)
        temp=self.fc4(temp,graphs[2])
        temp=torch.matmul(Ss[1].T,temp)
        temp=self.elu(temp)
        temp=self.fc5(temp,graphs[1])
        temp=torch.matmul(Ss[0].T,temp)
        temp=self.elu(temp)
        temp=self.fc6(temp,graphs[0])
        x=self.vol(temp)
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=Decoder()
        self.encoder=Encoder()
    
    def forward(self,x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat,z
    
vae=VAE()



    
def train(vae,epochs):
    opt = torch.optim.Adam(vae.parameters(),lr=0.01)
    errortrain=[]
    errortest=[]
    prob=torch.zeros(len(datatraintorch[0]))
    for _ in range(epochs):
        losstrain=0
        losstest=0
        opt.zero_grad()
        zlist=torch.tensor([])
        for i in range(len(datatraintorch)):
            x_hat,z=vae(datatraintorch[i])
            losstrain=losstrain+F.mse_loss(x_hat, datatraintorch[i])
            zlist=torch.concat((zlist,z.reshape(1,-1)))
        losstrain=losstrain/len(datatraintorch)
        print(losstrain)
        losstrain.backward()
        opt.step()
        for i in range(len(datatesttorch)):
            x_hat,z=vae(datatesttorch[i])
            losstest=losstest+torch.norm(datatesttorch[i]-x_hat)/torch.norm(datatesttorch[i])
            losstest=losstest/len(datatesttorch)
        errortrain.append(losstrain.clone().detach().item())
        errortest.append(losstest.clone().detach().item())
    return errortrain,errortest,zlist



errortrain,errortest,zlist=train(vae,100)
hidden_dim=30
hidden_channels=1
mean=torch.mean(zlist,dim=0)
var=torch.var(zlist,dim=0)

def sample_mesh(mean,var,vae):
    temp=torch.distributions.normal.Normal(0,1,len(var)).sample()*torch.sqrt(var)+mean
    temp=temp.reshape(-1,1)
    x=vae.decoder(temp)        
    mesh=applytopology(x.reshape(meshsize//3,3),M)
    return mesh

    

plt.plot([i for i in range(len(errortrain))],errortrain,'g')
plt.plot([i for i in range(len(errortest))],errortest,'r')


temp=sample_mesh(mean,var,vae)
temp=vae.decoder(vae.encoder(datatraintorch[0]))
temp=applytopology(temp.reshape(meshsize//3,3),M)
newmesh = np.zeros(len(temp), dtype=mesh.Mesh.dtype)
newmesh['vectors'] = temp.cpu().detach().numpy().copy()
mymesh = mesh.Mesh(newmesh.copy())
mymesh.save('test.stl', mode=stl.Mode.ASCII)

    
    
