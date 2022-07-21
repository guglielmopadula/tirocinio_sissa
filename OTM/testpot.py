#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:15:04 2022

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
import pyro.distributions as dist
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
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ot

########################################à 
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


def generate_square(length,n_points):
    square = np.array([(np.random.uniform(-0.5*length, 0.5*length), np.random.uniform(-0.5*length, 0.5*length)) for _ in range(n_points)])
    plt.scatter(square[0],square[1])
    return square

def rotate(p, angle=0):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    result=(R@p.T).T
    return result

def generate_ellipse(radius,n_points,a):
    x=[]
    y=[]
    for i in range(n_points):
        r = radius*np.sqrt(np.random.uniform())
        theta = np.random.uniform() * 2 *math.pi
        x.append(r*a*math.cos(theta))
        y.append(r*(1/a)*math.sin(theta))
    
    return np.array([x,y]).T



def generate_circle(radius,n_points):
    return generate_ellipse(radius, n_points, 1)


mesh0,M0=getinfo("bulboa.stl")
mesh1,M1=getinfo("bulbob.stl")
mesh2,M2=getinfo("bulboc.stl")

mesh0=mesh0.numpy()
mesh1=mesh1.numpy()
mesh2=mesh2.numpy()


def calculate_wesserstain(X,w):
    X_init=np.mean(X,axis=0)
    a=[np.ones(len(X[0]))*1/len(X[0])]*3
    Y=ot.lp.free_support_barycenter(X,a,X_init,None,w)
    return Y

X=[mesh0,mesh1,mesh2]
w=np.ones(3)/3
Y=calculate_wesserstain(X, w)

temp=applytopology(torch.tensor(Y),M0)
newmesh = np.zeros(len(temp), dtype=mesh.Mesh.dtype)
newmesh['vectors'] = temp.cpu().detach().numpy().copy()
mymesh = mesh.Mesh(newmesh.copy())
mymesh.save('test_wasser.stl', mode=stl.Mode.ASCII)
##############################################################

square=generate_square(1,1000)
circle=generate_circle(math.sqrt(1/math.pi),1000)
ellipse=generate_ellipse(math.sqrt(1/math.pi),1000,2)
rotation=rotate(square,angle=math.pi/4)

allotm=np.zeros([5,5,1000,2])
allotm[0,0]=square
allotm[0,4]=rotation
allotm[4,4]=ellipse
allotm[4,0]=circle

for i in range(1,4):
    print("doing",0,i)
    allotm[0,i]=calculate_wesserstain([allotm[0,0],allotm[0,4]], [(4-i)/4,i/4])

for i in range(1,4):
    print("doing",4,i)
    allotm[4,i]=calculate_wesserstain([allotm[4,0],allotm[4,4]], [(4-i)/4,i/4])

for j in range(1,4):
    print("doing",j,0)
    allotm[j,0]=calculate_wesserstain([allotm[0,0],allotm[4,0]], [(4-j)/4,j/4])

for j in range(1,4):
    print("doing",j,4)
    allotm[j,4]=calculate_wesserstain([allotm[0,4],allotm[4,4]], [(4-j)/4,j/4])

for j in range(1,4):
    for i in range(1,4):
        print("doing",j,i)
        allotm[j,i]=calculate_wesserstain([allotm[j,0],allotm[j,4]], [(4-i)/4,i/4])



fig, ax = plt.subplots(5,5)
for i in range(5):
    for j in range(5):
        ax[i,j].set_xlim(-1,1)
        ax[i,j].set_ylim(-1,1)
        ax[i,j].axis('off')
        ax[i,j].plot(allotm[i,j,:,0],allotm[i,j,:,1])
'''
ax[0,0].plot(square[:,0],square[:,1])
ax[0,4].plot(rotation[:,0],rotation[:,1])
ax[4,4].plot(ellipse[:,0],ellipse[:,1])
ax[4,0].plot(circle[:,0],circle[:,1])
'''
plt.show()