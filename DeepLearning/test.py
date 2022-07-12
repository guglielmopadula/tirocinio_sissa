#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:13:27 2022

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
#device='cpu' 
torch.manual_seed(0)



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

V,M=getinfo("cube.stl")





def calcvolume(V,M):
    A=V[M]
    a=0
    for i in range(len(A)):
        a=a+torch.abs(torch.dot(A[i][0],torch.cross(A[i][1],A[i][2])))/6
    return a


def calcvolume(V,M):
    A=V[M]
    return torch.abs(torch.det(A)).sum()/6