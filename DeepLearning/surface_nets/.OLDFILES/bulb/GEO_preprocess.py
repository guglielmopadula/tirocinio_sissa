#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:30:42 2022

@author: cyberguli
"""

import numpy as np
import meshio
import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, dense_mincut_pool
from torch_geometric import utils
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch_geometric.data import Data
import os.path as osp
from torch_geometric.datasets import Planetoid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
torch.manual_seed(0)

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


pointslist=[]
Mlist=[]
for i in range(9):
    points,M=getinfo("../../Data/bulbo_red{}.stl".format(i))
    pointslist.append(points)
    Mlist.append(M)
    

graph=[]
for i in range(9):
    graph.append(generate_graph(Mlist[i]))
    
Slist=[]
for i in range(8):
    S=torch.zeros(torch.max(graph[i+1])+1,torch.max(graph[i]+1))
    for k in range(len(pointslist[i+1])):
        dist=float('Inf')
        ass=k
        for h in range(len(pointslist[i])):
            if torch.linalg.norm(pointslist[i+1][k]-pointslist[i][h])<1e-4:
               S[k,h]=1
               break
    Slist.append(S)
           
            
torch.save(graph,"graphs.pt")
torch.save(Slist,"Ss.pt")