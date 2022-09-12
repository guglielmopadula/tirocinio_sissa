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
import pickle
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
import networkx as nx
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


def getinfo(stl):
    your_mesh = mesh.Mesh.from_file(stl)
    myList = list(OrderedSet(tuple(map(tuple,your_mesh.vectors.reshape(np.prod(your_mesh.vectors.shape)//3,3)))))
    array=your_mesh.vectors
    topo=np.zeros((np.prod(your_mesh.vectors.shape)//9,3))
    for i in range(np.prod(your_mesh.vectors.shape)//9):
        for j in range(3):
            topo[i,j]=myList.index(tuple(array[i,j].tolist()))
    return torch.tensor(myList),torch.tensor(topo, dtype=torch.int64)

def create_adjacency(topo):
    matrix=torch.zeros(605,605)
    for i in range(len(topo)):
        matrix[topo[i][0],topo[i][1]]=1
        matrix[topo[i][1],topo[i][0]]=1
        matrix[topo[i][1],topo[i][2]]=1
        matrix[topo[i][2],topo[i][1]]=1
        matrix[topo[i][2],topo[i][0]]=1
        matrix[topo[i][0],topo[i][2]]=1
    return matrix

def show_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=1)
    plt.show()


_,topo=getinfo('test.stl')
matrix=create_adjacency(topo.tolist())

def topo2newadj(topo):
    matrix=torch.zeros(len(topo),len(topo),dtype=torch.int)
    for i in range(len(topo)):
        for j in range(i+1,len(topo)):
            if len(set(topo[i]).intersection(set(topo[j])))>=2:
                matrix[i,j]=1
    return matrix
    
matrix2=topo2newadj(topo)
def adj2topo(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    return [sorted(c) for c in nx.minimum_cycle_basis(G)]

'''
topo2=adj2topo(matrix2)
with open('1reducedtopology','wb') as file:
    pickle.dump(topo2,file)
'''
with open('1reducedtopology','rb') as file:
    topo2=pickle.load(file)

matrix3=topo2newadj(topo2)
topo3=adj2topo(matrix3)
with open('2reducedtopology','wb') as file:
    pickle.dump(topo3,file)

