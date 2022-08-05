#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 11:26:49 2022

@author: cyberguli
"""

import numpy as np
from stl import mesh
import stl
import torch
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
torch.manual_seed(0)
import math
import ot

def getinfo(stl):
    your_mesh = mesh.Mesh.from_file(stl)
    myList = list(OrderedSet(tuple(map(tuple,your_mesh.vectors.reshape(np.prod(your_mesh.vectors.shape)//3,3)))))
    array=your_mesh.vectors
    topo=np.zeros((np.prod(your_mesh.vectors.shape)//9,3))
    for i in range(np.prod(your_mesh.vectors.shape)//9):
        for j in range(3):
            topo[i,j]=myList.index(tuple(array[i,j].tolist()))
    return torch.tensor(myList),torch.tensor(topo, dtype=torch.int64)


mesh0,M0=getinfo("../../Data/circle.stl")
mesh1,M1=getinfo("../../Data/double_circle.stl")

