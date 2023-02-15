#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:38:18 2022

@author: cyberguli
"""
from time import time
import scipy
import numpy as np
np.random.seed(0)
import pickle 
import skdim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

       
points=pickle.load(open("./data_objects/data.npy", 'rb'))

print(skdim.id.TwoNN().fit(points).dimension_)
print(skdim.id.CorrInt().fit(points).dimension_)
print(skdim.id.MiND_ML().fit(points).dimension_)
print(skdim.id.lPCA().fit(points).dimension_)


