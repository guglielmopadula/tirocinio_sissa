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
import meshio
import skdim
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

       
def getinfo(stl):
    mesh=meshio.read(stl)
    mesh.points[abs(mesh.points)<10e-05]=0
    points_old=mesh.points.astype(np.float32)
    indices=np.arange(len(points_old))[points_old[:,1]>0]
    points=points_old[points_old[:,1]>0]
    points_zero=points_old[points_old[:,1]==0]
    barycenter=np.mean(points,axis=0)
    return points,points_zero,points_old,indices,barycenter


points,points_zero,points_old,indices,barycenter=getinfo("../data_objects/rabbit_translated.ply")

alls_1=np.zeros([600,21306,3])

for i in range(600):
    tmp,_,_,_,_=getinfo("../data_objects/rabbit_{}.ply".format(i))
    alls_1[i]=tmp.copy()

dims_twonn_1=np.zeros(20)
dims_corrint_1=np.zeros(20)
dims_danco_1=np.zeros(20)
dims_fishers_1=np.zeros(20)
dims_cpca_1=np.zeros(20)
dims_mindmle_1=np.zeros(20)


alls_1=alls_1.reshape(600,-1)

for i in range(1,20):
    print(i)
    dims_twonn_1[i]=skdim.id.TwoNN().fit(alls_1[:np.max([(i*30+1),3])]).dimension_
    dims_corrint_1[i]=skdim.id.CorrInt().fit(alls_1[:np.max([(i*30+1),3])]).dimension_
    dims_mindmle_1[i]=skdim.id.MiND_ML().fit(alls_1[:np.max([(i*30+1),3])]).dimension_
    dims_cpca_1[i]=skdim.id.lPCA().fit(alls_1[:np.max([(i*30+1),3])]).dimension_


fig1,ax1=plt.subplots()
ax1.set_title("Dimension of interior")
_=ax1.plot(500*np.arange(20),dims_twonn_1,label='twonn')
_=ax1.plot(500*np.arange(20),dims_corrint_1,label='corrint')
_=ax1.plot(500*np.arange(20),dims_mindmle_1,label='mindmle')
_=ax1.plot(500*np.arange(20),dims_cpca_1,label='cpca')
ax1.legend()
fig1.savefig("dimension_1.png")


pca_1=PCA()
pca_1.fit(alls_1)
cumsum_1=np.cumsum(pca_1.explained_variance_ratio_)
print(np.argmin(np.abs(cumsum_1-(1-1e-5))))
