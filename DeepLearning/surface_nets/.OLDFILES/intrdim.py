#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:10:39 2022

@author: cyberguli
"""

import meshio
import numpy as np
import skdim
import pandas


M=meshio.read("hull_{}.stl".format(0))
temp=M.points.copy()
temp=temp[temp[:,2]>0]
temp=temp[temp[:,1]>0]


meshs=np.zeros((2000,len(temp.reshape(-1))))
for i in range(2000):
    print(i)
    M=meshio.read("hull_{}.stl".format(i))
    temp=M.points.copy()
    temp=temp[temp[:,2]>0]
    temp=temp[temp[:,1]>0]
    meshs[i]=temp.reshape(-1)

lpca = skdim.id.lPCA().fit(meshs)
print("PCA dim is ",lpca.dimension_)
lfs=skdim.id.FisherS().fit(meshs)
print("LFS dim is ",lfs.dimension_)
lmads=skdim.id.MADA().fit(meshs) #manifold hypotesiss
print("MADA dim is ",lmads.dimension_)
lcorr=skdim.id.CorrInt().fit(meshs)
print("Correlation dim os",lcorr.dimension_)
ltwonn=skdim.id.TwoNN().fit(meshs)
print("ltwonn dim is",ltwonn.dimension_)
lmind=skdim.id.MiND_ML().fit(meshs) 
print("Mind dim is ",lmind.dimension_)
less= skdim.id.ESS().fit(meshs)
print("ESS dim is ",less.dimension_)
lmle= skdim.id.MLE().fit(meshs)
print("MLE dim is ",lmle.dimension_)
ldanco=lfs=skdim.id.DANCo().fit(meshs)
print("DANCO dim is ",ldanco.dimension_)
