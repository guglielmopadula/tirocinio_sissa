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

meshs=np.zeros((500,2127))
for i in range(500):
    M=meshio.read("bulbo_{}.stl".format(i))
    meshs[i]=M.points.reshape(-1)

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
