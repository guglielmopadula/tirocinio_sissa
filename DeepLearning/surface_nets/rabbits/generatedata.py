#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:38:18 2022

@author: cyberguli
"""

from numpy.random import Generator, PCG64
import pickle
from os import path
import os
import scipy
import numpy as np
import meshio
from sklearn.decomposition import PCA
import time
from tqdm import trange
from cpffd import BPFFD

        



def getinfo(stl):
    mesh=meshio.read(stl)
    mesh.points[abs(mesh.points)<10e-05]=0
    points=mesh.points
    barycenter=np.mean(points,axis=0)
    return points,barycenter


points,barycenter=getinfo("./data_objects/rabbit.ply")


NUMBER_SAMPLES=600
alls=np.zeros([NUMBER_SAMPLES,*points.shape])


nx=3
ny=3
nz=3
latent=np.zeros([NUMBER_SAMPLES,nx,ny,nz,3])

ffd=BPFFD(box_origin=[np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])],box_length=[np.max(points[:,0])-np.min(points[:,0]), np.max(points[:,1])-np.min(points[:,1]), np.max(points[:,2])-np.min(points[:,2])],n_control_points=np.array([nx, ny, nz]))
points_local=ffd.mesh_to_local_space(points)

M=np.eye(nx*ny*nz*3)
print(points.shape)
epsilon=0.005
for i in range(nx*ny*nz*3):
    tmp=i%(nx*ny*nz)
    M[i,i]=1/(ffd.f(tmp)[2]+epsilon)

print(np.mean(points,axis=0))
min=np.min([M[i,i] for i in range(nx*ny*nz*3)])
M=M*(1/min)
print("compiling")
a=0.2
ffd.array_mu_x=a*np.random.uniform(size=(nx,ny,nz))
ffd.array_mu_x[:,0,:]=0
ffd.array_mu_y[:,0,:]=0
ffd.array_mu_z[:,0,:]=0
points2_local=ffd.barycenter_ffd(points_local,M)


print("creating meshes")
for i in trange(600):
    ffd.array_mu_x=a*np.random.uniform(size=(nx,ny,nz))*(np.arange(nz).reshape(1,1,-1))
    ffd.array_mu_y=a*np.random.uniform(size=(nx,ny,nz))*(np.arange(nz).reshape(1,1,-1))
    ffd.array_mu_z=a*np.random.uniform(size=(nx,ny,nz))*(np.arange(nz).reshape(1,1,-1))
    latent[i,:,:,:,0]=ffd.array_mu_x
    latent[i,:,:,:,1]=ffd.array_mu_y
    latent[i,:,:,:,2]=ffd.array_mu_z
    points2_local=ffd.barycenter_ffd(points_local,M)
    points2=ffd.mesh_to_global_space(points2_local)
    alls[i]=points2
    meshio.write_points_cells("./data_objects/rabbit_{}.ply".format(i), points2,[])



'''
for i in trange(max(index,0),NUMBER_SAMPLES):
    a=0.3
    init_deform=-a+2*a*rng.uniform(size=(nx,ny,nz,3))
    latent[i]=init_deform
    modifiable=np.full((nx,ny,nz,3), True)
    M=points.copy()
    temp_new=ffd.ffd(M)    
    alls[i]=temp_new
    meshio.write_points_cells("./data_objects/rabbit_{}.ply".format(i), temp_new,[])

with open('seed_state', 'wb') as handle:
    pickle.dump(rng.bit_generator.state,handle)
'''

latent=latent.reshape(NUMBER_SAMPLES,-1)
np.save("latent_ffd",latent)
pca=PCA()
alls=alls.reshape(NUMBER_SAMPLES,-1)
pca.fit(alls)
precision=np.cumsum(pca.explained_variance_ratio_)
print(np.argmin(np.abs(precision-(1-1e-10))))
