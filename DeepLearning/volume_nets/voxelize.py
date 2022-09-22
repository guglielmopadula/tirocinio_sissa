#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:02:00 2022

@author: cyberguli
"""
import time
import numpy as np
import torch
import meshio
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

def reshape_array(arr):
    sub_shape = (2,2,2)
    view_shape = tuple(np.subtract(arr.shape, sub_shape) + 1) + sub_shape
    arr_view = as_strided(arr, view_shape, arr.strides * 2)
    arr_view = arr_view.reshape((-1,) + sub_shape)
    return arr_view

class Bounding_box:
    def __init__(self, npoints_dim):
        self.npoints_dim=npoints_dim
        self.index=np.arange(npoints_dim**3).reshape(npoints_dim,npoints_dim,npoints_dim)
        self.cubes=reshape_array(self.index)
        self.color_points=np.zeros(npoints_dim**3)
        self.color_cube=np.zeros((npoints_dim-1)**3)
        
    def local_index(self,index):
        z=index//(self.npoints_dim**2)
        temp=(index)%(self.npoints_dim**2)
        y=temp//self.npoints_dim
        x=temp%self.npoints_dim
        return np.array([x,y,z],dtype=np.int64)
    
    def coord(self,index):
        local_index=self.local_index(index)
        return -1+local_index/(self.npoints_dim-1)*2
    
    def points(self):
        return self.coord(self.index.reshape(-1)).T
    
    def intersect(self,mesh):
        for i in range(len(mesh.cells_dict["tetra"])):
            h=pointInside(self.points(),mesh.points[mesh.cells_dict["tetra"][i]])
            self.color_points[h]+=1
        self.color_cube=(self.color_points[self.cubes].sum(axis=(1,2,3))>0).astype(int)

    def save(self, name):
        temp=torch.tensor(self.color_cube.reshape(self.npoints_dim-1,self.npoints_dim-1,self.npoints_dim-1))
        torch.save(temp, name)
        
    def plot(self):
        voxelarray=(self.color_cube==1).reshape(self.npoints_dim-1,self.npoints_dim-1,self.npoints_dim-1)
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxelarray, facecolors='black', edgecolor='k')
        plt.show()
        

def rescale_mesh(mesh,value):
    temp=mesh
    temp.points=mesh.points/value*2
    temp.points=temp.points-(np.max(mesh.points,axis=0)+np.min(mesh.points,axis=0))/2
    return temp



def voxelize(STRING_INPUT,num,res,STRING_OUTPUT):
    value=0
    for i in range(num):
        mesh=meshio.read(STRING_INPUT.format(i))
        value=np.max([value,np.max((np.max(mesh.points,axis=0)+np.min(mesh.points,axis=0))),1])
    
    for i in range(num):
        mesh=meshio.read(STRING_INPUT.format(i))
        mesh=rescale_mesh(mesh,value)
        a=Bounding_box(res)
        a.intersect(mesh)
        a.plot()
        a.save(STRING_OUTPUT.format(i))

                     

def pointInside(point, vertices):
    origin, *rest = vertices
    mat = (np.array(rest) - origin).T
    tetra = np.linalg.inv(mat)
    newp = np.matmul(tetra, (point-origin).T).T
    return np.all(newp>=0, axis=-1) & np.all(newp <=1, axis=-1) & (np.sum(newp, axis=-1) <=1)

t=time.time()
voxelize("bulbo_{}.vtk",200,31,"bulbo_{}.pt")
s=time.time()


