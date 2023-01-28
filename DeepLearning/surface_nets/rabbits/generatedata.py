#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:38:18 2022

@author: cyberguli
"""

import scipy
import numpy as np
np.random.seed(0)
import meshio
from sklearn.decomposition import PCA
import time
from tqdm import trange

class FFD():
    def __init__(self,box_origin,box_length,n_control,modifiable=0,initial_deform=0):
        self.box_origin=np.array(box_origin)
        self.box_length=np.array(box_length)
        self.n_control=np.array(n_control, dtype=np.int64)
        self.control_points=np.zeros([n_control[0]+1,n_control[1]+1,n_control[2]+1,3])
        self.modifiable=modifiable
        self.initial_deform=initial_deform
        for i in range(n_control[0]+1):
            for j in range(n_control[1]+1):
                for k in range(n_control[2]+1):
                    self.control_points[i,j,k]=1/(self.n_control)*np.array([i,j,k])
    
        self.control_points=self.control_points+initial_deform
        
        
        
        
        
    def bernestein_point(self,x):
        b_x=scipy.special.comb(self.n_control[0],np.arange(self.n_control[0]+1))*x[0]**(np.arange(self.n_control[0]+1))*(1-x[0])**(self.n_control[0]-np.arange(self.n_control[0]+1))
        b_y=scipy.special.comb(self.n_control[1],np.arange(self.n_control[1]+1))*x[1]**(np.arange(self.n_control[1]+1))*(1-x[1])**(self.n_control[1]-np.arange(self.n_control[1]+1))
        b_z=scipy.special.comb(self.n_control[2],np.arange(self.n_control[2]+1))*x[2]**(np.arange(self.n_control[2]+1))*(1-x[2])**(self.n_control[2]-np.arange(self.n_control[2]+1))
        return np.einsum('i,j,k->ijk', b_x.ravel(), b_y.ravel(), b_z.ravel())

    def bernestein_mesh(self,mesh_points):
        new_mesh=np.zeros([self.n_control[0]+1,self.n_control[1]+1,self.n_control[2]+1,len(mesh_points)])
        for i in range(len(mesh_points)):
            new_mesh[:,:,:,i]=self.bernestein_point(mesh_points[i])
        return new_mesh
        
    
    def mesh_to_local_space(self, mesh):
        return (mesh-self.box_origin)/self.box_length;
        
    def mesh_to_global_space(self,mesh):
        return mesh*self.box_length+self.box_origin
    
    def apply_to_point(self,x):
        return np.sum(self.control_points*np.repeat(self.bernestein_point(x)[:,:,:,np.newaxis],3,axis=3),axis=(0,1,2))
    
    
    def apply_to_mesh(self,mesh_points):
        a=np.repeat(self.control_points[:,:,:,np.newaxis,:],len(mesh_points),axis=3)
        b=np.repeat(self.bernestein_mesh(mesh_points)[:,:,:,:,np.newaxis],3,axis=4)
        return np.sum(a*b,axis=(0,1,2))

    def adjust_def(self,M_local):
        _sum=np.sum(M_local,axis=0)
        _sum_new=np.sum(self.apply_to_mesh(M_local),axis=0)
        diff=_sum-_sum_new
        bmesh=self.bernestein_mesh(M_local)
        alpha_x=np.sum(bmesh,axis=3)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if self.modifiable[i,j,k,0]==False:
                        
                        alpha_x[i,j,k]=0
        def_x=alpha_x*diff[0]/np.sum(alpha_x**2)
        self.control_points[:,:,:,0]=self.control_points[:,:,:,0]+def_x
        alpha_y=np.sum(bmesh,axis=3)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if self.modifiable[i,j,k,1]==False:
                        
                        alpha_y[i,j,k]=0
        def_y=alpha_y*diff[1]/np.sum(alpha_y**2)
        self.control_points[:,:,:,1]=self.control_points[:,:,:,1]+def_y
        alpha_z=np.sum(bmesh,axis=3)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if self.modifiable[i,j,k,2]==False:
                        alpha_z[i,j,k]=0
        def_z=alpha_z*diff[2]/np.sum(alpha_z**2)
        self.control_points[:,:,:,2]=self.control_points[:,:,:,2]+def_z
        

        

    def ffd(self,M):
        M=self.mesh_to_local_space(M)
        self.adjust_def(M)
        M=self.apply_to_mesh(M)
        M=self.mesh_to_global_space(M)
        return M
        
        



def getinfo(stl):
    mesh=meshio.read(stl)
    mesh.points[abs(mesh.points)<10e-05]=0
    points_old=mesh.points.astype(np.float32)
    indices=np.arange(len(points_old))[points_old[:,1]>0]
    points=points_old[points_old[:,1]>0]
    points_zero=points_old[points_old[:,1]==0]
    barycenter=np.mean(points,axis=0)
    return points,points_zero,points_old,indices,barycenter


points,points_zero,points_old,indices,barycenter=getinfo("./data_objects/rabbit_translated.ply")

alls=np.zeros([600,21306,3])


for i in trange(600):
    if i%100==0:
        print(i)
    a=0.1
    init_deform=-a+2*a*np.random.rand(4,4,4,3)
    init_deform[:,0,:,:]=0
    modifiable=np.full((4,4,4,3), True)
    modifiable[:,0,:,:]=False    
    M=points.copy()
    ffd=FFD([np.min(M[:,0]), np.min(M[:,1]), np.min(M[:,2])],[np.max(M[:,0])-np.min(M[:,0]), np.max(M[:,1])-np.min(M[:,1]), np.max(M[:,2])-np.min(M[:,2])],[3, 3, 3], modifiable, init_deform)
    temp_new=ffd.ffd(M)    
    points_new=points_old.copy()
    points_new[indices,:]=temp_new
    alls[i]=temp_new
    meshio.write_points_cells("./data_objects/rabbit_{}.ply".format(i), points_new,[])
    
pca=PCA()
alls=alls.reshape(600,-1)
pca.fit(alls)
precision=np.cumsum(pca.explained_variance_ratio_)
print(np.argmin(np.abs(precision-(1-1e-10))))

