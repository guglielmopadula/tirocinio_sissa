#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cyberguli
"""

import scipy
import numpy as np
np.random.seed(0)
import meshio
from tqdm import trange
from sklearn.decomposition import PCA
from skdim.id import TwoNN
from scipy.stats import truncnorm

def volume_prism_x(M):
    return np.sum(M[:,0])*(np.linalg.det(M[1:,1:]-M[0,1:])/6)

def volume_prism_y(M):
    return np.sum(M[:,1])*(np.linalg.det(M[np.ix_((0,2),(0,2))]-M[1,[0,2]]))/6

def volume_prism_z(M):
    return np.sum(M[:,2])*(np.linalg.det(M[:2,:2]-M[2,:2])/6)

def volume_2_x(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_prism_x(mesh[i,:,:])
    return volume

def volume_2_y(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_prism_y(mesh[i,:,:])
    return volume

def volume_2_z(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_prism_z(mesh[i,:,:])
    return volume

def volume_2_x(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,0],axis=2)*(np.linalg.det(mesh[:,:,1:,1:]-np.expand_dims(mesh[:,:,0,1:],2))/6),axis=1)
    return tmp.reshape(shape[:-3])


def volume_2_z(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,2],axis=2)*(np.linalg.det(mesh[:,:,:2,:2]-np.expand_dims(mesh[:,:,2,:2],2))/6),axis=1)
    return tmp.reshape(shape[:-3])

def volume_2_y(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,1],axis=2)*(np.linalg.det(mesh[np.ix_(np.arange(mesh.shape[0]),np.arange(mesh.shape[1]),(0,2),(0,2))]-mesh[np.ix_(np.arange(mesh.shape[0]),np.arange(mesh.shape[1]),[1],(0,2))])/6),axis=1)
    return tmp.reshape(shape[:-3])



def get_coeff_z(vertices_face,points_zero,newtriangles_zero):
    return np.array([np.sum(np.linalg.det(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,:2,:2]-np.repeat(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,2,:2,np.newaxis],2,axis=0).reshape(-1,2,2))/6) for i in range(len(vertices_face))])

def get_coeff_x(vertices_face,points_zero,newtriangles_zero):
    return np.array([np.sum(np.linalg.det(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,1:,1:]-np.repeat(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,0,1:,np.newaxis],2,axis=0).reshape(-1,2,2))/6) for i in range(len(vertices_face))])

def get_coeff_y(vertices_face,points_zero,newtriangles_zero):
    return np.array([np.sum(np.linalg.det(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][np.ix_(np.arange(len(vertices_face[i])),(0,2),(0,2))]-np.repeat(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,1,[0,2],np.newaxis],2,axis=0).reshape(-1,2,2))/6) for i in range(len(vertices_face))])


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
        return (mesh-self.box_origin)/self.box_length
        
    def mesh_to_global_space(self,mesh):
        return mesh*self.box_length+self.box_origin
    
    def apply_to_point(self,x):
        return np.sum(self.control_points*np.repeat(self.bernestein_point(x)[:,:,:,np.newaxis],3,axis=3),axis=(0,1,2))
    
    
    def apply_to_mesh(self,mesh_points):
        a=np.repeat(self.control_points[:,:,:,np.newaxis,:],len(mesh_points),axis=3)
        b=np.repeat(self.bernestein_mesh(mesh_points)[:,:,:,:,np.newaxis],3,axis=4)
        return np.sum(a*b,axis=(0,1,2))

    def adjust_def(self,M_local,triangles):
        Vtrue=volume_2_x(M_local[triangles])
        Vnew=volume_2_x(self.apply_to_mesh(M_local)[triangles])
        ax=1/3*(Vtrue-Vnew)
        ay=1/3*(Vtrue-Vnew)
        az=1/3*(Vtrue-Vnew)
        bmesh=self.bernestein_mesh(M_local)
        M_def=self.apply_to_mesh(M_local)
        temp=np.tile(M_def[np.newaxis,np.newaxis,np.newaxis,:,:],[self.n_control[0]+1,self.n_control[1]+1,self.n_control[2]+1,1,1])
        temp_x=temp
        temp_x[:,:,:,:,0]=bmesh
        temp_x=temp_x[:,:,:,triangles,:]
        alpha_x=np.zeros(self.n_control+1)
        '''
        for i in range(self.n_control[0]+1):
            for j in range(self.n_control[1]+1):
                for k in range(self.n_control[2]+1):
                    if self.modifiable[i,j,k,0]==True:
                        alpha_x[i,j,k]=volume_2_x(temp_x[i,j,k])
        '''
        alpha_x=volume_2_x(temp_x)*self.modifiable[:,:,:,0]
        def_x=alpha_x*ax/np.sum(alpha_x**2)
        self.control_points[:,:,:,0]=self.control_points[:,:,:,0]+def_x
        bmesh=self.bernestein_mesh(M_local)
        M_def=self.apply_to_mesh(M_local)
        temp=np.tile(M_def[np.newaxis,np.newaxis,np.newaxis,:,:],[self.n_control[0]+1,self.n_control[1]+1,self.n_control[2]+1,1,1])
        temp_y=temp
        temp_y[:,:,:,:,1]=bmesh
        temp_y=temp_y[:,:,:,triangles,:]
        alpha_y=np.zeros(self.n_control+1)
        '''
        for i in range(self.n_control[0]+1):
            for j in range(self.n_control[1]+1):
                for k in range(self.n_control[2]+1):
                    if self.modifiable[i,j,k,1]==True:
                        alpha_y[i,j,k]=volume_2_y(temp_y[i,j,k])
        '''
        alpha_y=volume_2_y(temp_y)*self.modifiable[:,:,:,1]
        def_y=alpha_y*ay/np.sum(alpha_y**2)
        self.control_points[:,:,:,1]=self.control_points[:,:,:,1]+def_y
        bmesh=self.bernestein_mesh(M_local)
        M_def=self.apply_to_mesh(M_local)
        temp=np.tile(M_def[np.newaxis,np.newaxis,np.newaxis,:,:],[self.n_control[0]+1,self.n_control[1]+1,self.n_control[2]+1,1,1])
        temp_z=temp
        temp_z[:,:,:,:,2]=bmesh
        temp_z=temp_z[:,:,:,triangles,:]
        alpha_z=np.zeros(self.n_control+1)
        '''
        for i in range(self.n_control[0]+1):
            for j in range(self.n_control[1]+1):
                for k in range(self.n_control[2]+1):
                    if self.modifiable[i,j,k,2]==True:
                        alpha_z[i,j,k]=volume_2_z(temp_z[i,j,k])
        '''
        alpha_z=volume_2_z(temp_z)*self.modifiable[:,:,:,2]
        def_z=alpha_z*az/np.sum(alpha_z**2)
        self.control_points[:,:,:,2]=self.control_points[:,:,:,2]+def_z

    def ffd(self,M,triangles):
        a=volume_2_y(M[triangles])
        M=self.mesh_to_local_space(M)
        self.adjust_def(M, triangles)
        M=self.apply_to_mesh(M)
        M=self.mesh_to_global_space(M)
        b=volume_2_y(M[triangles])
        return M
        
        
        
        

def getinfo(stl,flag):
    mesh=meshio.read(stl)
    mesh.points[abs(mesh.points)<10e-05]=0
    points_old=mesh.points.astype(np.float32)
    points=points_old[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)]
    if flag==True:
        newmesh_indices_global=np.arange(len(mesh.points))[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)].tolist()
        triangles=mesh.cells_dict['triangle'].astype(np.int64)
        newtriangles=[]
        for T in triangles:
            if T[0] in newmesh_indices_global and T[1] in newmesh_indices_global and T[2] in newmesh_indices_global:
                newtriangles.append([newmesh_indices_global.index(T[0]),newmesh_indices_global.index(T[1]),newmesh_indices_global.index(T[2])])
        #newmesh_indices_global_zero=np.arange(len(mesh.points))[np.logical_and(points_old[:,2]>=0,points_old[:,0]>=0)].tolist()
        tmp=triangles[np.in1d(triangles,np.array(newmesh_indices_global)).reshape(-1,3).sum(axis=1).astype(bool)]
        newmesh_indices_global_zero=np.unique(tmp.reshape(-1)).tolist()
        points_zero=points_old[newmesh_indices_global_zero]
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])
        newmesh_indices_local=np.arange(len(points_zero))[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)].tolist()
        newtriangles_local_3=[]
        newtriangles_local_2=[]
        newtriangles_local_1=[]
        edge_matrix=np.zeros([np.max(newtriangles_zero)+1,np.max(newtriangles_zero)+1])
        vertices_face=[set({}) for i in range(len(newmesh_indices_local))]
        for T in newtriangles_zero:
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==3:
                newtriangles_local_3.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==2:
                newtriangles_local_2.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==1:
                newtriangles_local_1.append([T[0],T[1],T[2]])
        
        for i in range(len(newtriangles_zero)):
            T=newtriangles_zero[i]
            if T[0] in newmesh_indices_local:
                edge_matrix[T[0],T[1]]=1
                edge_matrix[T[0],T[2]]=1
                vertices_face[newmesh_indices_local.index(T[0])].add(i)
            else:
                edge_matrix[T[0],T[0]]=1
                
            if T[1] in newmesh_indices_local:
                edge_matrix[T[1],T[2]]=1
                edge_matrix[T[1],T[0]]=1
                vertices_face[newmesh_indices_local.index(T[1])].add(i)
            else:
                edge_matrix[T[1],[T[1]]]=1
                
                
            if T[2] in newmesh_indices_local:
                edge_matrix[T[2],T[0]]=1
                edge_matrix[T[2],T[1]]=1
                vertices_face[newmesh_indices_local.index(T[2])].add(i)
            else:
                edge_matrix[T[2],[T[2]]]=1
        vertices_face=[list(t) for t in vertices_face]

    else:
        triangles=0
        newtriangles=0
        newmesh_indices_local=0
        newtriangles_zero=0
        newtriangles_local_1=0
        newtriangles_local_2=0
        newtriangles_local_3=0
        vertices_face=0
        edge_matrix=0
        
    return points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix,vertices_face


points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix,vertices_face=getinfo("data_objects/segmented.stl",True)

temp=points_zero.copy()
base=np.arange(len(points_zero))[(points_zero[:,0]>0)*(points_zero[:,2]>0)*(points_zero[:,1]==0)]

temp1=points_zero[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)]
NUM_SAMPLES=600
alls=np.zeros([NUM_SAMPLES,2572,3])

latent=np.zeros([NUM_SAMPLES,nx,ny,nz,3])
nx=5
ny=5
nz=5
b=((np.outer(np.outer(1/np.arange(1,nx+1),1/np.arange(1,ny+1)),1/np.arange(1,nz+1)).reshape(nx,ny,nz,1)).repeat(3,3))**(1/4)

for i in trange(NUM_SAMPLES):
    a=0.15
    init_deform=a+truncnorm.rvs(a=-1,b=1,size=(nx,ny,nz,3))
    init_deform[0,:,:,:]=0
    init_deform[:,0,:,:]=0
    init_deform[:,:,0,:]=0
    init_deform[nx-1,:,:,:]=0
    init_deform[:,ny-1,:,:]=0
    init_deform[:,:,nz-1,:]=0
    modifiable=np.full((nx,ny,nz,3), True)
    modifiable[0,:,:,:]=False
    modifiable[:,0,:,:]=False
    modifiable[:,:,0,:]=False
    modifiable[nx-1,:,:,:]=False
    modifiable[:,ny-1,:,:]=False
    modifiable[:,:,nz-1,:]=False
    modifiable[nx-1,0,:,0]=True
    init_deform[nx-1,0,:,0]=a+0.01*truncnorm.rvs(a=-1,b=1)
    latent[i]=init_deform
    print(np.sum(modifiable))
    
    M=temp
    ffd=FFD([np.min(M[:,0]), np.min(M[:,1]), np.min(M[:,2])],[np.max(M[:,0])-np.min(M[:,0]), np.max(M[:,1])-np.min(M[:,1]), np.max(M[:,2])-np.min(M[:,2])],[nx-1, ny-1, nz-1], modifiable, init_deform)

    
    temp_new=ffd.ffd(M,newtriangles_zero)
    if np.prod(temp_new[base,1]==0)==0:
        print("errore")
        break
    
    points_new=points_old.copy()
    points_new[newmesh_indices_global_zero]=temp_new
    alls[i]=temp_new
    meshio.write_points_cells("./data_objects/hull_{}.stl".format(i), points_new, [("triangle", triangles)])
    
latent=latent.reshape(NUM_SAMPLES,-1)[:,modifiable.reshape(-1)]
print(latent.shape)
np.save("ffd_latent",latent)
pca=PCA()
alls=alls.reshape(10,-1)
pca.fit(alls)
precision=np.cumsum(pca.explained_variance_ratio_)
print(np.argmin(np.abs(precision-(1-1e-12))))
print(TwoNN().fit(alls).dimension_)

