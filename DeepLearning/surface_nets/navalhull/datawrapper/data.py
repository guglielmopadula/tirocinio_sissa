#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:28:55 2023

@author: cyberguli
"""
import torch
import meshio
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import copy

from models.basic_layers.PCA import PCA
from torch.utils.data import random_split

def getinfo(stl,batch_size,flag):
    mesh=meshio.read(stl)
    points_all=torch.tensor(mesh.points.astype(np.float32))
    points_fixed=points_all[(points_all[:,2]>=0) & (points_all[:,0]>=0)] #NOT NUMERICALLY STABLE
    newmesh_indices_local_1=torch.arange(len(points_fixed))[(points_fixed[:,2]>0) & (points_fixed[:,0]>0) & (points_fixed[:,1]>0) ].tolist()
    newmesh_indices_local_2=torch.arange(len(points_fixed))[(points_fixed[:,2]>0) & (points_fixed[:,0]>0) & (points_fixed[:,1]==0) ].tolist()
    newmesh_indices_global_1=np.arange(len(points_all))[(points_all[:,2]>0) & (points_all[:,0]>0) & (points_all[:,1]>0) ].tolist()
    newmesh_indices_global_2=np.arange(len(points_all))[(points_all[:,2]>0) & (points_all[:,0]>0) & (points_all[:,1]==0) ].tolist()
    points_interior=points_all[newmesh_indices_global_1]
    points_boundary=points_all[newmesh_indices_global_2] #NOT NUMERICALLY STABLE
    if flag==True:
        triangles=torch.tensor(mesh.cells_dict['triangle'].astype(np.int64))
        triangles=triangles.long()
        newmesh_indices_global_zero=np.arange(len(mesh.points))[(points_all[:,2]>=0) & (points_all[:,0]>=0)].tolist()
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])

        edge_matrix=torch.zeros(torch.max(torch.tensor(newtriangles_zero))+1,torch.max(torch.tensor(newtriangles_zero))+1)
        vertices_face=[set({}) for i in range(len(newmesh_indices_local_1))]
        for i in range(len(newtriangles_zero)):
            T=newtriangles_zero[i]
            if T[0] in newmesh_indices_local_1:
                edge_matrix[T[0],T[1]]=1
                edge_matrix[T[0],T[2]]=1
                vertices_face[newmesh_indices_local_1.index(T[0])].add(i)
            else:
                edge_matrix[T[0],T[0]]=1
                
            if T[1] in newmesh_indices_local_1:
                edge_matrix[T[1],T[2]]=1
                edge_matrix[T[1],T[0]]=1
                vertices_face[newmesh_indices_local_1.index(T[1])].add(i)
            else:
                edge_matrix[T[1],[T[1]]]=1
                
                
            if T[2] in newmesh_indices_local_1:
                edge_matrix[T[2],T[0]]=1
                edge_matrix[T[2],T[1]]=1
                vertices_face[newmesh_indices_local_1.index(T[2])].add(i)
            else:
                edge_matrix[T[2],[T[2]]]=1
        vertices_face=[list(t) for t in vertices_face]
        x = cp.Variable((batch_size,len(vertices_face)))
        coeff = cp.Parameter((batch_size, len(vertices_face)))
        a=cp.Parameter(batch_size)
        x0=cp.Parameter((batch_size,len(vertices_face)))
        prob = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(x)),
                         [-x <= (1-0.05)*x0,
                           cp.sum(cp.multiply(x,coeff),axis=1) == a])
        cvxpylayer = CvxpyLayer(prob, parameters=[coeff, x0,a], variables=[x])
        x2 = cp.Variable((1,len(vertices_face)))
        coeff2 = cp.Parameter((1, len(vertices_face)))
        a2=cp.Parameter(1)
        x02=cp.Parameter((1,len(vertices_face)))
        prob2 = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(x2)),
                          [-x2 <= (1-0.05)*x02,
                           cp.sum(cp.multiply(x2,coeff2),axis=1) == a2])
        cvxpylayer_single = CvxpyLayer(prob2, parameters=[coeff2, x02,a2], variables=[x2])
                

        

    else:
        triangles=0
        newmesh_indices_local_1=0
        newmesh_indices_local_2=0
        newmesh_indices_global_1=0
        newmesh_indices_global_2=0
        newtriangles_zero=0
        cvxpylayer=0
        cvxpylayer_single=0
        edge_matrix=0
        vertices_face=0
        
    return points_interior,points_boundary,points_fixed,points_all,newmesh_indices_local_1,newmesh_indices_local_2,newmesh_indices_global_1,newmesh_indices_global_2,triangles,newtriangles_zero,edge_matrix,vertices_face,[cvxpylayer,cvxpylayer_single]


class Data(LightningDataModule):
    def get_size(self):
        temp_interior,temp_boundary,_,_,_,_,_,_,_,_,_,_,_=getinfo(self.string.format(0),self.batch_size,False)
        return ((1,temp_interior.shape[0],3),(1,temp_boundary.shape[0],2))
    
    def get_reduced_size(self):
        return (self.reduced_dimension_1,self.reduced_dimension_2)

    def __init__(
        self,batch_size,num_workers,num_train,num_val,num_test,reduced_dimension_1,reduced_dimension_2,string,use_cuda):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.num_train=num_train
        self.num_workers = num_workers
        self.num_val=num_val
        self.num_test=num_test
        self.reduced_dimension_1=reduced_dimension_1
        self.reduced_dimension_2=reduced_dimension_2
        self.string=string
        self.num_samples=self.num_test+self.num_train+self.num_val
        
        _,_,self.temp_zero,self.oldmesh,self.local_indices_1,self.local_indices_2,self.global_indices_1,self.global_indices_2,self.oldM,self.newtriangles_zero,self.edge_matrix,self.vertices_face,self.cvxpylayer=getinfo(self.string.format(0),self.batch_size,True)
        vertices_face_2=copy.deepcopy(self.vertices_face)
        self.vertices_face=torch.zeros(self.get_size()[0][1],len(self.newtriangles_zero))
        for i in range(len(vertices_face_2)):
            for j in vertices_face_2[i]:
                self.vertices_face[i,j]=1
        self.data_interior=torch.zeros(self.num_samples,self.get_size()[0][1],self.get_size()[0][2])
        self.data_boundary=torch.zeros(self.num_samples,self.get_size()[1][1],self.get_size()[1][2])
        for i in range(0,self.num_samples):
            if i%100==0:
                print(i)
            self.data_interior[i],tmp,_,_,_,_,_,_,_,_,_,_,_=getinfo(self.string.format(i),self.batch_size,False)
            self.data_boundary[i,:,0]=tmp[:,0]
            self.data_boundary[i,:,1]=tmp[:,2]
        self.data=torch.utils.data.TensorDataset(self.data_interior,self.data_boundary)
        self.pca_1=PCA(self.reduced_dimension_1)
        self.pca_2=PCA(self.reduced_dimension_2)
        if use_cuda:
            self.pca_1.fit(self.data_interior.reshape(self.num_samples,-1).cuda())
            self.pca_2.fit(self.data_boundary.reshape(self.num_samples,-1).cuda())
            self.temp_zero=self.temp_zero.cuda()
            self.vertices_face=self.vertices_face.cuda()
            self.edge_matrix=self.edge_matrix.cuda()
        else:
            self.pca_1.fit(self.data_interior.reshape(self.num_samples,-1))
            self.pca_2.fit(self.data_boundary.reshape(self.num_samples,-1))

        self.data_train, self.data_val,self.data_test = random_split(self.data, [self.num_train,self.num_val,self.num_test])    

    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )