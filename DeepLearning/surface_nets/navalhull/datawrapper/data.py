#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:28:55 2023

@author: cyberguli
"""
import torch
import meshio
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import copy

from models.basic_layers.PCA import PCA
from torch.utils.data import random_split

def getinfo(stl,batch_size,flag):
    mesh=meshio.read(stl)
    points_all=torch.tensor(mesh.points.astype(np.float32))
    newmesh_indices_global_1=np.arange(len(points_all))[(points_all[:,2]>0) & (points_all[:,0]>0) & (points_all[:,1]>0) ].tolist()
    newmesh_indices_global_2=np.arange(len(points_all))[(points_all[:,2]>0) & (points_all[:,0]>0) & (points_all[:,1]==0) ].tolist()
    points_interior=points_all[newmesh_indices_global_1]
    points_boundary=points_all[newmesh_indices_global_2] #NOT NUMERICALLY STABLE
    if flag==True:
        triangles=torch.tensor(mesh.cells_dict['triangle'].astype(np.int64))
        triangles=triangles.long()
        #newmesh_indices_global_zero=np.arange(len(mesh.points))[(points_all[:,2]>=0) & (points_all[:,0]>=0)].tolist()
        tmp=triangles[torch.isin(triangles,torch.tensor(newmesh_indices_global_1+newmesh_indices_global_2)).reshape(-1,3).sum(axis=1).bool()]
        newmesh_indices_global_zero=torch.unique(tmp.reshape(-1)).tolist()
        points_fixed=points_all[newmesh_indices_global_zero] #NOT NUMERICALLY STABLE
        newmesh_indices_local_1=torch.arange(len(points_fixed))[(points_fixed[:,2]>0) & (points_fixed[:,0]>0) & (points_fixed[:,1]>0) ].tolist()
        newmesh_indices_local_2=torch.arange(len(points_fixed))[(points_fixed[:,2]>0) & (points_fixed[:,0]>0) & (points_fixed[:,1]==0) ].tolist()
        newmesh_indices_local=newmesh_indices_local_1+newmesh_indices_local_2
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])

        edge_matrix=torch.zeros(torch.max(torch.tensor(newtriangles_zero))+1,torch.max(torch.tensor(newtriangles_zero))+1)
        vertices_face_x=[set({}) for i in range(len(newmesh_indices_local_1))]
        vertices_face_xy=[set({}) for i in range(len(newmesh_indices_local))]

        for i in range(len(newtriangles_zero)):
            T=newtriangles_zero[i]
            if T[0] in newmesh_indices_local_1:
                vertices_face_x[newmesh_indices_local_1.index(T[0])].add(i)
                vertices_face_xy[newmesh_indices_local.index(T[0])].add(i)
                edge_matrix[T[0],T[1]]=1
                edge_matrix[T[0],T[2]]=1
            else:
                edge_matrix[T[0],T[0]]=1

                
            if T[1] in newmesh_indices_local_1:
                vertices_face_x[newmesh_indices_local_1.index(T[1])].add(i)
                vertices_face_xy[newmesh_indices_local.index(T[1])].add(i)
                edge_matrix[T[1],T[0]]=1
                edge_matrix[T[1],T[2]]=1
            else:
                edge_matrix[T[1],T[1]]=1

                
            if T[2] in newmesh_indices_local_1:
                vertices_face_x[newmesh_indices_local_1.index(T[2])].add(i)
                vertices_face_xy[newmesh_indices_local.index(T[2])].add(i)
                edge_matrix[T[2],T[0]]=1
                edge_matrix[T[2],T[1]]=1
            else:
                edge_matrix[T[2],T[2]]=1

            if T[0] in newmesh_indices_local_2:
                vertices_face_xy[newmesh_indices_local.index(T[0])].add(i)
                
            if T[1] in newmesh_indices_local_2:

                vertices_face_xy[newmesh_indices_local.index(T[1])].add(i)

            if T[2] in newmesh_indices_local_2:
                vertices_face_xy[newmesh_indices_local.index(T[2])].add(i)

        


            
        vertices_face_x=[list(t) for t in vertices_face_x]
        vertices_face_xy=[list(t) for t in vertices_face_xy]

        

    else:
        triangles=0
        newmesh_indices_local_1=0
        newmesh_indices_local_2=0
        newmesh_indices_global_1=0
        newmesh_indices_global_2=0
        newtriangles_zero=0
        edge_matrix=0
        vertices_face_x=0
        vertices_face_xy=0
        points_fixed=0
        
    return points_interior,points_boundary,points_fixed,points_all,newmesh_indices_local_1,newmesh_indices_local_2,newmesh_indices_global_1,newmesh_indices_global_2,triangles,newtriangles_zero,edge_matrix,vertices_face_x,vertices_face_xy


class Data(LightningDataModule):
    def get_size(self):
        return ((1,self.size_interior,3),(1,self.size_boundary,2))
    
    def get_reduced_size(self):
        return self.reduced_dimension

    def __init__(
        self,batch_size,num_workers,num_train,num_test,num_val,reduced_dimension,string,use_cuda):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.num_train=num_train
        self.num_val=num_val
        self.num_workers = num_workers
        self.num_test=num_test
        self.reduced_dimension=reduced_dimension
        self.string=string
        self.num_samples=self.num_test+self.num_train+self.num_val
        temp_interior,temp_boundary,_,_,_,_,_,_,_,_,_,_,_=getinfo(self.string.format(0),self.batch_size,False)
        self.size_interior=temp_interior.shape[0]
        self.size_boundary=temp_boundary.shape[0]
        
        
        _,_,self.temp_zero,self.oldmesh,self.local_indices_1,self.local_indices_2,self.global_indices_1,self.global_indices_2,self.oldM,self.newtriangles_zero,self.edge_matrix,self.vertices_face_x,self.vertices_face_xy=getinfo(self.string.format(0),self.batch_size,True)
        vertices_face_2_x=copy.deepcopy(self.vertices_face_x)
        self.vertices_face_x=torch.zeros(self.get_size()[0][1],len(self.newtriangles_zero))
        for i in range(len(vertices_face_2_x)):
            for j in vertices_face_2_x[i]:
                self.vertices_face_x[i,j]=1
        vertices_face_2_xy=copy.deepcopy(self.vertices_face_xy)
        self.vertices_face_xy=torch.zeros(self.get_size()[0][1]+self.get_size()[1][1],len(self.newtriangles_zero))
        for i in range(len(vertices_face_2_xy)):
            for j in vertices_face_2_xy[i]:
                self.vertices_face_xy[i,j]=1

        data_interior=torch.zeros(self.num_samples,self.get_size()[0][1],self.get_size()[0][2])
        data_boundary=torch.zeros(self.num_samples,self.get_size()[1][1],self.get_size()[1][2])
        for i in range(0,self.num_samples):
            if i%100==0:
                print(i)
            tmp1,tmp2,_,_,_,_,_,_,_,_,_,_,_=getinfo(self.string.format(i),self.batch_size,False)
            data_interior[i]=tmp1
            data_boundary[i,:,0]=tmp2[:,0]
            data_boundary[i,:,1]=tmp2[:,2]
        self.data=torch.concat((data_interior.reshape(self.num_samples,-1),data_boundary.reshape(self.num_samples,-1)),axis=1)
        
        self.pca=PCA(self.reduced_dimension)
        if use_cuda:
            self.pca.fit(self.data.cuda())
            self.temp_zero=self.temp_zero.cuda()
            self.vertices_face_x=self.vertices_face_x.cuda()
            self.vertices_face_xy=self.vertices_face_xy.cuda()
            self.edge_matrix=self.edge_matrix.cuda()
        else:
            self.pca.fit(self.data)

        self.data_train,self.data_val,self.data_test = random_split(self.data, [self.num_train,self.num_val,self.num_test])    

    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True

        )