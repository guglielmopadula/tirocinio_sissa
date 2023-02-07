#!/usr/bin/env python
# coding: utf-8

import numpy as np
from torch.utils.data import DataLoader
import meshio 
import os
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import random_split
import logging
import trimesh
from torch import autograd
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt
import itertools

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda=True if torch.cuda.is_available() else False
torch.manual_seed(100)
np.random.seed(100)

NUM_LAPL=100
REDUCED_DIMENSION_1=24
REDUCED_DIMENSION_2=6   


#NUM_TRAIN_SAMPLES=4
#NUM_VAL_SAMPLES=2
#NUM_TEST_SAMPLES=2
#NUMBER_SAMPLES=NUM_TRAIN_SAMPLES+NUM_TEST_SAMPLES+NUM_VAL_SAMPLES
#BATCH_SIZE = 2
#MAX_EPOCHS=5


NUM_TRAIN_SAMPLES=400
NUM_VAL_SAMPLES=100
NUM_TEST_SAMPLES=100
NUMBER_SAMPLES=NUM_TRAIN_SAMPLES+NUM_TEST_SAMPLES+NUM_VAL_SAMPLES
BATCH_SIZE = 20
MAX_EPOCHS=500




STRING="/home/cyberguli/tirocinio_sissa/DeepLearning/surface_nets/navalhull/hull_{}.stl"
AVAIL_GPUS = torch.cuda.device_count()
NUM_WORKERS = int(os.cpu_count() / 2)
LATENT_DIM_1=11
LATENT_DIM_2=3
LOGGING=1
SMOOTHING_DEGREE=1
DROP_PROB=0.1


class PCA():
    def __init__(self,reduced_dim):
        self._reduced_dim=reduced_dim
        
    def fit(self,matrix):
        self._n=matrix.shape[0]
        self._p=matrix.shape[1]
        mean=torch.mean(matrix,dim=0)
        self._mean_matrix=torch.mm(torch.ones(self._n,1),mean.reshape(1,self._p))
        X=matrix-self._mean_matrix
        Cov=np.matmul(X.t(),X)/self._n
        self._V,S,_=torch.linalg.svd(Cov)
        self._V=self._V[:,:self._reduced_dim]
        
    def transform(self,matrix):
        return torch.matmul(matrix-self._mean_matrix[:matrix.shape[0],:],self._V)
    
    def inverse_transform(self,matrix):
        return torch.matmul(matrix,self._V.t())+self._mean_matrix[:matrix.shape[0],:]
    
    
def smoother(mesh,edge_matrix):
    mesh_temp=mesh
    mesh_temp=torch.transpose(mesh_temp,1,2)
    mesh_temp=torch.matmul(mesh_temp,edge_matrix.T)    
    mesh_temp=torch.transpose(mesh_temp,1,2)
    num=torch.sum(edge_matrix,dim=1)
    num=num.reshape(1,-1,1)
    num=num.repeat(mesh_temp.shape[0],1,mesh_temp.shape[2])
    mesh_temp=mesh_temp/num
    return mesh_temp

def k_smoother(k,mesh,edge_matrix):
    mesh_temp=mesh
    for _ in range(k):
        mesh_temp=smoother(mesh_temp,edge_matrix)
    
    return mesh_temp

    
def getinfo(stl,flag):
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
        x = cp.Variable((BATCH_SIZE,len(vertices_face)))
        coeff = cp.Parameter((BATCH_SIZE, len(vertices_face)))
        a=cp.Parameter(BATCH_SIZE)
        x0=cp.Parameter((BATCH_SIZE,len(vertices_face)))
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


def volume_prism_x(M):
    return torch.sum(M[:,:,:,0],dim=2)*(torch.linalg.det(M[:,:,1:,1:]-M[:,:,0,1:].reshape(M.shape[0],M.shape[1],1,-1))/6)

def volume_prism_y(M):
    return torch.sum(M[:,:,:,1],dim=2)*(torch.linalg.det(M[:,:,torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[0],torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[1]]-M[:,:,1,[0,2]].reshape(M.shape[0],M.shape[1],1,-1))/6)

def volume_prism_z(M):
    return torch.sum(M[:,:,:,2],dim=2)*(torch.linalg.det(M[:,:,:2,:2]-M[:,:,2,:2].reshape(M.shape[0],M.shape[1],1,-1))/6)


def volume_2_x(mesh):
    return torch.sum(volume_prism_x(mesh),dim=1)

def volume_2_y(mesh):
    return torch.sum(volume_prism_y(mesh),dim=1)

def volume_2_z(mesh):
    return torch.sum(volume_prism_z(mesh),dim=1)

def get_coeff_z(vertices_face,points_zero,newtriangles_zero):
    tmp=points_zero[:,newtriangles_zero]
    tmp1=torch.linalg.det(tmp[:,:,:2,:2]-tmp[:,:,2,:2].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    tmp2=tmp1@vertices_face.T
    return tmp2

def get_coeff_x(vertices_face,points_zero,newtriangles_zero):
    tmp=points_zero[:,newtriangles_zero]
    tmp1=torch.linalg.det(tmp[:,:,1:,1:]-tmp[:,:,0,1:].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    tmp2=tmp1@vertices_face.T
    return tmp2

def get_coeff_y(vertices_face,points_zero,newtriangles_zero):
    tmp=points_zero[:,newtriangles_zero]
    tmp1=torch.linalg.det(tmp[:,:,torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[0],torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[1]]-tmp[:,:,1,[0,2]].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    tmp2=tmp1@vertices_face.T
    return tmp2



def volume_norm(points,y,points_zero,indices_1,indices_2,newtriangles_zero, vertices_face,cvxpylayer):
    volume_const=volume_2_x(points_zero[newtriangles_zero].unsqueeze(0))
    points_zero_2=points_zero.clone().unsqueeze(0).repeat(len(points),1,1)
    points_zero_2[:,indices_2,0]=y[:,:,0]
    points_zero_2[:,indices_2,2]=y[:,:,1]
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    a=1/3*(volume_const-volume_2_x(points_zero_2[:,newtriangles_zero]))*torch.ones(len(points)).float()    
    coeffz=get_coeff_z(vertices_face, points_zero_2, newtriangles_zero)
    hz=points[:,:,2].reshape(BATCH_SIZE,-1)
    def_z,=cvxpylayer(coeffz,hz,a)
    points[:,:,2]= points[:,:,2]+def_z
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    coeffy=get_coeff_y(vertices_face, points_zero_2, newtriangles_zero)
    hy=points[:,:,1].reshape(BATCH_SIZE,-1)
    def_y,=cvxpylayer(coeffy,hy,a)
    points[:,:,1]= points[:,:,1]+def_y
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    coeffx=get_coeff_x(vertices_face, points_zero_2, newtriangles_zero)
    hx=points[:,:,0].reshape(BATCH_SIZE,-1)
    def_x,=cvxpylayer(coeffx,hx,a)
    points[:,:,0]= points[:,:,0]+def_x
    return points

def volume_norm_single(points,y,points_zero,indices_1,indices_2,newtriangles_zero, vertices_face,cvxpylayer):
    volume_const=volume_2_x(points_zero[newtriangles_zero].unsqueeze(0))
    points_zero_2=points_zero.clone().unsqueeze(0).repeat(len(points),1,1)
    points_zero_2[:,indices_2,0]=y[:,:,0]
    points_zero_2[:,indices_2,2]=y[:,:,1]
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    a=1/3*(volume_const-volume_2_x(points_zero_2[:,newtriangles_zero]))*torch.ones(len(points)).float()    
    coeffz=get_coeff_z(vertices_face, points_zero_2, newtriangles_zero)
    hz=points[:,:,2].reshape(1,-1)
    def_z,=cvxpylayer(coeffz,hz,a)
    points[:,:,2]= points[:,:,2]+def_z
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)

    
    coeffy=get_coeff_y(vertices_face, points_zero_2, newtriangles_zero)
    hy=points[:,:,1].reshape(1,-1)
    def_y,=cvxpylayer(coeffy,hy,a)
    points[:,:,1]= points[:,:,1]+def_y
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)

    coeffx=get_coeff_x(vertices_face, points_zero_2, newtriangles_zero)
    hx=points[:,:,0].reshape(1,-1)
    def_x,=cvxpylayer(coeffx,hx,a)
    points[:,:,0]= points[:,:,0]+def_x
    return points


def relativemmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian'))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian'))))/(np.sqrt(1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian')))+np.sqrt(1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))))
    
def mmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian'))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian'))))

class Data(LightningDataModule):
    def get_size(self):
        temp_interior,temp_boundary,_,_,_,_,_,_,_,_,_,_,_=getinfo(STRING.format(0),False)
        return ((1,temp_interior.shape[0],3),(1,temp_boundary.shape[0],2))
    
    def get_reduced_size(self):
        return (REDUCED_DIMENSION_1,REDUCED_DIMENSION_2)

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        num_samples: int = NUMBER_SAMPLES,
    ):
        super().__init__()
        self.batch_size=batch_size
        self.num_workes=num_workers
        self.num_samples=num_samples
        self.num_workers = num_workers
        self.num_samples=num_samples
        _,_,self.temp_zero,self.oldmesh,self.local_indices_1,self.local_indices_2,self.global_indices_1,self.global_indices_2,self.oldM,self.newtriangles_zero,self.edge_matrix,self.vertices_face,self.cvxpylayer=getinfo(STRING.format(0),True)
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
            self.data_interior[i],tmp,_,_,_,_,_,_,_,_,_,_,_=getinfo(STRING.format(i),False)
            self.data_boundary[i,:,0]=tmp[:,0]
            self.data_boundary[i,:,1]=tmp[:,2]
        self.data=torch.utils.data.TensorDataset(self.data_interior,self.data_boundary)
        self.pca_1=PCA(REDUCED_DIMENSION_1)
        self.pca_2=PCA(REDUCED_DIMENSION_2)
        self.pca_1.fit(self.data_interior.reshape(self.num_samples,-1))
        self.pca_2.fit(self.data_boundary.reshape(self.num_samples,-1))
        self.data_train, self.data_val,self.data_test = random_split(self.data, [NUM_TRAIN_SAMPLES,NUM_VAL_SAMPLES,NUM_TEST_SAMPLES])    

    
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
    

class VolumeNormalizer(nn.Module):
    def __init__(self,temp_zero,newtriangles_zero,vertices_face,cvxpylayer,local_indices_1,local_indices_2):
        super().__init__()
        self.newtriangles_zero=newtriangles_zero
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        self.temp_zero=temp_zero
        self.flag=True
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2

    def forward(self, x,y):
        tmp=self.temp_zero.clone()
        if self.flag:
            y=y.reshape(y.shape[0],-1,2)
            return volume_norm(x,y, tmp,self.local_indices_1,self.local_indices_2 ,self.newtriangles_zero, self.vertices_face, self.cvxpylayer[0])
        else:
            y=y.reshape(1,-1,2)
            return volume_norm_single(x,y,tmp,self.local_indices_1,self.local_indices_2,self.newtriangles_zero, self.vertices_face, self.cvxpylayer[1])

            
class Smoother(nn.Module):
    def __init__(self,edge_matrix,k,temp_zero,local_indices_1,local_indices_2):
        super().__init__()
        self.k=k
        self.edge_matrix=edge_matrix
        self.temp_zero=temp_zero
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
    
    def forward(self,x,y):
        temp=self.temp_zero.clone()
        temp=temp.repeat(x.shape[0],1,1)
        y=y.reshape(y.shape[0],-1,2)
        temp[:,self.local_indices_1,:]=x.reshape(x.shape[0],-1,3)
        temp[:,self.local_indices_2,0]=y[:,:,0]
        temp[:,self.local_indices_2,2]=y[:,:,1] 
        return k_smoother(self.k, temp, self.edge_matrix)[:,torch.diag(self.edge_matrix==0),:]
    


def L2_loss(x_hat, x):
    loss=F.mse_loss(x.reshape(-1), x_hat.reshape(-1), reduction="none")
    loss=loss.mean()
    return loss


class LBR(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=nn.Linear(in_features, out_features)
        self.batch=nn.BatchNorm1d(out_features)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(DROP_PROB)
    
    def forward(self,x):
        return self.dropout(self.relu(self.batch(self.lin(x))))


class Decoder_base(nn.Module):
    def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k):
        super().__init__()
        self.data_shape=data_shape
        self.pca_1=pca_1
        self.pca_2=pca_2
        self.newtriangles_zero=newtriangles_zero
        self.vertices_face=vertices_face
        self.edge_matrix=edge_matrix
        self.cvxpylayer=cvxpylayer
        self.temp_zero=temp_zero
        self.k=k
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.fc_interior_1 = LBR(latent_dim_1, hidden_dim)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_7 = nn.Linear(hidden_dim, int(np.prod(self.data_shape[0])))
        self.fc_boundary_1 = LBR(latent_dim_2, hidden_dim)
        self.fc_boundary_2 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_3 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_4 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_5 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_6 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_7 = nn.Linear(hidden_dim, int(np.prod(self.data_shape[1])))
        self.smoother=Smoother(edge_matrix=self.edge_matrix, k=self.k,temp_zero=self.temp_zero, local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
        self.vol_norm=VolumeNormalizer(temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
        self.relu = nn.ReLU()
        

    def forward(self, z,w):
        result_interior=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z)))))))
        result_boundary=self.fc_boundary_7(self.fc_boundary_6(self.fc_boundary_5(self.fc_boundary_4(self.fc_boundary_3(self.fc_boundary_2(self.fc_boundary_1(z)))))))
        result_interior=self.pca_1.inverse_transform(result_interior)
        result_boundary=self.pca_2.inverse_transform(result_boundary)
        result_interior=self.smoother(result_interior,result_boundary)
        result_interior=result_interior.reshape(result_interior.shape[0],-1,3)
        result=self.vol_norm(result_interior,result_boundary)
        result_interior=result_interior.view(result.size(0),-1)
        return result_interior,result_boundary

    

class Encoder_base(nn.Module):
    def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,pca_1,pca_2):
        super().__init__()
        self.data_shape=data_shape
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.pca_1=pca_1
        self.pca_2=pca_2
        self.fc_interior_1 = LBR(int(np.prod(self.data_shape[0])), hidden_dim)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim)
        self.fc_interior_7 = nn.Linear(hidden_dim, latent_dim_1)
        self.fc_boundary_1 = LBR(int(np.prod(self.data_shape[1])), hidden_dim)
        self.fc_boundary_2 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_3 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_4 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_5 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_6 = LBR(hidden_dim, hidden_dim)
        self.fc_boundary_7 = nn.Linear(hidden_dim, latent_dim_2)

        self.tanh=nn.Tanh()
        self.batch_mu_1=nn.BatchNorm1d(self.latent_dim_1,affine=False,track_running_stats=False)
        self.batch_mu_2=nn.BatchNorm1d(self.latent_dim_2,affine=False,track_running_stats=False)


    def forward(self, x,y):
        x=x.reshape(x.size(0),-1)
        x=self.pca_1.transform(x)
        mu_1=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(x)))))))
        mu_1=self.batch_mu_1(mu_1)
        y=y.reshape(y.size(0),-1)
        y=self.pca_2.transform(y)
        mu_2=self.fc_boundary_7(self.fc_boundary_6(self.fc_boundary_5(self.fc_boundary_4(self.fc_boundary_3(self.fc_boundary_2(self.fc_boundary_1(y)))))))
        mu_2=self.batch_mu_2(mu_2)
        return mu_1,mu_2



class Variance_estimator(nn.Module):
    def __init__(self,latent_dim_1,latent_dim_2,hidden_dim,data_shape):
        super().__init__()
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.var1 = nn.Linear(latent_dim_1, latent_dim_1)
        self.var2 = nn.Linear(latent_dim_2, latent_dim_2)
        self.batch_sigma1=nn.BatchNorm1d(self.latent_dim_1)
        self.batch_sigma2=nn.BatchNorm1d(self.latent_dim_2)

    
    def forward(self,mu_1,mu_2):
        sigma_1=self.batch_sigma1(self.var1(mu_1))
        sigma_1=torch.exp(sigma_1)
        sigma_2=self.batch_sigma2(self.var2(mu_2))
        sigma_2=torch.exp(sigma_2)
        return sigma_1,sigma_2



class Discriminator_base_latent(nn.Module):
    def __init__(self, latent_dim_1, latent_dim_2, hidden_dim,data_shape):
        super().__init__()
        self.data_shape=data_shape
        self.fc1_interior = LBR(latent_dim_1,hidden_dim)
        self.fc2_interior = LBR(hidden_dim,hidden_dim)
        self.fc3_interior = LBR(hidden_dim,hidden_dim)
        self.fc4_interior = LBR(hidden_dim,hidden_dim)
        self.fc5_interior = LBR(hidden_dim,hidden_dim)
        self.fc6_interior = LBR(hidden_dim,hidden_dim)
        self.fc7_interior = nn.Linear(hidden_dim,1)
        self.fc1_boundary = LBR(latent_dim_2,hidden_dim)
        self.fc2_boundary = LBR(hidden_dim,hidden_dim)
        self.fc3_boundary = LBR(hidden_dim,hidden_dim)
        self.fc4_boundary = LBR(hidden_dim,hidden_dim)
        self.fc5_boundary = LBR(hidden_dim,hidden_dim)
        self.fc6_boundary = LBR(hidden_dim,hidden_dim)
        self.fc7_boundary = nn.Linear(hidden_dim,1)
        self.sigmoid=torch.nn.Sigmoid()
        
    def forward(self,x,y):
        x_hat=self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x)))))))
        y_hat=self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(y)))))))
        return x_hat,y_hat



class AE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,pca_1,pca_2):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2)
            
        def forward(self,x,y):
            x_hat,y_hat=self.encoder_base(x,y)
            return x_hat,y_hat
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k)

        def forward(self,x,y):
            return self.decoder_base(x,y)
    
    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k=SMOOTHING_DEGREE,hidden_dim: int= 300,latent_dim_1: int = LATENT_DIM_1,latent_dim_2: int = LATENT_DIM_1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.edge_matrix=edge_matrix
        self.k=k
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim,pca_1=self.pca_1,pca_2=self.pca_2)


    def training_step(self, batch, batch_idx):
        x,y=batch
        z,w=self.encoder(x,y)
        x_hat,y_hat=self.decoder(z,w)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        loss = 0.5*L2_loss(x_hat,x)+0.5*L2_loss(y_hat,y)
        if LOGGING:
            self.log("train_ae_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y=batch
        z,w=self.encoder(x,y)
        x_hat,y_hat=self.decoder(z,w)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        loss = 0.5*L2_loss(x_hat,x)+0.5*L2_loss(y_hat,y)
        if LOGGING:
            self.log("validation_ae_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x,y=batch
        z,w=self.encoder(x,y)
        x_hat,y_hat=self.decoder(z,w)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        loss = 0.5*L2_loss(x_hat,x)+0.5*L2_loss(y_hat,y)
        if LOGGING:
            self.log("test_ae_loss", loss)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim_1)
            mean_2=torch.zeros(1,self.latent_dim_2)

        if var==None:
            var_1=torch.ones(1,self.latent_dim_1)
            var_2=torch.ones(1,self.latent_dim_2)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        temp_interior,temp_boundary=self.decoder(z,w)
        return temp_interior,temp_boundary
    


class AAE(LightningModule):
    
    

    class Encoder(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,pca_1,pca_2):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2)
            
        def forward(self,x,y):
            x_hat,y_hat=self.encoder_base(x,y)
            return x_hat,y_hat
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k)

        def forward(self,x,y):
            return self.decoder_base(x,y)
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape):
            super().__init__()
            self.discriminator=Discriminator_base_latent(latent_dim_1=latent_dim_1, data_shape=data_shape,hidden_dim=hidden_dim, latent_dim_2=latent_dim_2)
             
        def forward(self,x,y):
            x_hat,y_hat=self.discriminator(x,y)
            return x_hat,y_hat



    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k=SMOOTHING_DEGREE,hidden_dim: int= 300,latent_dim_1: int = LATENT_DIM_1,latent_dim_2: int = LATENT_DIM_1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,ae_hyp=0.999,batch_size: int = BATCH_SIZE,**kwargs):
        super().__init__()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.edge_matrix=edge_matrix
        self.k=k
        self.ae_hyp=ae_hyp
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim,pca_1=self.pca_1,pca_2=self.pca_2)
        self.discriminator=self.Discriminator(data_shape=self.data_shape, latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2 ,hidden_dim=self.hidden_dim)
    

    def training_step(self, batch, batch_idx, optimizer_idx ):
        x,y=batch
        z_enc_1,z_enc_2=self.encoder(x,y)
        z_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_2=torch.randn(len(y), self.latent_dim_2).type_as(y)
        x_disc_e,y_disc_e=self.discriminator(z_enc_1,z_enc_2)
        x_disc,y_disc=self.discriminator(z_1,z_2)
        x_hat,y_hat=self.decoder(z_enc_1,z_enc_2)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)


        if optimizer_idx==0:
            ae_loss = 0.5*self.ae_hyp*(L2_loss(x_hat,x)+L2_loss(y_hat,y))-0.5*(1-self.ae_hyp)*(x_disc_e+y_disc_e).mean()
            if LOGGING:
                self.log("train_ae_loss", ae_loss)
            return ae_loss
        
        if optimizer_idx==1:
            real_loss = 0.5*(x_disc_e+y_disc_e).mean()
            fake_loss = -0.5*(x_disc+y_disc).mean()   
            tot_loss= (real_loss+fake_loss)/2
            if LOGGING:
                self.log("train_aee_loss", tot_loss)
            return tot_loss
            
    def validation_step(self, batch, batch_idx):
        x,y=batch
        z_enc_1,z_enc_2=self.encoder(x,y)
        z_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_2=torch.randn(len(y), self.latent_dim_2).type_as(y)
        x_disc_e,y_disc_e=self.discriminator(z_enc_1,z_enc_2)
        x_disc,y_disc=self.discriminator(z_1,z_2)
        x_hat,y_hat=self.decoder(z_enc_1,z_enc_2)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        ae_loss = self.ae_hyp*(L2_loss(x_hat,x)+L2_loss(y_hat,y))
        if LOGGING:
            self.log("val_ae_loss", ae_loss)
        return ae_loss
        
        
    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def test_step(self, batch, batch_idx):
        x,y=batch
        z_enc_1,z_enc_2=self.encoder(x,y)
        z_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_2=torch.randn(len(y), self.latent_dim_2).type_as(y)
        x_disc_e,y_disc_e=self.discriminator(z_enc_1,z_enc_2)
        x_disc,y_disc=self.discriminator(z_1,z_2)
        x_hat,y_hat=self.decoder(z_enc_1,z_enc_2)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        ae_loss = self.ae_hyp*(L2_loss(x_hat,x)+L2_loss(y_hat,y))
        if LOGGING:
            self.log("test_ae_loss", ae_loss)
        return ae_loss


    def configure_optimizers(self):
        optimizer_ae = torch.optim.AdamW(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-3)
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-3)
        return [optimizer_ae,optimizer_disc], []
    
    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim_1)
            mean_2=torch.zeros(1,self.latent_dim_2)

        if var==None:
            var_1=torch.ones(1,self.latent_dim_1)
            var_2=torch.ones(1,self.latent_dim_2)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        temp_interior,temp_boundary=self.decoder(z,w)
        return temp_interior,temp_boundary



class VAE(LightningModule):
    
    class Encoder(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim,data_shape,pca_1,pca_2):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2)
            self.variance_estimator=Variance_estimator(latent_dim_1,latent_dim_2, hidden_dim, data_shape)
            
        def forward(self,x,y):
            mu_1,mu_2=self.encoder_base(x,y)
            sigma_1,sigma_2=self.variance_estimator(mu_1,mu_2)
            return mu_1,mu_2,sigma_1,sigma_2
        
    class Decoder(nn.Module):

        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k)

        def forward(self,x,y):
            return self.decoder_base(x,y)


    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k=SMOOTHING_DEGREE,hidden_dim: int= 300,latent_dim_1: int = LATENT_DIM_1,latent_dim_2: int = LATENT_DIM_1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,ae_hyp=0.999999,**kwargs):
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.log_scale=nn.Parameter(torch.Tensor([0.0]))
        self.edge_matrix=edge_matrix
        self.ae_hyp=ae_hyp
        self.k=k
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.decoder = self.Decoder(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k)
        self.encoder = self.Encoder(data_shape=self.data_shape, latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim,pca_1=self.pca_1,pca_2=self.pca_2)
        
    
    def training_step(self, batch, batch_idx):
        x,y=batch
        mu_1,mu_2,sigma_1,sigma_2 = self.encoder(x,y)
        q_1 = torch.distributions.Normal(mu_1, sigma_1)
        q_2 = torch.distributions.Normal(mu_2, sigma_2)
        standard_1=torch.distributions.Normal(torch.zeros_like(mu_1), torch.ones_like(sigma_1))
        standard_2=torch.distributions.Normal(torch.zeros_like(mu_2), torch.ones_like(sigma_2))
        z1_sampled = q_1.rsample()
        z2_sampled = q_2.rsample()
        x_hat,y_hat = self.decoder(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)
        loss=(0.5*L2_loss(x_hat, x)+0.5*L2_loss(y_hat, y))/(2*torch.exp(self.log_scale))
        reg=0.5*torch.distributions.kl_divergence(q_1, standard_1).mean()+0.5*torch.distributions.kl_divergence(q_2, standard_2).mean()-self.log_scale-torch.log(2*torch.pi)
        if LOGGING:
            self.log("train_vae_loss", loss)
        return loss+reg
    
    
    def get_latent(self,data):
        return self.encoder.forward(data)[0]

    
    def validation_step(self, batch, batch_idx):
        x,y=batch
        mu_1,mu_2,sigma_1,sigma_2 = self.encoder(x,y)
        q_1 = torch.distributions.Normal(mu_1, sigma_1)
        q_2 = torch.distributions.Normal(mu_2, sigma_2)
        standard_1=torch.distributions.Normal(torch.zeros_like(mu_1), torch.ones_like(sigma_1))
        standard_2=torch.distributions.Normal(torch.zeros_like(mu_2), torch.ones_like(sigma_2))
        z1_sampled = q_1.rsample()
        z2_sampled = q_2.rsample()
        x_hat,y_hat = self.decoder(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)

        loss=0.5*L2_loss(x_hat, x)+0.5*L2_loss(y_hat, y)
        reg=0.5*torch.distributions.kl_divergence(q_1, standard_1).mean()+0.5*torch.distributions.kl_divergence(q_2, standard_2).mean()
                
        if LOGGING:
             self.log("val_vae_loss", loss)
        return loss+reg

    def test_step(self, batch, batch_idx):
        x,y=batch
        mu_1,mu_2,sigma_1,sigma_2 = self.encoder(x,y)
        q_1 = torch.distributions.Normal(mu_1, sigma_1)
        q_2 = torch.distributions.Normal(mu_2, sigma_2)
        standard_1=torch.distributions.Normal(torch.zeros_like(mu_1), torch.ones_like(sigma_1))
        standard_2=torch.distributions.Normal(torch.zeros_like(mu_2), torch.ones_like(sigma_2))
        z1_sampled = q_1.rsample()
        z2_sampled = q_2.rsample()
        x_hat,y_hat = self.decoder(z1_sampled,z2_sampled)
        x_hat=x_hat.reshape(x.shape)
        y_hat=y_hat.reshape(y.shape)

        loss=0.5*L2_loss(x_hat, x)+0.5*L2_loss(y_hat, y)
        reg=0.5*torch.distributions.kl_divergence(q_1, standard_1).mean()+0.5*torch.distributions.kl_divergence(q_2, standard_2).mean()

        if LOGGING:
            self.log("test_vae_loss", loss)
        return loss+reg

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}
    def sample_mesh(self,mean=None,var=None):
        if mean==None:
            mean_1=torch.zeros(1,self.latent_dim_1)
            mean_2=torch.zeros(1,self.latent_dim_2)

        if var==None:
            var_1=torch.ones(1,self.latent_dim_1)
            var_2=torch.ones(1,self.latent_dim_2)

        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        temp_interior,temp_boundary=self.decoder(z,w)
        return temp_interior,temp_boundary





class BEGAN(LightningModule):
    
    
    class Generator(nn.Module):

        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2):
            super().__init__()
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k)

        def forward(self,x,y):
            return self.decoder_base(x,y)
        
    class Discriminator(nn.Module):
        def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,local_indices_1,local_indices_2):
            super().__init__()
            self.encoder_base=Encoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,pca_1=pca_1,pca_2=pca_2)
            self.decoder_base=Decoder_base(latent_dim_1=latent_dim_1,latent_dim_2=latent_dim_2, hidden_dim=hidden_dim, data_shape=data_shape,local_indices_1=local_indices_1,local_indices_2=local_indices_2,temp_zero=temp_zero,newtriangles_zero=newtriangles_zero,pca_1=pca_1,pca_2=pca_2,edge_matrix=edge_matrix,vertices_face=vertices_face,cvxpylayer=cvxpylayer,k=k)
             
        def forward(self,x,y):
            x_hat,y_hat=self.decoder_base(*self.encoder_base(x,y))
            return x_hat,y_hat
         
    def __init__(self,data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k=SMOOTHING_DEGREE,hidden_dim: int= 300,latent_dim_1: int = LATENT_DIM_1,latent_dim_2: int = LATENT_DIM_1,lr: float = 0.0002,b1: float = 0.5,b2: float = 0.999,batch_size: int = BATCH_SIZE,ae_hyp=0.999999,**kwargs):
        super().__init__()
        super().__init__()
        #self.save_hyperparameters()
        self.temp_zero=temp_zero
        self.newtriangles_zero=newtriangles_zero
        self.pca_1=pca_1
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.pca_2=pca_2
        self.edge_matrix=edge_matrix
        self.ae_hyp=ae_hyp
        self.k=k
        self.latent_dim_1=latent_dim_1
        self.latent_dim_2=latent_dim_2
        self.hidden_dim=hidden_dim
        self.vertices_face=vertices_face
        self.cvxpylayer=cvxpylayer
        # networks
        self.data_shape = data_shape
        self.discriminator = self.Discriminator(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k)
        self.generator = self.Generator(latent_dim_1=self.latent_dim_1,latent_dim_2=self.latent_dim_2,hidden_dim=self.hidden_dim ,data_shape=self.data_shape,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2,temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,pca_1=self.pca_1,pca_2=self.pca_2,edge_matrix=self.edge_matrix,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,k=self.k)


        
    def forward(self, x):
        x_hat=self.discriminator(x)
        return x_hat.reshape(x.shape)
    
    def disc_loss(self, x,y):
        x_hat,y_hat=self.discriminator(x,y)
        loss=0.5*F.mse_loss(x, x_hat.reshape(x.shape), reduction="none").mean()+0.5*F.mse_loss(y, y_hat.reshape(y.shape), reduction="none").mean()
        return loss
    
    def training_step(self, batch, batch_idx, optimizer_idx ):
        x,y=batch
        z_p_1=torch.randn(len(x), self.latent_dim_1).type_as(x)
        z_p_2=torch.randn(len(y), self.latent_dim_2).type_as(y)

        z_d_1,z_d_2=self.discriminator.encoder_base(x,y)
        z_d_1=z_d_1.reshape(len(x),self.latent_dim_1)
        z_d_2=z_d_2.reshape(len(y),self.latent_dim_2)
        

        batch_p_1,batch_p_2=self.generator(z_p_1,z_p_2)
        batch_d_1,batch_d_2=self.generator(z_d_1,z_d_2)
        
        gamma=0.5
        k=0
        lambda_k = 0.001
        
        if optimizer_idx==0:
            loss=self.disc_loss(batch_p_1,batch_p_2)
            if LOGGING:
                self.log("train_generator_loss", loss)
            return loss
        

        if optimizer_idx==1:    
            loss_disc=self.disc_loss(x,y)-k*self.disc_loss(batch_d_1,batch_d_2)
            loss_gen=self.disc_loss(batch_p_1,batch_p_2)
            if LOGGING:
                self.log("train_discriminagtor_loss", loss_disc)
            diff = torch.mean(gamma * self.disc_loss(*batch) - loss_gen)
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)
            return loss_disc
        
    def validation_step(self, batch, batch_idx):
        x,y=batch
        if LOGGING:
            self.log("val_began_loss", self.disc_loss(x,y))
        return self.disc_loss(x,y)

        
    def test_step(self, batch, batch_idx):
        x,y=batch
        if LOGGING:
            self.log("test_began_loss", self.disc_loss(x,y))
        return self.disc_loss(x,y)
        

    def configure_optimizers(self): #0.039,.0.2470, 0.2747
        optimizer_gen = torch.optim.AdamW(self.generator.parameters(), lr=0.02) #0.02
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=0.05) #0.050
        return [optimizer_gen,optimizer_disc], []

    def sample_mesh(self):
        mean_1=torch.zeros(1,self.latent_dim_1)
        mean_2=torch.zeros(1,self.latent_dim_2)
        var_1=torch.ones(1,self.latent_dim_1)
        var_2=torch.ones(1,self.latent_dim_2)
        z = torch.sqrt(var_1)*torch.randn(1,self.latent_dim_1)+mean_1
        w = torch.sqrt(var_2)*torch.randn(1,self.latent_dim_2)+mean_2
        temp_interior,temp_boundary=self.generator(z,w)
        return temp_interior,temp_boundary



print("Loading data")
data=Data()
oldmesh=data.oldmesh.clone().numpy()
area_real=np.zeros(NUMBER_SAMPLES)
curvature_gaussian_real=np.zeros([NUMBER_SAMPLES,np.max(data.newtriangles_zero)+1])
#curvature_mean_real=np.zeros([NUMBER_SAMPLES,np.max(data.newtriangles_zero)+1])
curvature_total_real=np.zeros(NUMBER_SAMPLES)
temp_zero=data.temp_zero.clone().numpy()
    
print("Getting properties of the data")
for i in range(NUMBER_SAMPLES):
    temp_zero=data.temp_zero.clone().numpy()
    temp_zero[data.local_indices_1]=data.data_interior[i].reshape(data.get_size()[0][1],data.get_size()[0][2]).detach().numpy()
    temp_zero[data.local_indices_2,0]=data.data_boundary[i].reshape(data.get_size()[1][1],data.get_size()[1][2]).detach().numpy()[:,0]
    temp_zero[data.local_indices_2,2]=data.data_boundary[i].reshape(data.get_size()[1][1],data.get_size()[1][2]).detach().numpy()[:,1]
    mesh_object=trimesh.base.Trimesh(temp_zero,data.newtriangles_zero,process=False)
    curvature_gaussian_real[i]=trimesh.curvature.discrete_gaussian_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    #curvature_mean_real[i]=trimesh.curvature.discrete_mean_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_total_real[i]=np.sum(curvature_gaussian_real[i])
    area_real[i]=mesh_object.area
curvature_total_real=curvature_total_real.reshape(-1,1)
area_real=area_real.reshape(-1,1)


d=dict
d={
  #AE: "AE",
  #AAE: "AAE",
  #VAE: "VAE", ##ALZARE AE_HYP
  BEGAN: "BEGAN",
}



for wrapper, name in d.items():
    torch.manual_seed(100)
    np.random.seed(100)
    if not LOGGING:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
   
    if LOGGING:
        if AVAIL_GPUS:
            trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,log_every_n_steps=1,track_grad_norm=2,
                              gradient_clip_val=0.1
                              )
        else:
            trainer=Trainer(max_epochs=MAX_EPOCHS,log_every_n_steps=1,track_grad_norm=2,
                            gradient_clip_val=0.1
                            )   
    else:
        if AVAIL_GPUS:
            trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1,track_grad_norm=2,
                              gradient_clip_val=0.1
                              )
        else:
            trainer=Trainer(max_epochs=MAX_EPOCHS,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1,track_grad_norm=2,
                            gradient_clip_val=0.1
                            )
    
    model=wrapper(data.get_reduced_size(),data.temp_zero,data.local_indices_1,data.local_indices_2,data.newtriangles_zero,data.pca_1,data.pca_2,data.edge_matrix,data.vertices_face,data.cvxpylayer)
    print("Training of "+name+ "Has started")
    trainer.fit(model, data)
    trainer.validate(model,data)
    trainer.test(model,data)
    

    model.eval()
        if hasattr(model, 'decoder'):
            model.decoder.decoder_base.vol_norm.flag=False
        else:
            model.generator.decoder_base.vol_norm.flag=False
    temp_interior,temp_boundary = model.sample_mesh()
    
    error=0
    temparr=torch.zeros(NUMBER_SAMPLES,*tuple(temp_interior.shape))
    vol=torch.zeros(NUMBER_SAMPLES)
    curvature_gaussian_sampled=np.zeros([NUMBER_SAMPLES,np.max(data.newtriangles_zero)+1])
    curvature_total_sampled=np.zeros(NUMBER_SAMPLES)
    area_sampled=np.zeros(NUMBER_SAMPLES)
    print("Sampling of "+name+ " has started")

    for i in range(NUMBER_SAMPLES):
        temp_zero=data.temp_zero.clone().numpy()
        temp_interior,temp_boundary = model.sample_mesh()
        temp_interior=temp_interior.detach()
        temp_boundary=temp_boundary.detach()
        oldmesh=data.oldmesh.clone()
        temparr[i]=temp_interior
        oldmesh[data.global_indices_1]=temp_interior.reshape(-1,3)
        oldmesh[data.global_indices_2,0]=temp_boundary.reshape(-1,2)[:,0]
        oldmesh[data.global_indices_2,2]=temp_boundary.reshape(-1,2)[:,1]
        meshio.write_points_cells(name+'_{}.stl'.format(i),oldmesh,[("triangle", data.oldM)])
        true_interior=data.data_interior.reshape(data.num_samples,-1)
        true_boundary=data.data_boundary.reshape(data.num_samples,-1)
        temp_interior=temp_interior.reshape(1,-1)
        temp_boundary=temp_boundary.reshape(1,-1)
        error=error+0.5*torch.min(torch.norm(temp_interior-true_interior,dim=1))/torch.norm(temp_interior)/NUMBER_SAMPLES+0.5*torch.min(torch.norm(temp_boundary-true_boundary,dim=1))/torch.norm(temp_boundary)/NUMBER_SAMPLES
        vol[i]=volume_2_z(oldmesh[data.oldM].unsqueeze(0))
        temp_zero[data.local_indices_1]=temp_interior.reshape(data.get_size()[0][1],data.get_size()[0][2]).numpy()
        temp_zero[data.local_indices_2,0]=temp_boundary.reshape(data.get_size()[1][1],data.get_size()[1][2]).numpy()[:,0]
        temp_zero[data.local_indices_2,2]=temp_boundary.reshape(data.get_size()[1][1],data.get_size()[1][2]).numpy()[:,1]
        mesh_object=trimesh.base.Trimesh(temp_zero,data.newtriangles_zero,process=False)
        curvature_gaussian_sampled[i]=trimesh.curvature.discrete_gaussian_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
        #curvature_mean_sampled[i]=trimesh.curvature.discrete_mean_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
        curvature_total_sampled[i]=np.sum(curvature_gaussian_sampled[i])
        area_sampled[i]=mesh_object.area
    curvature_total_sampled=curvature_total_sampled.reshape(-1,1)
    area_sampled=area_sampled.reshape(-1,1)

    variance=torch.sum(torch.var(temparr,dim=0))
    variance_vol=torch.sum(torch.var(vol,dim=0))
    print("MMD Total Curvature distance of", name, "is", relativemmd(curvature_total_real,curvature_total_sampled))
    print("MMD Area distance of", name, "is", relativemmd(area_real,area_sampled))
    print("Percentage error ",  name, " is", error.item())
    print("Variance from prior of ", name, "is", variance.item())
    print("MMD Gaussian Curvature distance of", name, "is", relativemmd(curvature_gaussian_real,curvature_gaussian_sampled))
    print("MMD Id distance of", name, "is", relativemmd(temparr.reshape(NUMBER_SAMPLES,-1).detach(),data.data_interior[:].reshape(NUMBER_SAMPLES,-1).detach()))
    voltrue=volume_2_z(data.oldmesh[data.oldM].unsqueeze(0))
    print("Mean of Volume Abs Percentage error is", torch.mean(torch.abs(vol-voltrue)/voltrue))
    print("Variance of Volume Abs Percentage error is", torch.var(torch.abs(vol-voltrue)/voltrue))


    fig1,ax1=plt.subplots()
    ax1.set_title("Area of "+name)
    _=ax1.plot(np.sort(area_real.reshape(-1)), np.linspace(0, 1, len(area_real)),'r',label='true')
    _=ax1.plot(np.sort(area_sampled.reshape(-1)), np.linspace(0, 1, len(area_sampled)),'g',label='sampled')
    ax1.legend()
    fig1.savefig("Area_cdf_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("Area of "+name)
    _=ax2.hist([area_real.reshape(-1),area_sampled.reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("Area_hist_"+name+".png")
    fig3,ax3=plt.subplots()
    ax3.set_title("TC of "+name)
    _=ax3.plot(np.sort(curvature_total_real.reshape(-1)), np.linspace(0, 1, len(curvature_total_real)),color='r',label='true'),
    _=ax3.plot(np.sort(curvature_total_sampled.reshape(-1)), np.linspace(0, 1, len(curvature_total_sampled)),'g',label='sampled')
    ax3.legend()
    fig3.savefig("TC_cdf_"+name+".png")
    fig4,ax4=plt.subplots()
    ax4.set_title("TC of "+name)
    _=ax4.hist([curvature_total_real.reshape(-1),curvature_total_sampled.reshape(-1)],50,label=['real','sampled'])
    ax4.legend()
    fig4.savefig("TC_hist_"+name+".png")
        

temparr=torch.zeros(NUMBER_SAMPLES,*data.data_interior[0].shape)
for i in range(NUMBER_SAMPLES):
    temparr[i]=data.data_interior[i]    
    

variance=torch.sum(torch.var(temparr,dim=0))
print("Variance of data is", variance.item())


####TEST####    
#points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,edge_matrix=getinfo("test_VAE_intPCA.stl",True)
#temp=k_smoother(10,points_zero.reshape(1,points_zero.shape[0],-1),edge_matrix)
#points_old[points_old[:,2]>=0]=temp
#meshio.write_points_cells('test.stl',points_old,[("triangle", data.oldM)])
