import torch
from torch import nn


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