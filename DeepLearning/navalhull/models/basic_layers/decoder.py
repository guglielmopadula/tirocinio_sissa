from models.basic_layers.lbr import LBR
from models.basic_layers.smoother import Smoother
from models.basic_layers.volumenormalizer import VolumeNormalizer

import numpy as np
import torch
from torch import nn

class Decoder_base(nn.Module):
    def __init__(self, latent_dim, hidden_dim,reduced_data_shape, data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca,edge_matrix,vertices_face_x,vertices_face_xy,k,drop_prob):
        super().__init__()
        self.data_shape=data_shape
        self.reduced_data_shape=reduced_data_shape
        self.pca=pca
        self.drop_prob=drop_prob
        self.newtriangles_zero=newtriangles_zero
        self.vertices_face_x=vertices_face_x
        self.edge_matrix=edge_matrix
        self.vertices_face_xy=vertices_face_xy
        self.temp_zero=temp_zero
        self.k=k
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.fc_interior_1 = LBR(latent_dim, hidden_dim,drop_prob)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_7 = nn.Linear(hidden_dim,self.reduced_data_shape)
        self.smoother=Smoother(edge_matrix=self.edge_matrix, k=self.k,temp_zero=self.temp_zero, local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
        self.vol_norm=VolumeNormalizer(temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,vertices_face_x=self.vertices_face_x,vertices_face_xy=self.vertices_face_xy,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
        self.relu = nn.ReLU()
        

    def forward(self, z):
        tmp=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z)))))))
        x=self.pca.inverse_transform(tmp)
        result_interior=x[:,:np.prod(self.data_shape[0])]
        result_boundary=x[:,np.prod(self.data_shape[0]):]
        #result_interior=self.smoother(result_interior,result_boundary)
        result_interior,result_boundary=self.vol_norm(result_interior,result_boundary)
        result_interior=result_interior.reshape(result_interior.shape[0],-1)
        result_boundary=result_boundary.reshape(result_interior.shape[0],-1)
        return torch.concat((result_interior,result_boundary),axis=1)
 
