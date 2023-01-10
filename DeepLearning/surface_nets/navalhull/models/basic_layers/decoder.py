from models.basic_layers.lbr import LBR
from models.basic_layers.smoother import Smoother
from models.basic_layers.volumenormalizer import VolumeNormalizer

import numpy as np
from torch import nn

class Decoder_base(nn.Module):
    def __init__(self, latent_dim_1,latent_dim_2, hidden_dim, data_shape,temp_zero,local_indices_1,local_indices_2,newtriangles_zero,pca_1,pca_2,edge_matrix,vertices_face,cvxpylayer,k,drop_prob):
        super().__init__()
        self.data_shape=data_shape
        self.pca_1=pca_1
        self.pca_2=pca_2
        self.drop_prob=drop_prob
        self.newtriangles_zero=newtriangles_zero
        self.vertices_face=vertices_face
        self.edge_matrix=edge_matrix
        self.cvxpylayer=cvxpylayer
        self.temp_zero=temp_zero
        self.k=k
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2
        self.fc_interior_1 = LBR(latent_dim_1, hidden_dim,drop_prob)
        self.fc_interior_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_interior_7 = nn.Linear(hidden_dim, int(np.prod(self.data_shape[0])))
        self.fc_boundary_1 = LBR(latent_dim_2, hidden_dim,drop_prob)
        self.fc_boundary_2 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_3 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_4 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_5 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_6 = LBR(hidden_dim, hidden_dim,drop_prob)
        self.fc_boundary_7 = nn.Linear(hidden_dim, int(np.prod(self.data_shape[1])))
        self.smoother=Smoother(edge_matrix=self.edge_matrix, k=self.k,temp_zero=self.temp_zero, local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
        self.vol_norm=VolumeNormalizer(temp_zero=self.temp_zero,newtriangles_zero=self.newtriangles_zero,vertices_face=self.vertices_face,cvxpylayer=self.cvxpylayer,local_indices_1=self.local_indices_1,local_indices_2=self.local_indices_2)
        self.relu = nn.ReLU()
        

    def forward(self, z,w):
        result_interior=self.fc_interior_7(self.fc_interior_6(self.fc_interior_5(self.fc_interior_4(self.fc_interior_3(self.fc_interior_2(self.fc_interior_1(z)))))))
        result_boundary=self.fc_boundary_7(self.fc_boundary_6(self.fc_boundary_5(self.fc_boundary_4(self.fc_boundary_3(self.fc_boundary_2(self.fc_boundary_1(w)))))))
        result_interior=self.pca_1.inverse_transform(result_interior)
        result_boundary=self.pca_2.inverse_transform(result_boundary)
        result_interior=self.smoother(result_interior,result_boundary)
        result_interior=result_interior.reshape(result_interior.shape[0],-1,3)
        result=self.vol_norm(result_interior,result_boundary)
        result_interior=result_interior.view(result.size(0),-1)
        return result_interior,result_boundary
 
