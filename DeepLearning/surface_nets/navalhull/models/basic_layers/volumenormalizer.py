import torch
from torch import nn

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
    a=1/3*(volume_const-volume_2_x(points_zero_2[:,newtriangles_zero]))*torch.ones(len(points),device=points_zero.device).float()    
    coeffz=get_coeff_z(vertices_face, points_zero_2, newtriangles_zero)
    hz=points[:,:,2].reshape(points.shape[0],-1)
    def_z,=cvxpylayer(coeffz,hz,a)
    points[:,:,2]= points[:,:,2]+def_z
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    coeffy=get_coeff_y(vertices_face, points_zero_2, newtriangles_zero)
    hy=points[:,:,1].reshape(points.shape[0],-1)
    def_y,=cvxpylayer(coeffy,hy,a)
    points[:,:,1]= points[:,:,1]+def_y
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    coeffx=get_coeff_x(vertices_face, points_zero_2, newtriangles_zero)
    hx=points[:,:,0].reshape(points.shape[0],-1)
    def_x,=cvxpylayer(coeffx,hx,a)
    points[:,:,0]= points[:,:,0]+def_x
    return points

def volume_norm_single(points,y,points_zero,indices_1,indices_2,newtriangles_zero, vertices_face,cvxpylayer):
    volume_const=volume_2_x(points_zero[newtriangles_zero].unsqueeze(0))
    points_zero_2=points_zero.clone().unsqueeze(0).repeat(len(points),1,1)
    points_zero_2[:,indices_2,0]=y[:,:,0]
    points_zero_2[:,indices_2,2]=y[:,:,1]
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    a=1/3*(volume_const-volume_2_x(points_zero_2[:,newtriangles_zero]))*torch.ones(len(points),device=points_zero.device)
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

    
