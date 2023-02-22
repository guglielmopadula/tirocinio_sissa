import torch
from torch import nn



def volume_prism_x(M):
    return torch.sum(M[:,:,:,0],dim=2)*(torch.linalg.det(M[:,:,1:,1:]-M[:,:,0,1:].reshape(M.shape[0],M.shape[1],1,-1))/6)

def volume_prism_y(M):
    return torch.sum(M[:,:,:,1],dim=2)*(torch.linalg.det(M[:,:,torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[0],torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])],indexing="ij")[1]]-M[:,:,1,[0,2]].reshape(M.shape[0],M.shape[1],1,-1))/6)

def volume_prism_z(M):
    return torch.sum(M[:,:,:,2],dim=2)*(torch.linalg.det(M[:,:,:2,:2]-M[:,:,2,:2].reshape(M.shape[0],M.shape[1],1,-1))/6)


def volume_2_x(mesh):
    return torch.sum(volume_prism_x(mesh),dim=1)

def volume_2_y(mesh):
    return torch.sum(volume_prism_y(mesh),dim=1)

def volume_2_z(mesh):
    return torch.sum(volume_prism_z(mesh),dim=1)

def get_coeff_z(vertices_face_xy,points_zero,newtriangles_zero):
    tmp=points_zero[:,newtriangles_zero]
    tmp1=torch.linalg.det(tmp[:,:,:2,:2]-tmp[:,:,2,:2].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    tmp2=tmp1@vertices_face_xy.T
    return tmp2

def get_coeff_x(vertices_face_xy,points_zero,newtriangles_zero):
    tmp=points_zero[:,newtriangles_zero]
    tmp1=torch.linalg.det(tmp[:,:,1:,1:]-tmp[:,:,0,1:].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    tmp2=tmp1@vertices_face_xy.T
    return tmp2

def get_coeff_y(vertices_face_x,points_zero,newtriangles_zero):
    tmp=points_zero[:,newtriangles_zero]
    tmp1=torch.linalg.det(tmp[:,:,torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[0],torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])],indexing="ij")[1]]-tmp[:,:,1,[0,2]].reshape(tmp.shape[0],tmp.shape[1],1,-1))/6
    tmp2=tmp1@vertices_face_x.T
    return tmp2



def volume_norm(x,y,points_zero,indices_1,indices_2,newtriangles_zero, vertices_face_x,vertices_face_xy):
    x=x.reshape(len(x),-1,3)
    volume_const=volume_2_y(points_zero[newtriangles_zero].unsqueeze(0))
    points_zero_2=points_zero.clone().unsqueeze(0).repeat(len(x),1,1)
    points_zero_2[:,indices_2,0]=y[:,:,0]
    points_zero_2[:,indices_2,2]=y[:,:,1]
    points_zero_2[:,indices_1,:]=x.reshape(len(x),-1,3)
    a=1/2*((volume_const-volume_2_y(points_zero_2[:,newtriangles_zero]))*torch.ones(len(x),device=points_zero.device).float()).reshape(-1,1,1)  
    coeffy=get_coeff_y(vertices_face_x, points_zero_2, newtriangles_zero).unsqueeze(1)
    def_y=torch.bmm(torch.transpose(coeffy,1,2),torch.linalg.solve((torch.bmm(coeffy,torch.transpose(coeffy,1,2))),a)).reshape(x.shape[0],-1)
    points_zero_2[:,indices_1,1]=points_zero_2[:,indices_1,1]+def_y
    coeffz=get_coeff_z(vertices_face_xy, points_zero_2, newtriangles_zero).unsqueeze(1)
    def_z=torch.bmm(torch.transpose(coeffz,1,2),torch.linalg.solve((torch.bmm(coeffz,torch.transpose(coeffz,1,2))),a)).reshape(x.shape[0],-1)
    points_zero_2[:,indices_1+indices_2,2]=points_zero_2[:,indices_1+indices_2,2]+def_z
    #coeffx=get_coeff_x(vertices_face_xy, points_zero_2, newtriangles_zero).unsqueeze(1)
    #def_x=torch.bmm(torch.bmm(torch.transpose(coeffx,1,2),torch.inverse(torch.bmm(coeffx,torch.transpose(coeffx,1,2)))),a).reshape(x.shape[0],-1)
    #points_zero_2[:,indices_1+indices_2,0]=points_zero_2[:,indices_1+indices_2,0]+def_x
    grid=torch.meshgrid([torch.arange(x.shape[0]),torch.tensor(indices_2),torch.tensor([0,2])],indexing="ij")
    return points_zero_2[:,indices_1,:],points_zero_2[grid]

class VolumeNormalizer(nn.Module):
    def __init__(self,temp_zero,newtriangles_zero,vertices_face_x,vertices_face_xy,local_indices_1,local_indices_2):
        super().__init__()
        self.newtriangles_zero=newtriangles_zero
        self.vertices_face_x=vertices_face_x
        self.vertices_face_xy=vertices_face_xy
        self.temp_zero=temp_zero
        self.flag=True
        self.local_indices_1=local_indices_1
        self.local_indices_2=local_indices_2

    def forward(self, x,y):
        tmp=self.temp_zero.clone()
        y=y.reshape(y.shape[0],-1,2)
        return volume_norm(x,y, tmp,self.local_indices_1,self.local_indices_2 ,self.newtriangles_zero, self.vertices_face_x, self.vertices_face_xy)

    
