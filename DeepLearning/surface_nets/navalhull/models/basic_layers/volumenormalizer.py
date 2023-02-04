import torch
from torch import nn
from models.basic_layers.qpth.qp import QPFunction

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
    qp=QPFunction(1e-8,check_Q_spd=False)
    Q=torch.eye(points.shape[1],device=points.device)
    G=-Q
    p=torch.zeros(points.shape[1],device=points.device)
    points=points.reshape(len(points),-1,3)
    volume_const=volume_2_x(points_zero[newtriangles_zero].unsqueeze(0))
    points_zero_2=points_zero.clone().unsqueeze(0).repeat(len(points),1,1)
    points_zero_2[:,indices_2,0]=y[:,:,0]
    points_zero_2[:,indices_2,2]=y[:,:,1]
    points_zero_2[:,indices_1,:]=points.reshape(len(points),-1,3)
    a=(1/3*(volume_const-volume_2_x(points_zero_2[:,newtriangles_zero]))*torch.ones(len(points),device=points_zero.device).float()).reshape(-1,1)  
    coeffz=get_coeff_z(vertices_face, points_zero_2, newtriangles_zero).unsqueeze(1)
    hz=points[:,:,2].reshape(points.shape[0],-1)
    print(torch.linalg.det(Q))
    def_z=qp(Q,p,G,hz,coeffz,a)
    points_zero_2[:,indices_1,2]=points_zero_2[:,indices_1,2]+def_z
    coeffy=get_coeff_y(vertices_face, points_zero_2, newtriangles_zero).unsqueeze(1)
    hy=points[:,:,1].reshape(points.shape[0],-1)
    def_y=qp(Q,p,G,hy,coeffy,a)
    points_zero_2[:,indices_1,2]=points_zero_2[:,indices_1,1]+def_y
    coeffx=get_coeff_x(vertices_face, points_zero_2, newtriangles_zero).unsqueeze(1)
    hx=points[:,:,0].reshape(points.shape[0],-1)
    def_x=qp(Q,p,G,hx,coeffx,a)
    return points+torch.concat((def_x.unsqueeze(2),def_y.unsqueeze(2),def_z.unsqueeze(2)),axis=2)

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
        y=y.reshape(y.shape[0],-1,2)
        return volume_norm(x,y, tmp,self.local_indices_1,self.local_indices_2 ,self.newtriangles_zero, self.vertices_face, self.cvxpylayer[0])

    
