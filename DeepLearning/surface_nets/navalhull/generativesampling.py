#!/usr/bin/env python3  
# -*- coding: utf-8 -*-
"""
@author: cyberguli
"""
from datawrapper.data import Data
import os
from models.AE import AE
from models.AAE import AAE
from models.VAE import VAE
from tqdm import trange
from models.BEGAN import BEGAN
import trimesh
import matplotlib.pyplot as plt
import torch
#import numpy as np
import meshio
from models.losses.losses import relativemmd
cuda_avail=True if torch.cuda.is_available() else False
torch.set_default_dtype(torch.float64)
cuda_avail=False
from pytorch_lightning.plugins.environments import SLURMEnvironment
def volume_prism_y(M):
    return torch.sum(M[:,:,:,1],dim=2)*(torch.linalg.det(M[:,:,torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])])[0],torch.meshgrid([torch.tensor([0,2]),torch.tensor([0,2])],indexing="ij")[1]]-M[:,:,1,[0,2]].reshape(M.shape[0],M.shape[1],1,-1))/6)


def volume_2_y(mesh):
    return torch.sum(volume_prism_y(mesh),dim=1)


class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


def myarccos(x):
    return torch.arccos(torch.minimum(torch.maximum(-torch.ones(x.shape),x),torch.ones(x.shape)))


def area(vertices, triangles):
    triangles=torch.tensor(triangles)
    v1 = vertices[triangles[:,0]]
    v2 = vertices[triangles[:,1]]
    v3 = vertices[triangles[:,2]]
    a = torch.linalg.norm(torch.cross(v2 - v1, v3 - v1), axis=1) / 2
    return torch.sum(a)

def gaussian_curvature(vertices, triangles):
    triangles=torch.tensor(triangles)
    ab=vertices[triangles[:,1]] - vertices[triangles[:,0]]
    ac=vertices[triangles[:,2]] - vertices[triangles[:,0]]
    bc=vertices[triangles[:,2]] - vertices[triangles[:,1]]
    angleA=myarccos(torch.einsum('ij,ij->i',ab,ac)/(torch.linalg.norm(ab,axis=1)*torch.linalg.norm(ac,axis=1)))
    angleB=myarccos(torch.einsum('ij,ij->i',-ab,bc)/(torch.linalg.norm(bc,axis=1)*torch.linalg.norm(ab,axis=1)))
    angleC=myarccos(torch.einsum('ij,ij->i',-bc,-ac)/(torch.linalg.norm(bc,axis=1)*torch.linalg.norm(ac,axis=1)))
    angles=torch.concatenate((angleA.reshape(-1,1),angleB.reshape(-1,1),angleC.reshape(-1,1)),axis=1)
    Area=torch.linalg.norm(torch.cross(ab,ac),axis=1)/2
    Area=Area.reshape(-1,1).repeat(3,1)
    triangles=triangles.flatten()
    Area=Area.flatten()
    angles=angles.flatten()
    gaussian_curvatures=(2*torch.pi-torch.bincount(triangles,angles))/torch.bincount(triangles,(1/3)*Area)
    return gaussian_curvatures

NUM_WORKERS = int(os.cpu_count() / 2)

NUM_TRAIN_SAMPLES=5
NUM_TEST_SAMPLES=0
NUMBER_SAMPLES=NUM_TEST_SAMPLES+NUM_TRAIN_SAMPLES



print("Loading data")

data=torch.load("./data_objects/data.pt", map_location="cpu")

d={
  AE: "AE",
  #AAE: "AAE",
  #VAE: "VAE", 
  #BEGAN: "BEGAN",
}

print("Getting properties of the data")
oldmesh=data.oldmesh.clone()
area_real=torch.zeros(NUMBER_SAMPLES)
curvature_gaussian_real=torch.zeros([NUMBER_SAMPLES,torch.max(torch.tensor(data.newtriangles_zero)).item()+1])
curvature_total_real=torch.zeros(NUMBER_SAMPLES)
datanumpy=data.data[:].detach().cpu()
datanumpy=datanumpy[:NUMBER_SAMPLES,:]
true_interior=datanumpy[:,:torch.prod(torch.tensor(data.get_size()[0]))].reshape(NUMBER_SAMPLES,-1)
true_boundary=datanumpy[:,torch.prod(torch.tensor(data.get_size()[0])):].reshape(NUMBER_SAMPLES,-1)
moment_tensor_data=torch.zeros((NUMBER_SAMPLES,3,3))
vol_real=torch.zeros(NUMBER_SAMPLES)


for i in trange(NUMBER_SAMPLES):
    temp_zero=data.temp_zero.clone()
    temp_zero[data.local_indices_1]=true_interior[i].reshape(-1,3)
    temp_zero[data.local_indices_2,0]=true_boundary[i].reshape(-1,2)[:,0]
    temp_zero[data.local_indices_2,2]=true_boundary[i].reshape(-1,2)[:,1]
    curvature_gaussian_real[i]=gaussian_curvature(temp_zero,data.newtriangles_zero)
    curvature_total_real[i]=torch.sum(curvature_gaussian_real[i])
    area_real[i]=area(temp_zero,data.newtriangles_zero)
    vol_real[i]=volume_2_y(temp_zero[data.newtriangles_zero].reshape(1,*temp_zero[data.newtriangles_zero].shape)).reshape(-1)
    for j in range(3):
        for k in range(3):
            moment_tensor_data[i,j,k]=torch.mean(temp_zero.reshape(-1,3)[:,j]*temp_zero.reshape(-1,3)[:,k],axis=0)


curvature_total_real=curvature_total_real.reshape(-1,1)
area_real=area_real.reshape(-1,1)
vol_real=vol_real.reshape(-1,1)


for wrapper, name in d.items():
    torch.manual_seed(100)
    if cuda_avail==False:
        model=torch.load("./saved_models/"+name+".pt",map_location=torch.device('cpu'))
    else:
        model=torch.load("./saved_models/"+name+".pt")

    model.eval()
    if hasattr(model, 'decoder'):
        model.decoder.decoder_base.vol_norm.flag=False

    else:
        model.generator.decoder_base.vol_norm.flag=False
    error=0
    tmp = model.sample_mesh()
    temparr=torch.zeros((NUMBER_SAMPLES,*tuple(tmp.shape)))
    vol_sampled=torch.zeros(NUMBER_SAMPLES)
    curvature_gaussian_sampled=torch.zeros([NUMBER_SAMPLES,torch.max(torch.tensor(data.newtriangles_zero)).item()+1])
    moment_tensor_sampled=torch.zeros((NUMBER_SAMPLES,3,3))
    area_sampled=torch.zeros(NUMBER_SAMPLES)
    print("Sampling of "+name+ " has started")
    oldmesh=data.oldmesh.clone().cpu()

    for i in trange(NUMBER_SAMPLES):
        temp_zero=data.temp_zero.clone().cpu()
        tmp = model.sample_mesh()
        tmp=tmp.reshape(-1).detach().cpu()
        temp_interior=tmp[:torch.prod(torch.tensor(data.get_size()[0]))]
        temp_boundary=tmp[torch.prod(torch.tensor(data.get_size()[0])):]
        temparr[i]=tmp
        oldmesh[data.global_indices_1]=temp_interior.reshape(-1,3)
        oldmesh[data.global_indices_2,0]=temp_boundary.reshape(-1,2)[:,0]
        oldmesh[data.global_indices_2,2]=temp_boundary.reshape(-1,2)[:,1]
        meshio.write_points_cells("./inference_objects/"+name+'_sampled_{}.stl'.format(i),oldmesh,[("triangle", data.oldM)])
        error=error+torch.min(torch.linalg.norm(tmp-datanumpy,axis=1))/torch.linalg.norm(datanumpy)/NUMBER_SAMPLES
        temp_zero[data.local_indices_1]=temp_interior.reshape(-1,3)
        temp_zero[data.local_indices_2,0]=temp_boundary.reshape(-1,2)[:,0]
        temp_zero[data.local_indices_2,2]=temp_boundary.reshape(-1,2)[:,1]
        vol_sampled[i]=volume_2_y(temp_zero[data.newtriangles_zero].reshape(1,*temp_zero[data.newtriangles_zero].shape)).reshape(-1)
        curvature_gaussian_sampled[i]=gaussian_curvature(temp_zero,data.newtriangles_zero)
        area_sampled[i]=area(temp_zero,data.newtriangles_zero)
        for j in range(3):
            for k in range(3):
                moment_tensor_sampled[i,j,k]=torch.mean(temp_zero.reshape(-1,3)[:,j]*temp_zero.reshape(-1,3)[:,k],axis=0)

    area_sampled=area_sampled.reshape(-1,1)    
    vol_real=vol_real.reshape(-1)
    vol_sampled=vol_sampled.reshape(-1)

    #print(vol_sampled)
    variance=torch.sum(torch.var(temparr,axis=0))
    f = open("./inference_measures/"+name+"_sampled.txt", "a")
    f.write("MMD Area distance of"+name+" is "+str(relativemmd(area_real,area_sampled))+"\n")
    print("MMD Area distance of"+name+" is "+str(relativemmd(area_real,area_sampled))+"\n")
    f.write("Percentage error "+name+" is "+str(error.item())+"\n")
    print("Percentage error "+name+" is "+str(error.item())+"\n")
    f.write("Variance from prior of "+name+" is "+str(variance.item())+"\n")
    print("Variance from prior of "+name+" is "+str(variance.item())+"\n")
    f.write("MMD Gaussian Curvature distance of"+name+" is "+str(relativemmd(curvature_gaussian_real,curvature_gaussian_sampled))+"\n")
    print("MMD Gaussian Curvature distance of"+name+" is "+str(relativemmd(curvature_gaussian_real,curvature_gaussian_sampled))+"\n")
    f.write("MMD Id distance of"+name+" is "+str(relativemmd(temparr.reshape(NUMBER_SAMPLES,-1),datanumpy.reshape(NUMBER_SAMPLES,-1)))+"\n")
    print("MMD Id distance of"+name+" is "+str(relativemmd(temparr.reshape(NUMBER_SAMPLES,-1),datanumpy.reshape(NUMBER_SAMPLES,-1)))+"\n")
    f.close()

    fig2,ax2=plt.subplots()
    ax2.set_title("Area of sampled"+name)
    _=ax2.hist([area_real.reshape(-1),area_sampled.reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/Area_hist_sampled_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("XX moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,0].reshape(-1),moment_tensor_sampled[:,0,0].reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/XXaxis_hist_sampled_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("YY moment of "+name)
    _=ax2.hist([moment_tensor_data[:,1,1].reshape(-1),moment_tensor_sampled[:,1,1].reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/YYaxis_hist_sampled_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("ZZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,2,2].reshape(-1),moment_tensor_sampled[:,2,2].reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/ZZaxis_hist_sampled_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("XY moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,1].reshape(-1),moment_tensor_sampled[:,0,1].reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/XYaxis_hist_sampled_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("XZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,2].reshape(-1),moment_tensor_sampled[:,0,2].reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/XZaxis_hist_sampled_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("YZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,1,2].reshape(-1),moment_tensor_sampled[:,1,2].reshape(-1)],50,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/YZaxis_hist_sampled_"+name+".png")

print("Variance of data is", torch.sum(torch.var(datanumpy[:],axis=0)))
print("Variance of vol true is",torch.var(vol_real))
print("Variance of vol sampled is",torch.var(vol_sampled))
print(torch.linalg.norm(vol_real-vol_sampled)/torch.linalg.norm(vol_real))

    
    
    
