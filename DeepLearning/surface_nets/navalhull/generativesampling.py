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
import numpy as np
import meshio
from models.losses.losses import relativemmd
cuda_avail=True if torch.cuda.is_available() else False
cuda_avail=False
from pytorch_lightning.plugins.environments import SLURMEnvironment

class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return

def volume_2_y(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,1],axis=2)*(np.linalg.det(mesh[np.ix_(np.arange(mesh.shape[0]),np.arange(mesh.shape[1]),(0,2),(0,2))]-mesh[np.ix_(np.arange(mesh.shape[0]),np.arange(mesh.shape[1]),[1],(0,2))])/6),axis=1)
    return tmp.reshape(shape[:-3])


def myarccos(x):
    return np.arccos(np.minimum(np.maximum(-np.ones(x.shape),x),np.ones(x.shape)))


def area(vertices, triangles):
    triangles=np.array(triangles)
    v1 = vertices[triangles[:,0]]
    v2 = vertices[triangles[:,1]]
    v3 = vertices[triangles[:,2]]
    a = np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1) / 2
    return np.sum(a)

def gaussian_curvature(vertices, triangles):
    triangles=np.array(triangles)
    ab=vertices[triangles[:,1]] - vertices[triangles[:,0]]
    ac=vertices[triangles[:,2]] - vertices[triangles[:,0]]
    bc=vertices[triangles[:,2]] - vertices[triangles[:,1]]
    angleA=myarccos(np.einsum('ij,ij->i',ab,ac)/(np.linalg.norm(ab,axis=1)*np.linalg.norm(ac,axis=1)))
    angleB=myarccos(np.einsum('ij,ij->i',-ab,bc)/(np.linalg.norm(bc,axis=1)*np.linalg.norm(ab,axis=1)))
    angleC=myarccos(np.einsum('ij,ij->i',-bc,-ac)/(np.linalg.norm(bc,axis=1)*np.linalg.norm(ac,axis=1)))
    angles=np.concatenate((angleA.reshape(-1,1),angleB.reshape(-1,1),angleC.reshape(-1,1)),axis=1)
    Area=np.linalg.norm(np.cross(ab,ac),axis=1)/2
    Area=Area.reshape(-1,1).repeat(3,1)
    triangles=triangles.flatten()
    Area=Area.flatten()
    angles=angles.flatten()
    gaussian_curvatures=(2*np.pi-np.bincount(triangles,angles))/np.bincount(triangles,(1/3)*Area)
    return gaussian_curvatures

NUM_WORKERS = int(os.cpu_count() / 2)

LATENT_DIM_1=20
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
REDUCED_DIMENSION_1=120
NUMBER_SAMPLES=NUM_TEST_SAMPLES+NUM_TRAIN_SAMPLES



print("Loading data")

data=torch.load("./data_objects/data.pt", map_location="cpu")

d={
  #AE: "AE",
  #AAE: "AAE",
  VAE: "VAE", 
  #BEGAN: "BEGAN",
}

print("Getting properties of the data")
oldmesh=data.oldmesh.clone().numpy()
area_real=np.zeros(NUMBER_SAMPLES)
curvature_gaussian_real=np.zeros([NUMBER_SAMPLES,np.max(data.newtriangles_zero)+1])
curvature_total_real=np.zeros(NUMBER_SAMPLES)
datanumpy=data.data[:NUMBER_SAMPLES].detach().cpu().numpy()   
true_interior=datanumpy[:,:np.prod(data.get_size()[0])].reshape(NUMBER_SAMPLES,-1)
true_boundary=datanumpy[:,np.prod(data.get_size()[0]):].reshape(NUMBER_SAMPLES,-1)
moment_tensor_data=np.zeros((NUMBER_SAMPLES,3,3))
 


for i in trange(NUMBER_SAMPLES):
    temp_zero=data.temp_zero.clone().numpy()
    temp_zero[data.local_indices_1]=datanumpy[i,:np.prod(data.get_size()[0])].reshape(data.get_size()[0][1],data.get_size()[0][2])
    temp_zero[data.local_indices_2,0]=datanumpy[i,np.prod(data.get_size()[0]):].reshape(data.get_size()[1][1],data.get_size()[1][2])[:,0]
    temp_zero[data.local_indices_2,2]=datanumpy[i,np.prod(data.get_size()[0]):].reshape(data.get_size()[1][1],data.get_size()[1][2])[:,1]
    mesh_object=trimesh.base.Trimesh(temp_zero,data.newtriangles_zero,process=False)
    curvature_gaussian_real[i]=gaussian_curvature(temp_zero,data.newtriangles_zero)
    curvature_total_real[i]=np.sum(curvature_gaussian_real[i])
    area_real[i]=area(temp_zero,data.newtriangles_zero)
    for j in range(3):
        for k in range(3):
            moment_tensor_data[i,j,k]=np.mean(temp_zero.reshape(-1,3)[:,j]*temp_zero.reshape(-1,3)[:,k],axis=0)


curvature_total_real=curvature_total_real.reshape(-1,1)
area_real=area_real.reshape(-1,1)

for wrapper, name in d.items():
    torch.manual_seed(100)
    np.random.seed(100)
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
    temparr=np.zeros((NUMBER_SAMPLES,*tuple(tmp.shape)))
    vol=np.zeros(NUMBER_SAMPLES)
    curvature_gaussian_sampled=np.zeros([NUMBER_SAMPLES,np.max(data.newtriangles_zero)+1])
    moment_tensor_sampled=np.zeros((NUMBER_SAMPLES,3,3))
    area_sampled=np.zeros(NUMBER_SAMPLES)
    print("Sampling of "+name+ " has started")
    oldmesh=data.oldmesh.clone().cpu().numpy()

    for i in trange(NUMBER_SAMPLES):
        temp_zero=data.temp_zero.clone().cpu().numpy()
        tmp = model.sample_mesh()
        tmp=tmp.reshape(-1).detach().cpu().numpy()
        temp_interior=tmp[:np.prod(data.get_size()[0])]
        temp_boundary=tmp[np.prod(data.get_size()[0]):]
        temparr[i]=tmp
        oldmesh[data.global_indices_1]=temp_interior.reshape(-1,3)
        oldmesh[data.global_indices_2,0]=temp_boundary.reshape(-1,2)[:,0]
        oldmesh[data.global_indices_2,2]=temp_boundary.reshape(-1,2)[:,1]
        vol[i]=volume_2_y(oldmesh[data.oldM].reshape(1,oldmesh[data.oldM].shape[0],oldmesh[data.oldM].shape[1],oldmesh[data.oldM].shape[2])).reshape(-1)
        meshio.write_points_cells("./inference_objects/"+name+'_sampled_{}.stl'.format(i),oldmesh,[("triangle", data.oldM)])
        temp_interior=temp_interior.reshape(1,-1)
        temp_boundary=temp_boundary.reshape(1,-1)
        error=error+np.min(np.linalg.norm(tmp-datanumpy,axis=1))/np.linalg.norm(datanumpy)/NUMBER_SAMPLES
        temp_zero[data.local_indices_1]=temp_interior.reshape(data.get_size()[0][1],data.get_size()[0][2])
        temp_zero[data.local_indices_2,0]=temp_boundary.reshape(data.get_size()[1][1],data.get_size()[1][2])[:,0]
        temp_zero[data.local_indices_2,2]=temp_boundary.reshape(data.get_size()[1][1],data.get_size()[1][2])[:,1]
        mesh_object=trimesh.base.Trimesh(temp_zero,data.newtriangles_zero,process=False)
        curvature_gaussian_sampled[i]=gaussian_curvature(temp_zero,data.newtriangles_zero)
        area_sampled[i]=area(temp_zero,data.newtriangles_zero)
        for j in range(3):
            for k in range(3):
                moment_tensor_sampled[i,j,k]=np.mean(temp_zero.reshape(-1,3)[:,j]*temp_zero.reshape(-1,3)[:,k],axis=0)

    area_sampled=area_sampled.reshape(-1,1)

    variance=np.sum(np.var(temparr,axis=0))
    variance_vol=np.sum(np.var(vol,axis=0))
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

    fig1,ax1=plt.subplots()
    ax1.set_title("Area of sampled"+name)
    _=ax1.plot(np.sort(area_real.reshape(-1)), np.linspace(0, 1, len(area_real)),'r',label='true')
    _=ax1.plot(np.sort(area_sampled.reshape(-1)), np.linspace(0, 1, len(area_sampled)),'g',label='sampled')
    ax1.legend()
    fig1.savefig("./inference_graphs/Area_cdf_sampled_"+name+".png")
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

print("Variance of data is", np.sum(np.var(datanumpy[:],axis=0)))
    
    
    
    
