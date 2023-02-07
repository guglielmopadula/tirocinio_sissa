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
from models.basic_layers.volumenormalizer import volume_2_y
from models.BEGAN import BEGAN
import trimesh
import matplotlib.pyplot as plt
import torch
import numpy as np
import meshio
from models.losses.losses import relativemmd
cuda_avail=True if torch.cuda.is_available() else False

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


NUM_WORKERS = int(os.cpu_count() / 2)

LATENT_DIM_1=20
LATENT_DIM_2=1
NUM_TRAIN_SAMPLES=100
NUM_TEST_SAMPLES=0
REDUCED_DIMENSION_1=126
REDUCED_DIMENSION_2=1
BATCH_SIZE = 20
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1
NUMBER_SAMPLES=NUM_TEST_SAMPLES+NUM_TRAIN_SAMPLES



print("Loading data")

data=torch.load("./data_objects/data.pt", map_location="cpu")

d={
   #AE: "AE",
  #AAE: "AAE",
  #VAE: "VAE", 
  BEGAN: "BEGAN",
}

print("Getting properties of the data")
oldmesh=data.oldmesh.clone().numpy()
area_real=np.zeros(NUMBER_SAMPLES)
curvature_gaussian_real=np.zeros([NUMBER_SAMPLES,np.max(data.newtriangles_zero)+1])
curvature_total_real=np.zeros(NUMBER_SAMPLES)
    
for i in range(NUMBER_SAMPLES):
    temp_zero=data.temp_zero.clone().numpy()
    temp_zero[data.local_indices_1]=data.data[i,:np.prod(data.get_size()[0])].reshape(data.get_size()[0][1],data.get_size()[0][2]).detach().numpy()
    temp_zero[data.local_indices_2,0]=data.data[i,np.prod(data.get_size()[0]):].reshape(data.get_size()[1][1],data.get_size()[1][2]).detach().numpy()[:,0]
    temp_zero[data.local_indices_2,2]=data.data[i,np.prod(data.get_size()[0]):].reshape(data.get_size()[1][1],data.get_size()[1][2]).detach().numpy()[:,1]
    mesh_object=trimesh.base.Trimesh(temp_zero,data.newtriangles_zero,process=False)
    curvature_gaussian_real[i]=trimesh.curvature.discrete_gaussian_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_total_real[i]=np.sum(curvature_gaussian_real[i])
    area_real[i]=mesh_object.area
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
    temparr=torch.zeros(NUMBER_SAMPLES,*tuple(tmp.shape))
    vol=torch.zeros(NUMBER_SAMPLES)
    curvature_gaussian_sampled=np.zeros([NUMBER_SAMPLES,np.max(data.newtriangles_zero)+1])
    area_sampled=np.zeros(NUMBER_SAMPLES)
    print("Sampling of "+name+ " has started")

    for i in range(NUMBER_SAMPLES):
        temp_zero=data.temp_zero.clone().cpu().numpy()
        tmp = model.sample_mesh()
        tmp=tmp.reshape(-1)
        temp_interior=tmp[:np.prod(data.get_size()[0])].cpu()
        temp_boundary=tmp[np.prod(data.get_size()[0]):].cpu()
        oldmesh=data.oldmesh.clone()
        temparr[i]=tmp
        oldmesh[data.global_indices_1]=temp_interior.reshape(-1,3)
        oldmesh[data.global_indices_2,0]=temp_boundary.reshape(-1,2)[:,0]
        oldmesh[data.global_indices_2,2]=temp_boundary.reshape(-1,2)[:,1]
        vol[i]=volume_2_y(oldmesh[data.oldM].reshape(1,oldmesh[data.oldM].shape[0],oldmesh[data.oldM].shape[1],oldmesh[data.oldM].shape[2])).reshape(-1)
        meshio.write_points_cells("./inference_objects/"+name+'_{}.stl'.format(i),oldmesh.detach(),[("triangle", data.oldM)])
        true_interior=data.data[:,:np.prod(data.get_size()[0])].reshape(data.num_samples,-1)
        true_boundary=data.data[:,np.prod(data.get_size()[0]):].reshape(data.num_samples,-1)
        temp_interior=temp_interior.reshape(1,-1)
        temp_boundary=temp_boundary.reshape(1,-1)
        error=error+0.5*torch.min(torch.norm(temp_interior-true_interior,dim=1))/torch.norm(temp_interior)/NUMBER_SAMPLES+0.5*torch.min(torch.norm(temp_boundary-true_boundary,dim=1))/torch.norm(temp_boundary)/NUMBER_SAMPLES
        temp_zero[data.local_indices_1]=temp_interior.reshape(data.get_size()[0][1],data.get_size()[0][2]).detach().numpy()
        temp_zero[data.local_indices_2,0]=temp_boundary.reshape(data.get_size()[1][1],data.get_size()[1][2]).detach().numpy()[:,0]
        temp_zero[data.local_indices_2,2]=temp_boundary.reshape(data.get_size()[1][1],data.get_size()[1][2]).detach().numpy()[:,1]
        mesh_object=trimesh.base.Trimesh(temp_zero,data.newtriangles_zero,process=False)
        curvature_gaussian_sampled[i]=trimesh.curvature.discrete_gaussian_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
        area_sampled[i]=mesh_object.area
    area_sampled=area_sampled.reshape(-1,1)

    variance=torch.sum(torch.var(temparr,dim=0))
    variance_vol=torch.sum(torch.var(vol,dim=0))
    f = open("./inference_measures/"+name+".txt", "a")
    f.write("MMD Area distance of"+name+" is "+str(relativemmd(area_real,area_sampled))+"\n")
    print("MMD Area distance of"+name+" is "+str(relativemmd(area_real,area_sampled))+"\n")
    f.write("Percentage error "+name+" is "+str(error.item())+"\n")
    print("Percentage error "+name+" is "+str(error.item())+"\n")
    f.write("Variance from prior of "+name+" is "+str(variance.item())+"\n")
    print("Variance from prior of "+name+" is "+str(variance.item())+"\n")
    f.write("MMD Gaussian Curvature distance of"+name+" is "+str(relativemmd(curvature_gaussian_real,curvature_gaussian_sampled))+"\n")
    print("MMD Gaussian Curvature distance of"+name+" is "+str(relativemmd(curvature_gaussian_real,curvature_gaussian_sampled))+"\n")
    f.write("MMD Id distance of"+name+" is "+str(relativemmd(temparr.reshape(NUMBER_SAMPLES,-1).detach(),data.data[:NUMBER_SAMPLES].reshape(NUMBER_SAMPLES,-1).detach()))+"\n")
    print("MMD Id distance of"+name+" is "+str(relativemmd(temparr.reshape(NUMBER_SAMPLES,-1).detach(),data.data[:NUMBER_SAMPLES].reshape(NUMBER_SAMPLES,-1).detach()))+"\n")
    f.close()

    fig1,ax1=plt.subplots()
    ax1.set_title("Area of "+name)
    _=ax1.plot(np.sort(area_real.reshape(-1)), np.linspace(0, 1, len(area_real)),'r',label='true')
    _=ax1.plot(np.sort(area_sampled.reshape(-1)), np.linspace(0, 1, len(area_sampled)),'g',label='sampled')
    ax1.legend()
    fig1.savefig("./inference_graphs/Area_cdf_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("Area of "+name)
    _=ax2.hist([area_real.reshape(-1),area_sampled.reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/Area_hist_"+name+".png")
print("Variance of data is", torch.sum(torch.var(data.data[:],dim=0)))
    
    
    
    
