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
from models.BEGAN import BEGAN
import trimesh
import matplotlib.pyplot as plt
import torch
import scipy
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

LATENT_DIM_1=10
LATENT_DIM_2=1
NUM_TRAIN_SAMPLES=600
REDUCED_DIMENSION=140
NUM_TEST_SAMPLES=0
BATCH_SIZE = 1
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1
NUMBER_SAMPLES=NUM_TEST_SAMPLES+NUM_TRAIN_SAMPLES




d={
  #AE: "AE",
  #AAE: "AAE",
  VAE: "VAE", 
  #BEGAN: "BEGAN",
}

data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data_objects/rabbit_{}.ply",
          use_cuda=False)


data=data.data[:].cpu().numpy().reshape(NUMBER_SAMPLES,-1)

moment_tensor_data=np.zeros((NUMBER_SAMPLES,3,3))



data2=data.copy()
data2=data2.reshape(NUMBER_SAMPLES,-1,3)
#print(np.mean(data2,axis=1))
#data2=data2-np.mean(data2,axis=1).reshape(NUMBER_SAMPLES,1,3).repeat(data2.shape[1],axis=1)
for j in range(3):
    for k in range(3):
        moment_tensor_data[:,j,k]=np.mean(data2.reshape(NUMBER_SAMPLES,-1,3)[:,:,j]*data2.reshape(NUMBER_SAMPLES,-1,3)[:,:,k],axis=1)

for wrapper, name in d.items():
    torch.manual_seed(100)
    np.random.seed(100)
    temp=np.zeros(data.shape)
    model=torch.load("./saved_models/"+name+".pt",map_location=torch.device('cpu'))
    model.eval()
    tmp,z=model.sample_mesh()
    latent_space=torch.zeros(NUMBER_SAMPLES,np.prod(z.shape))
    for i in range(NUMBER_SAMPLES):
        tmp,z=model.sample_mesh()
        latent_space[i]=z
        tmp=tmp.cpu().detach().numpy().reshape(-1,3)
        tmp=tmp.reshape(-1,3)
        #tmp=tmp-np.mean(tmp,axis=0)
        temp[i]=tmp.reshape(-1)
        meshio.write_points_cells("./inference_objects/"+name+"_{}.ply".format(i), tmp,[])
    moment_tensor_sampled=np.zeros((NUMBER_SAMPLES,3,3))

    print("Variance of ",name," is", np.sum(np.var(temp.reshape(NUMBER_SAMPLES,-1),axis=0)))
    np.save(name+"_latent",latent_space.numpy())
    
    for j in range(3):
        for k in range(3):
            moment_tensor_sampled[:,j,k]=np.mean(temp.reshape(NUMBER_SAMPLES,-1,3)[:,:,j]*temp.reshape(NUMBER_SAMPLES,-1,3)[:,:,k],axis=1)

    fig2,ax2=plt.subplots()
    ax2.set_title("XX moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,0].reshape(-1),moment_tensor_sampled[:,0,0].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/XXaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("YY moment of "+name)
    _=ax2.hist([moment_tensor_data[:,1,1].reshape(-1),moment_tensor_sampled[:,1,1].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/YYaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("ZZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,2,2].reshape(-1),moment_tensor_sampled[:,2,2].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/ZZaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("XY moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,1].reshape(-1),moment_tensor_sampled[:,0,1].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/XYaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("XZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,0,2].reshape(-1),moment_tensor_sampled[:,0,2].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/XZaxis_hist_"+name+".png")
    fig2,ax2=plt.subplots()
    ax2.set_title("YZ moment of "+name)
    _=ax2.hist([moment_tensor_data[:,1,2].reshape(-1),moment_tensor_sampled[:,1,2].reshape(-1)],8,label=['real','sampled'])
    ax2.legend()
    fig2.savefig("./inference_graphs/YZaxis_hist_"+name+".png")
    


print("Variance of data is", np.sum(np.var(data2[:].reshape(NUMBER_SAMPLES,-1),axis=0)))

