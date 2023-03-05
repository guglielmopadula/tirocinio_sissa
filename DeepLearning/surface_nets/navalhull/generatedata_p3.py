#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cyberguli
"""
from datawrapper.data import Data
import os
from tqdm import trange
from skdim.id import TwoNN
from models.DAE import DAE
import sys
import meshio
import matplotlib.pyplot as plt
from models.basic_layers.volumenormalizer import VolumeNormalizer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
torch.cuda.empty_cache()

class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return




NUM_WORKERS = 0
use_cuda=True if torch.cuda.is_available() else False
AVAIL_GPUS=1 if torch.cuda.is_available() else 0

LATENT_DIM=20
REDUCED_DIMENSION=30
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
NUM_SAMPLES=NUM_TRAIN_SAMPLES+NUM_TEST_SAMPLES
NUM_VAL_SAMPLES=0
BATCH_SIZE = 200

MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1

def area(vertices, triangles):
    triangles=torch.tensor(triangles)
    v1 = vertices[triangles[:,0]]
    v2 = vertices[triangles[:,1]]
    v3 = vertices[triangles[:,2]]
    a = torch.linalg.norm(torch.cross(v2 - v1, v3 - v1), axis=1) / 2
    return torch.sum(a)



data=torch.load("./data_objects/data.pt", map_location="cpu")
if use_cuda:
    data.pca.to("cuda:0")
    data.temp_zero=data.temp_zero.cuda()
    data.vertices_face_x=data.vertices_face_x.cuda()
    data.vertices_face_xy=data.vertices_face_xy.cuda()
    data.edge_matrix=data.edge_matrix.cuda()

    

d={
#VolumeNormalizer:"VOL",
DAE: "DAE",
}

if __name__ == "__main__":
    for wrapper, name in d.items():
        torch.manual_seed(100)
        np.random.seed(100)
        checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints/"+name+"/",every_n_epochs=50,filename='{epoch}',save_top_k=-1, save_last=True)
        if use_cuda:
            trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,logger=False,
                                  gradient_clip_val=0.1, plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                  callbacks=[checkpoint_callback]) 
        else:
            trainer=Trainer(max_epochs=MAX_EPOCHS,logger=False,
                                gradient_clip_val=0.1,plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                callbacks=[checkpoint_callback]
                                )   
        model=wrapper(data_shape=data.get_size(),reduced_data_shape=data.get_reduced_size(),temp_zero=data.temp_zero,local_indices_1=data.local_indices_1,local_indices_2=data.local_indices_2,newtriangles_zero=data.newtriangles_zero,pca=data.pca,edge_matrix=data.edge_matrix,vertices_face_x=data.vertices_face_x,vertices_face_xy=data.vertices_face_xy,k=SMOOTHING_DEGREE,latent_dim=LATENT_DIM,batch_size=BATCH_SIZE,drop_prob=DROP_PROB)
        #model=wrapper(temp_zero=data.temp_zero,local_indices_1=data.local_indices_1,local_indices_2=data.local_indices_2,newtriangles_zero=data.newtriangles_zero,vertices_face_x=data.vertices_face_x,vertices_face_xy=data.vertices_face_xy)

        oldmesh=data.oldmesh.clone().cpu()
        print("Training of "+name+ "has started")
        trainer.fit(model, data)
        model.eval()
        trainer.test(model,data)
        model=model.cuda()
        model.eval()
        alls=torch.zeros([NUM_SAMPLES,6989])
        area_real=torch.zeros(NUM_SAMPLES)
        area_sampled=torch.zeros(NUM_SAMPLES)
        moment_tensor_data=torch.zeros((NUM_SAMPLES,3,3))
        moment_tensor_sampled=torch.zeros((NUM_SAMPLES,3,3))

        for i in trange(NUM_SAMPLES):
            temp_zero=data.temp_zero.clone().cpu()
            oldmesh=data.oldmesh.clone().cpu()
            true_interior=data.data[i,:torch.prod(torch.tensor(data.get_size()[0]))].reshape(-1,3)
            true_boundary=data.data[i,torch.prod(torch.tensor(data.get_size()[0])):].reshape(-1,2)
            temp_zero[data.local_indices_1]=true_interior.reshape(-1,3)
            temp_zero[data.local_indices_2,0]=true_boundary.reshape(-1,2)[:,0]
            temp_zero[data.local_indices_2,2]=true_boundary.reshape(-1,2)[:,1]
            area_real[i]=area(temp_zero,data.newtriangles_zero)
            for j in range(3):
                for k in range(3):
                    moment_tensor_data[i,j,k]=torch.mean(temp_zero.reshape(-1,3)[:,j]*temp_zero.reshape(-1,3)[:,k],axis=0)
            temp_zero=data.temp_zero.clone().cpu()
            tmp = model.decoder(model.encoder(data.data[i].unsqueeze(0).cuda())).cpu().reshape(-1)
            alls[i]=tmp
            temp_interior=tmp[:torch.prod(torch.tensor(data.get_size()[0]))].reshape(-1,3)
            temp_boundary=tmp[torch.prod(torch.tensor(data.get_size()[0])):].reshape(-1,2)

            #temp_interior,temp_boundary=model(true_interior.cuda().unsqueeze(0),true_boundary.cuda().unsqueeze(0))
            oldmesh[data.global_indices_1]=temp_interior.reshape(-1,3)
            oldmesh[data.global_indices_2,0]=temp_boundary.reshape(-1,2)[:,0]
            oldmesh[data.global_indices_2,2]=temp_boundary.reshape(-1,2)[:,1]
            temp_zero[data.local_indices_1]=temp_interior.reshape(-1,3)
            temp_zero[data.local_indices_2,0]=temp_boundary.reshape(-1,2)[:,0]
            temp_zero[data.local_indices_2,2]=temp_boundary.reshape(-1,2)[:,1]
            area_sampled[i]=area(temp_zero,data.newtriangles_zero)
            for j in range(3):
                for k in range(3):
                    moment_tensor_sampled[i,j,k]=torch.mean(temp_zero.reshape(-1,3)[:,j]*temp_zero.reshape(-1,3)[:,k],axis=0)
            oldmesh=oldmesh.detach()
            meshio.write_points_cells("./data_objects/"+name+'_{}.stl'.format(i),oldmesh,[("triangle", data.oldM)])

        alls=alls.detach()
        print(TwoNN().fit(alls).dimension_)
        area_sampled=area_sampled.detach()
        moment_tensor_sampled=moment_tensor_sampled.detach()
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
