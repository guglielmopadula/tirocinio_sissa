#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cyberguli
"""
from datawrapper.data import Data
import os
from models.AE import AE
from models.AAE import AAE
import sys
from models.VAE import VAE
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from models.BEGAN import BEGAN
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
torch.cuda.empty_cache()
print("hello")

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

LATENT_DIM=10
REDUCED_DIMENSION=30
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
NUM_VAL_SAMPLES=0
BATCH_SIZE = 200

MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1


data=torch.load("./data_objects/data.pt", map_location="cpu")
if use_cuda:
    data.pca.to("cuda:0")
    data.temp_zero=data.temp_zero.cuda()
    data.vertices_face_x=data.vertices_face_x.cuda()
    data.vertices_face_xy=data.vertices_face_xy.cuda()
    data.edge_matrix=data.edge_matrix.cuda()


    

d={
  AE: "AE",
  #AAE: "AAE",
  #VAE: "VAE", 
  #BEGAN: "BEGAN",
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
        print("Training of "+name+ "has started")
        trainer.fit(model, data)
        trainer.test(model,data)
        torch.save(model,"./saved_models/"+name+".pt")
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
