#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""

from datawrapper.data import Data
import os
from models.AE import AE
from models.AAE import AAE
from models.VAE import VAE
from pytorch_lightning.profiler import AdvancedProfiler
from models.BEGAN import BEGAN
import torch
import numpy as np
from pytorch_lightning import Trainer

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




NUM_WORKERS = os.cpu_count()
use_cuda=True if torch.cuda.is_available() else False
AVAIL_GPUS=1 if torch.cuda.is_available() else 0


LATENT_DIM_1=5
LATENT_DIM_2=1
REDUCED_DIMENSION_1=24
REDUCED_DIMENSION_2=1
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
BATCH_SIZE = 20

MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1

data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension_1=REDUCED_DIMENSION_1, 
          reduced_dimension_2=REDUCED_DIMENSION_2, 
          string="./data_objects/hull_{}.stl",
          use_cuda=use_cuda)
d={
  #AE: "AE",
  #AAE: "AAE",
  VAE: "VAE", 
  #BEGAN: "BEGAN",
}
if __name__ == "__main__":
    for wrapper, name in d.items():
        torch.manual_seed(100)
        np.random.seed(100)
        if use_cuda:
            trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,log_every_n_steps=1,track_grad_norm=2,
                                  gradient_clip_val=0.1, plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                  )
        else:
            trainer=Trainer(max_epochs=MAX_EPOCHS,log_every_n_steps=1,track_grad_norm=2,
                                gradient_clip_val=0.1,plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                )   
        model=wrapper(data.get_reduced_size(),data.temp_zero,data.local_indices_1,data.local_indices_2,data.newtriangles_zero,data.pca_1,data.pca_2,data.edge_matrix,data.vertices_face,data.cvxpylayer,k=SMOOTHING_DEGREE,latent_dim_1=LATENT_DIM_1,latent_dim_2=LATENT_DIM_2,batch_size=BATCH_SIZE,drop_prob=DROP_PROB)
        print("Training of "+name+ "has started")
        trainer.fit(model, data)
        trainer.test(model,data)
        torch.save(model,"./saved_models/"+name+".pt")
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
