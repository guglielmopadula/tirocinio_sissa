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
from pytorch_lightning.callbacks import ModelCheckpoint
from models.BEGAN import BEGAN
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


LATENT_DIM_1=10
LATENT_DIM_2=1
REDUCED_DIMENSION_1=126
REDUCED_DIMENSION_2=1
NUM_TRAIN_SAMPLES=4
NUM_TEST_SAMPLES=2
BATCH_SIZE = 2

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
  AE: "AE",
  AAE: "AAE",
  VAE: "VAE", 
  BEGAN: "BEGAN",
}
if __name__ == "__main__":
    for wrapper, name in d.items():
        torch.manual_seed(100)
        np.random.seed(100)
        checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints/"+name+"/",every_n_epochs=50,filename='{epoch}',save_top_k=-1, save_last=True)
        if use_cuda:
            trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=MAX_EPOCHS,logger=False,
                                  gradient_clip_val=0.1, plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                  callbacks=[checkpoint_callback]
                                  )
        else:
            trainer=Trainer(max_epochs=MAX_EPOCHS,logger=False,
                                gradient_clip_val=0.1,plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                callbacks=[checkpoint_callback]
                                )   
        model=wrapper(data.get_reduced_size(),data.temp_zero,data.local_indices_1,data.local_indices_2,data.newtriangles_zero,data.pca_1,data.pca_2,data.edge_matrix,data.vertices_face,data.cvxpylayer,k=SMOOTHING_DEGREE,latent_dim_1=LATENT_DIM_1,latent_dim_2=LATENT_DIM_2,batch_size=BATCH_SIZE,drop_prob=DROP_PROB)
        print("Training of "+name+ "has started")
        trainer.fit(model, data)
        trainer.test(model,data)
        torch.save(model,"./saved_models/"+name+".pt")
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
