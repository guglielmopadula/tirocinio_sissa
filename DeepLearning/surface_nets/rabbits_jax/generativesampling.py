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
NUM_TRAIN_SAMPLES=100
REDUCED_DIMENSION=140
NUM_TEST_SAMPLES=0
BATCH_SIZE = 1
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1
NUMBER_SAMPLES=NUM_TEST_SAMPLES+NUM_TRAIN_SAMPLES




d={
  AE: "AE",
  AAE: "AAE",
  VAE: "VAE", 
  BEGAN: "BEGAN",
}

data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data_objects/rabbit_{}.ply",
          use_cuda=False)


data=data.data[:].cpu().numpy().reshape(NUMBER_SAMPLES,-1)


for wrapper, name in d.items():
    torch.manual_seed(100)
    np.random.seed(100)
    temp=np.zeros(data.shape)
    model=torch.load("./saved_models/"+name+".pt",map_location=torch.device('cpu'))
    model.eval()
    for i in range(100):
        tmp=model.sample_mesh().cpu().detach().numpy().reshape(-1,3)
        temp[i]=tmp.reshape(-1)
        #meshio.write_points_cells("./inference_objects/"+name+"_{}.ply".format(i), tmp,[])
    print("Second moments of data is ",np.mean(scipy.stats.moment(data,2))," and of ",name, " is ", np.mean(scipy.stats.moment(temp,2)))
    print("Third moments of data is ",np.mean(scipy.stats.moment(data,3))," and of ",name, " is ", np.mean(scipy.stats.moment(temp,3)))
    print("Fourth moments of data is ",np.mean(scipy.stats.moment(data,4))," and of ",name, " is ", np.mean(scipy.stats.moment(temp,4)))
    print("Fifth moments of data is ",np.mean(scipy.stats.moment(data,5))," and of ",name, " is ", np.mean(scipy.stats.moment(temp,5)))

