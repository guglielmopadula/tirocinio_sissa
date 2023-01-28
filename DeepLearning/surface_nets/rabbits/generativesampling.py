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
NUM_TEST_SAMPLES=0
BATCH_SIZE = 20
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1
NUMBER_SAMPLES=NUM_TEST_SAMPLES+NUM_TRAIN_SAMPLES



print("Loading data")

LATENT_DIM=10
REDUCED_DIMENSION=140
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
BATCH_SIZE = 2
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1

data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data_objects/rabbit_{}.ply",
          use_cuda=False)


d={
  AE: "AE",
  #AAE: "AAE",
  #VAE: "VAE", 
  #BEGAN: "BEGAN",
}

for wrapper, name in d.items():
    torch.manual_seed(100)
    np.random.seed(100)
    if cuda_avail==False:
        model=torch.load("./saved_models/"+name+".pt",map_location=torch.device('cpu'))
    else:
        model=torch.load("./saved_models/"+name+".pt")

    model.eval()
    for i in range(100):
        tmp=data.points_old.cpu().detach().numpy()
        tmp2=model.sample_mesh().cpu().detach().numpy()
        tmp[data.indices]=tmp2.reshape(-1,3)
        meshio.write_points_cells("./data_objects/AE_{}.ply".format(i), tmp,[])

    