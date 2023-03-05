#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:09:55 2023

@author: cyberguli
"""
from datawrapper.data import Data
import torch
NUM_WORKERS = 0
use_cuda=True if torch.cuda.is_available() else False
AVAIL_GPUS=1 if torch.cuda.is_available() else 0
REDUCED_DIMENSION=30
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
NUM_VAL_SAMPLES=0
BATCH_SIZE = 200

data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_val=NUM_VAL_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data_objects/hull_{}.stl",
          use_cuda=use_cuda)
print("Start saving")
torch.save(data,"./data_objects/data.pt")
