#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm import trange
from datawrapper.data import Data
import os
import pyro
from pyro.distributions import Normal
from models.BAE import BAE
import torch
import time
import numpy as np
from pyro.infer import MCMC,NUTS
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




NUM_WORKERS = os.cpu_count()//2
use_cuda=True if torch.cuda.is_available() else False
AVAIL_GPUS=1 if torch.cuda.is_available() else 0
use_cuda=False

LATENT_DIM=10
REDUCED_DIMENSION=140
NUM_TRAIN_SAMPLES=10
NUM_TEST_SAMPLES=200
BATCH_SIZE = 1
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1

data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data_objects/rabbit_{}.ply",
          use_cuda=use_cuda)




model=BAE(data_shape=data.get_reduced_size(),pca=data.pca,latent_dim=LATENT_DIM,batch_size=BATCH_SIZE,drop_prob=DROP_PROB,barycenter=data.barycenter)
def bayesian_constructor(model):
    def inner_model(x):
        prior={}
        for name, data in model.named_parameters():
            if 'weight' in name:
                prior[name]=Normal(loc=torch.zeros_like(data),scale=torch.ones_like(data)).to_event(2)
            if 'bias' in name:
                prior[name]=Normal(loc=torch.zeros_like(data),scale=torch.ones_like(data)).to_event(1)

        lifted_module = pyro.random_module("module", model, prior)
        # sample a regressor (which also samples w and b)
        lifted_model = lifted_module()
        with pyro.plate("data", x.shape[0]):
            x_hat=lifted_model(x)
            obs = pyro.sample("obs", Normal(x_hat,0.0001).to_event(1), obs=x,)

    return inner_model

bayesian_model=bayesian_constructor(model)
def run_inference(model, X):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        warmup_steps=5,
        num_samples=NUM_TRAIN_SAMPLES,
        num_chains=1)
    mcmc.run(X)
    print("Diagnostic\n")
    mcmc.diagnostics()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

x=run_inference(bayesian_model,data.data_train[:].reshape(NUM_TRAIN_SAMPLES,-1))
predictive = Predictive(bayesian_model, x)(data.data_train[:].reshape(NUM_TRAIN_SAMPLES,-1))
print(predictive["obs"].shape)
'''
guide = AutoDiagonalNormal(bayesian_model)
adam = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(bayesian_model, guide, adam, loss=Trace_ELBO())

pyro.clear_param_store()
for epoch in trange(20000):
    loss = svi.step(data.data_train[:].reshape(NUM_TRAIN_SAMPLES,-1))
'''
    
    
    
    
    
    
    
    
    
