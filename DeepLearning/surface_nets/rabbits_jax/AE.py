#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""

from datawrapper.data import Data
import os
from basic_layers.decoder import Decoder_base
from basic_layers.encoder import Encoder_base

import numpy as np
import jax
from flax import linen as nn
import optax
from flax.training import train_state
import jax.numpy as jnp
from flax import struct
from clu import metrics
from tqdm import tqdm
LATENT_DIM=10
REDUCED_DIMENSION=140
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
BATCH_SIZE = 100
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1
import matplotlib.pyplot as plt



data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          reduced_dimension=REDUCED_DIMENSION, 
          string="./data_objects/rabbit_{}.ply")


class Encoder(nn.Module):
    data_shape:None
    batch_size:None
    latent_dim:None
    pca:None
    drop_prob:None
    hidden_dim: None

    def setup(self):
        self.encoder_base = Encoder_base(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim,
                                         data_shape=self.data_shape, pca=self.pca, drop_prob=self.drop_prob, batch_size=self.batch_size)

    def __call__(self, x):
        z = self.encoder_base(x)
        return z

class Decoder(nn.Module):
    data_shape:None
    batch_size:None
    latent_dim:None
    pca:None
    drop_prob:None
    hidden_dim: None
    barycenter:None
    def setup(self):
        self.decoder_base = Decoder_base(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, data_shape=self.data_shape,pca=self.pca, batch_size=self.batch_size, drop_prob=self.drop_prob, barycenter=self.barycenter)

    def __call__(self, x):
        return self.decoder_base(x)


class AE(nn.Module):
    pca: None
    barycenter: None
    drop_prob: None
    latent_dim: None
    batch_size: None
    hidden_dim: None
    data_shape: None
    batch_size: None

    def setup(self):
        self.encoder = Encoder(data_shape=self.data_shape, latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, pca=self.pca, drop_prob=self.drop_prob, batch_size=self.batch_size)
        self.decoder = Decoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, data_shape=self.data_shape,drop_prob=self.drop_prob, pca=self.pca, batch_size=self.batch_size, barycenter=self.barycenter)
   
    def __call__(self, batch):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.reshape(x.shape)
        return x_hat

@struct.dataclass
class Metrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics

ae=AE(data_shape=data.get_reduced_size(),batch_size=BATCH_SIZE,drop_prob=0.1,hidden_dim=300,latent_dim=10,pca=data.pca,barycenter=data.pca)

def create_train_state(module, rng, learning_rate):
  params = module.init(rng, data.train_dataloader(0))['params'] # initialize parameters by passing a template image
  tx = optax.adamw(learning_rate)
  return TrainState.__dict__TrainState.create(
      apply_fn=module.apply, params=params, tx=tx)    
    

@jax.jit
def train_step(state, x):
    def loss_fn(params):
        x_hat = state.apply_fn({'params': params}, x)
        loss = jnp.linalg.norm(x-x_hat)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss

@jax.jit
def compute_metrics(*, state, batch):
    x_hat = state.apply_fn({'params': params}, batch)
    loss = jnp.linalg.norm(batch-x_hat)
    metric_updates = state.metrics.single_from_model_output(x_hat=x_hat,loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

learning_rate = 0.01
init_rng = jax.random.PRNGKey(0)
state = create_train_state(ae, init_rng, learning_rate)
num_train_steps_per_epoch=NUM_TRAIN_SAMPLES//BATCH_SIZE
num_test_steps_per_epoch=NUM_TEST_SAMPLES//BATCH_SIZE
metrics_history = {'train_loss': [],
                   'test_loss': [],
                  }
for epochs in tqdm(range(MAX_EPOCHS)):
    for i in range(num_train_steps_per_epoch):
        batch=data.train_dataloader(i)
        state = train_step(state, batch)
        state = compute_metrics(state=state, batch=batch)
    for metric,value in state.metrics.compute().items(): # compute metrics
      metrics_history[f'train_{metric}'].append(value) # record metrics
    state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch
    test_state=state
    for i in range(num_test_steps_per_epoch):
        batch=data.test_dataloader(i)
        test_state = compute_metrics(state=test_state, batch=batch)
    for metric,value in test_state.metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    print(f"epoch: {epochs}", f"train loss: {metrics_history['train_loss'][-1]}",f"test loss: {metrics_history['test_loss'][-1]}")


# Plot loss and accuracy in subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
ax1.set_title('Loss')
for dataset in ('train','test'):
  ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
ax1.legend()
plt.show()
plt.clf()