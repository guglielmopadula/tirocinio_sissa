#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""

from datawrapper.data import Data
from basic_layers.decoder import Decoder_base
from basic_layers.encoder import Encoder_base
from typing import Any
import numpy as np
import jax
#jax.config.update('jax_platform_name', 'cpu')
from flax import linen as nn
import optax
from flax.training import train_state
import jax.numpy as jnp
from flax import struct
from clu import metrics
from jax import random
from tqdm import tqdm
LATENT_DIM=20
REDUCED_DIMENSION=30
NUM_TRAIN_SAMPLES=50
NUM_TEST_SAMPLES=50
BATCH_SIZE = 50
MAX_EPOCHS=500
SMOOTHING_DEGREE=1
DROP_PROB=0.1
import matplotlib.pyplot as plt



data=Data(batch_size=BATCH_SIZE,
          num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          string='./data_objects/rabbit_{}.ply',red_dim=REDUCED_DIMENSION)


class Encoder(nn.Module):
    latent_dim:None
    hidden_dim: None
    pca: None

    def setup(self):
        self.encoder_base = Encoder_base(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim,pca=self.pca)

    def __call__(self, x, training):
        z = self.encoder_base(x, training)
        return z

class Decoder(nn.Module):
    size:None
    hidden_dim: None
    pca: None
    barycenter: None
    def setup(self):
        self.decoder_base = Decoder_base(hidden_dim=self.hidden_dim, size=self.size,pca=self.pca,barycenter=self.barycenter)

    def __call__(self, x, training):
        return self.decoder_base(x, training)

def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


class AE(nn.Module):
    latent_dim: None
    hidden_dim: None
    size: None
    pca: None
    barycenter: None

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim,pca=self.pca)
        self.decoder = Decoder(hidden_dim=self.hidden_dim, size=self.size,pca=self.pca,barycenter=self.barycenter)
   
    def __call__(self, batch, rng_key,training):
        x = batch
        mu,logsigma = self.encoder(x, training)
        z=reparameterize(rng_key,mu,logsigma)
        x_hat = self.decoder(z, training)
        return x_hat,mu,logsigma

@struct.dataclass
class Metrics(metrics.Collection):
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics
  key: Any
ae=AE(size=data.get_size(),hidden_dim=150,latent_dim=10,pca=data.pca,barycenter=data.barycenter)


def create_train_state(module, param_key, learning_rate,dropout_key,rng_key):
  variables = module.init(param_key, data.train_dataloader(0), rng_key,training=False) # initialize parameters by passing a template image
  params=variables['params']
  tx = optax.adamw(learning_rate)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,metrics=Metrics.empty(),key=dropout_key)    
    

@jax.jit
def train_step(state, x,dropout_key,rng_key):
    def loss_fn(params):
        x_hat,mu,logsigma = state.apply_fn({'params': params}, x, rng_key,training=True,rngs={'dropout': dropout_key})
        loss = jnp.linalg.norm(x-x_hat)+kl_divergence(mu,logsigma).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@jax.jit
def compute_metrics_train(*, state, batch,dropout_key,rng_key):
    x_hat,mu,logsigma = state.apply_fn({'params': state.params}, batch, rng_key,training=True,rngs={'dropout': dropout_key})
    loss = jnp.linalg.norm(batch-x_hat)/jnp.linalg.norm(batch)
    metric_updates = state.metrics.single_from_model_output(x_hat=x_hat,loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

@jax.jit
def compute_metrics_test(*, state, batch, rng_key):
    x_hat,mu,logsigma = state.apply_fn({'params': state.params}, batch,rng_key, training=False)
    loss = jnp.linalg.norm(batch-x_hat)/jnp.linalg.norm(batch)
    metric_updates = state.metrics.single_from_model_output(x_hat=x_hat,loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state



learning_rate = 0.001
init_rng = jax.random.PRNGKey(0)
main_key, params_key, dropout_key, rng_key = jax.random.split(key=init_rng, num=4)
state = create_train_state(ae, init_rng, learning_rate, dropout_key,rng_key)
num_train_steps_per_epoch=NUM_TRAIN_SAMPLES//BATCH_SIZE
num_test_steps_per_epoch=NUM_TEST_SAMPLES//BATCH_SIZE
metrics_history = {'train_loss': [],
                   'test_loss': [],
                  }
for epochs in tqdm(range(MAX_EPOCHS)):
    for i in range(num_train_steps_per_epoch):
        batch=data.train_dataloader(i)
        dropout_key,_=jax.random.split(dropout_key,num=2)
        rng_key,_=jax.random.split(rng_key,num=2)
        state = train_step(state, batch,dropout_key,rng_key)
        state = compute_metrics_train(state=state, batch=batch, dropout_key=dropout_key,rng_key=rng_key)
    for metric,value in state.metrics.compute().items(): # compute metrics
      metrics_history[f'train_{metric}'].append(value) # record metrics
    state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch
    test_state=state  
    for i in range(num_test_steps_per_epoch):
        batch=data.test_dataloader(i)
        rng_key,_=jax.random.split(rng_key,num=2)
        test_state = compute_metrics_test(state=test_state, batch=batch, rng_key=rng_key)
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