# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DP-GCN hyperparameter configuration."""

from typing import Any
import ml_collections


def get_hyper(hyper):
  """Defines the hyperparameter sweeps."""
  return hyper.product(
      [hyper.sweep('config.training_noise_multiplier', [0.5, 1., 2., 4.])])


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.dataset = 'ogbn-arxiv-disjoint'
  config.dataset_path = 'datasets/'
  config.wandb_project = 'gnn_adam_corr-v2'
  config.experiment_name = 'baseline'
  config.pad_subgraphs_to = 100
  config.multilabel = False
  config.adjacency_normalization = 'inverse-degree'
  config.model = 'gcn'
  config.latent_size = 100
  config.num_encoder_layers = 1
  config.num_message_passing_steps = 1
  config.num_decoder_layers = 2
  config.activation_fn = 'tanh'
  config.num_classes = 40
  config.max_degree = 7
  config.differentially_private_training = True
  config.num_estimation_samples = 10000
  config.l2_norm_clip_percentile = 75
  config.training_noise_multiplier = 2.
  config.num_training_steps = 3000
  config.max_training_epsilon = 12
  config.evaluate_every_steps = 50
  config.checkpoint_every_steps = 50
  config.rng_seed = 123
  config.optimizer = 'adam'
  config.learning_rate = 3e-3
  if config.optimizer == 'sgd':
    config.momentum = 0.
    config.nesterov = False
    config.learning_rate = 1.
  if config.optimizer == 'adamcorr':
    config.b1 = 0.9
    config.eps_root = 1e-8
  if config.optimizer == 'adam':
    config.b1 = 0.9
    config.eps = 1e-12
  config.batch_size = 10000
  return config
