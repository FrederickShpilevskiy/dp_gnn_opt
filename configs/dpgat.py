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

"""DP-GAT hyperparameter configuration."""

from typing import Any
import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.dataset = 'ogbn-arxiv-disjoint'
  config.dataset_path = 'datasets/'
  config.wandb_project = 'dp-gnn-extension'
  config.experiment_name = 'dpgat_baseline'
  config.group = 'gat_baseline'
  config.pad_subgraphs_to = 100
  config.multilabel = False
  config.adjacency_normalization = 'inverse-degree'
  config.model = 'gat'
  config.latent_size = 255
  config.num_encoder_layers = 1
  config.num_message_passing_steps = 1
  config.num_decoder_layers = 2
  config.activation_fn = 'tanh'
  config.num_classes = 40
  config.max_degree = 7
  config.differentially_private_training = True
  config.num_estimation_samples = 10000
  config.l2_norm_clip_percentile = 75
  config.l2_norm_threshold = 0.
  config.training_noise_multiplier = 2.
  config.num_training_steps = 10000
  config.max_training_epsilon = 12
  config.evaluate_every_steps = 10
  config.resample_every_steps = 0
  config.checkpoint_every_steps = 50
  config.rng_seed = 86583
  config.optimizer = 'adam'
  config.learning_rate = 4e-3
  config.momentum = 0.
  config.nesterov = False
  config.eps_root = 0.1
  config.b1 = 0.9
  config.eps = 1e-12
  config.batch_size = 20000
  config.negative_slope = 0.2
  return config
