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

"""Optimizers."""

import chex
import jax
import jax.numpy as jnp
import optax
from optim import adam, adamcorr

def compute_opt_noise(l2_norms_threshold, base_sensitivity, noise_multiplier):
  return jax.tree_map(lambda l2_norm_clip: l2_norm_clip * base_sensitivity * noise_multiplier, 
      l2_norms_threshold)


def clip_by_norm(updates,
                 l2_norms_threshold):
  """Standard clipping by L2 norm."""

  grad_norms = jax.tree_map(
      jax.vmap(jnp.linalg.norm),
      updates)
  divisors = jax.tree_map(
      lambda g_norm, l2_norm_clip: jnp.maximum(g_norm / l2_norm_clip, 1.0),
      grad_norms, l2_norms_threshold)
  return jax.tree_map(
      jax.vmap(lambda g, div: g / div),
      updates, divisors)


def dp_aggregate(
    l2_norms_threshold,
    base_sensitivity,
    noise_multiplier,
    init_rng,
    return_type='original',
):
  """Aggregates gradients based on the DP-SGD algorithm.

  This method clips per-example gradients to some l2 norm, sums them up,
  and adds noise to the sum.

  WARNING: Unlike other transforms, `dp_aggregate` expects
  the input updates to have a batch dimension in the 0th axis. That is, this
  function expects per-example gradients as input (which are easy to obtain in
  JAX using `jax.vmap`). It can still be composed with other transformations as
  long as it is the first in the chain.
  Further, each per-example gradient must already be divided by the batch size.

  References:
    [Abadi et al, 2016](https://arxiv.org/abs/1607.00133)

  Args:
    l2_norms_threshold: max L2 norm of the per-example gradients for each layer.
    base_sensitivity: ratio of sensitivity to the clipping norm.
    noise_multiplier: ratio of noise standard deviation to the sensitivity.
    return_type: 'original' or 'custom', determines if summed updates should be included too ('custom')
    init_rng: initial jax.random.PRNGKey

  Returns:
    A `GradientTransformation`.
  """
  noise_stds = compute_opt_noise(l2_norms_threshold, base_sensitivity, noise_multiplier)

  def init_fn(params):
    del params
    return optax.DifferentiallyPrivateAggregateState(
        rng_key=init_rng)

  def update_fn(updates, state, params):
    del params
    grads_flat, grads_treedef = jax.tree_flatten(updates)
    batch_size = grads_flat[0].shape[0]

    if any(g.ndim == 0 or batch_size != g.shape[0] for g in grads_flat):
      raise ValueError(
          'Unlike other transforms, `dp_aggregate` expects'
          ' `updates` to have a batch dimension in the 0th axis. That is, this'
          ' function expects per-example gradients as input.')

    new_key, *rngs = jax.random.split(state.rng_key, len(grads_flat) + 1)
    rng_tree = jax.tree_unflatten(grads_treedef, rngs)

    clipped_updates = clip_by_norm(updates, l2_norms_threshold)
    summed_updates = jax.tree_map(
        lambda g: jnp.sum(g, axis=0),
        clipped_updates)
    noise = jax.tree_map(
        lambda g, std, rng: (std * jax.random.normal(rng, g.shape, g.dtype)),
        summed_updates, noise_stds, rng_tree)
    noisy_updates = jax.tree_map(lambda g, noise: (g + noise), summed_updates,
                                 noise)
    if return_type == 'original':
      return (noisy_updates,
              optax.DifferentiallyPrivateAggregateState(rng_key=new_key))
    else:
      return ((summed_updates, noisy_updates),
              optax.DifferentiallyPrivateAggregateState(rng_key=new_key))

  return optax.GradientTransformation(init_fn, update_fn)


def dpsgd(learning_rate, l2_norms_threshold,
          base_sensitivity, noise_multiplier,
          init_rng, momentum,
          nesterov):
  """A differentially-private version of SGD."""
  return optax.chain(
      dp_aggregate(l2_norms_threshold, base_sensitivity, noise_multiplier,
                   init_rng), optax.sgd(learning_rate, momentum, nesterov))


# def dpadam(learning_rate, l2_norms_threshold,
#            base_sensitivity, noise_multiplier,
#            init_rng):
#   """A differentially-private version of Adam."""
#   return optax.chain(
#       dp_aggregate(l2_norms_threshold, base_sensitivity, noise_multiplier,
#                    init_rng), optax.adam(learning_rate))

def dpadam(learning_rate, b1, eps, l2_norms_threshold,
           base_sensitivity, noise_multiplier,
           init_rng):
  """A differentially-private version of Adam."""
  b2 = 1 - (1 - b1)**2
  return optax.chain(
      dp_aggregate(l2_norms_threshold, base_sensitivity, noise_multiplier,
                   init_rng, return_type='custom'), adam(learning_rate, b1, b2, eps))


def dpadamcorr(batch_size, learning_rate, b1, eps_root, l2_norms_threshold,
               base_sensitivity, noise_multiplier, init_rng):
  """A differentially-private version of Adam Corr."""
  b2 = 1 - (1 - b1)**2
  sigmas = compute_opt_noise(l2_norms_threshold, base_sensitivity, noise_multiplier)
  return optax.chain(
      dp_aggregate(l2_norms_threshold, base_sensitivity, noise_multiplier, init_rng,
                   return_type='custom'), adamcorr(sigmas, learning_rate, b1, b2, 0, eps_root))
