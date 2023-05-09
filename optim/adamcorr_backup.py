import jax
import jax.numpy as jnp
import os
import optax
from optax._src import base
from optax._src import utils
from optax._src import numerics
from optax._src import combine
from optax._src.transform import ScaleByAdamState, update_moment, update_moment_per_elem_norm, bias_correction
from optax._src.alias import _scale_by_learning_rate
from typing import Any, Callable, NamedTuple, Optional, Union
import chex

LOGGING = True
# LOGGING = False

def get_summary_stats(a, prefix, sqrt=False):
    a_flattened = tree_flatten_1dim(a)
    if sqrt:
        a_flattened = jnp.sqrt(a_flattened)
    a_min = jnp.min(a_flattened)
    a_max = jnp.max(a_flattened)
    a_mean = jnp.mean(a_flattened)
    a_median = jnp.median(a_flattened)
    a_q25 = jnp.quantile(a_flattened, q=0.25)
    a_q75 = jnp.quantile(a_flattened, q=0.75)
    stats = {'min': a_min, 'max': a_max, 'mean': a_mean, 'median': a_median, 'q25': a_q25, 'q75': a_q75}
    return add_prefix(stats, prefix)

def add_prefix(d, prefix):
    new_dict = {}
    for key, value in d.items():
        new_dict[prefix+'_'+key] = value
    return new_dict

def tree_flatten_1dim(tree):
    tree_flat, _ = jax.tree_flatten(tree)
    return jnp.concatenate([i.flatten() for i in tree_flat])

class ScaleByAdamStateCorr(NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_corr: base.Updates
    count_tree: None  # individual param record of update count


class ScaleByAdamStateCorrLong(NamedTuple):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_corr: base.Updates
    count_tree: None  # individual param record of update count
    mu_clean: base.Updates
    nu_clean: base.Updates
    summary_stats: dict


def scale_by_adam_corr(
        batch_size: int,
        noise_multipliers: float,
        l2_norms_threshold: float,
        b1: float,
        b2: float,
        eps: float,
        eps_root: float,
        mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        nu_corr = jax.tree_util.tree_map(jnp.zeros_like, params)  # corrected second moment
        mu_clean = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu_clean = jax.tree_util.tree_map(jnp.zeros_like, params)
        summary_stats={}
        return ScaleByAdamStateCorrLong(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_corr=nu_corr, count_tree=None,
            mu_clean=mu_clean, nu_clean=nu_clean, summary_stats=summary_stats,
        )

    def update_fn(updates, state, params=None):
        del params
        clean_updates, noised_updates = updates
        mu = update_moment(noised_updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(noised_updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat_uncorr = bias_correction(nu, b2, count_inc)
        # corr for noise variance
        # sum_i=1^t x^(t-i) = (x^t-1)/(x-1), multiply by (1-x) = 1-x^t
        noise_errs = jax.tree_map(
            lambda noise_multiplier, l2_norm: (1 / batch_size ** 2) * noise_multiplier ** 2 * l2_norm ** 2 * (1 - b2 ** count_inc),
            noise_multipliers, l2_norms_threshold)
        # # 1- replace small values with eps_root
        nu_corr = jax.tree_map(lambda x, noise_err: jnp.maximum(x - noise_err, eps_root), nu, noise_errs)
        nu_hat = bias_correction(nu_corr, b2, count_inc)
        # nu_corr_orig = jax.tree_map(lambda x: jnp.maximum(x - noise_err, 1e-30), nu)
        # nu_hat_corr_orig = bias_correction(nu_corr_orig, b2, count_inc)
        # updates = jax.tree_util.tree_map(
        #     lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v)), mu_hat, nu_hat)
        mu = utils.cast_tree(mu, mu_dtype)

        # clean states updated using clipped grads
        mu_clean = update_moment(clean_updates, state.mu_clean, b1, 1)
        nu_clean = update_moment_per_elem_norm(clean_updates, state.nu_clean, b2, 2)
        mu_hat_clean = bias_correction(mu_clean, b1, count_inc)
        nu_hat_clean = bias_correction(nu_clean, b2, count_inc)

        if LOGGING:
            summary_stats =  {**get_summary_stats(mu_hat_clean, 'mt_clean'), 
                              **get_summary_stats(mu_hat, 'mt_noised'),
                              **get_summary_stats(nu_hat_clean, 'vt_clean', sqrt=True),
                              **get_summary_stats(nu_hat_uncorr, 'vt_noised', sqrt=True),
                              **get_summary_stats(nu_hat, 'vt_corr', sqrt=True),}
        else:
            summary_stats = {}

        return updates, ScaleByAdamStateCorrLong(
            count=count_inc, mu=mu, nu=nu, nu_corr=nu_corr, count_tree=None,
            mu_clean=mu_clean, nu_clean=nu_clean, summary_stats=summary_stats
        )

    return base.GradientTransformation(init_fn, update_fn)

def adamcorr(
    batch_size: int,
    noise_multipliers: list,
    l2_norms_threshold: list,
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_adam_corr(
            batch_size=batch_size, noise_multipliers=noise_multipliers, l2_norms_threshold=l2_norms_threshold,
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype),
        _scale_by_learning_rate(learning_rate),
    )