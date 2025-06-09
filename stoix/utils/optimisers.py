### RMSProp implementation for PyTorch-style RMSProp
# see https://github.com/deepmind/optax/issues/532#discussioncomment-1676371843
from typing import Optional

import jax
import jax.numpy as jnp
from optax import update_moment_per_elem_norm
from optax._src import base, combine, transform
from optax._src.base import ScalarOrSchedule
from optax._src.transform import ScaleByRmsState, scale_by_learning_rate


def scale_by_rms_pytorch_style(
    decay: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.0
) -> base.GradientTransformation:
    """PyTorch-style RMSProp scaling transformation.

    See https://github.com/deepmind/optax/issues/532#discussioncomment-1676371843
    """

    def init_fn(params):
        nu = jax.tree_util.tree_map(
            lambda n: jnp.full_like(n, initial_scale), params
        )  # second moment
        return ScaleByRmsState(nu=nu)

    def update_fn(updates, state, params=None):
        del params
        nu = update_moment_per_elem_norm(updates, state.nu, decay, 2)
        updates = jax.tree_util.tree_map(lambda g, n: g / (jax.lax.sqrt(n) + eps), updates, nu)
        return updates, ScaleByRmsState(nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def rmsprop_pytorch_style(
    learning_rate: ScalarOrSchedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    momentum: Optional[float] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    """PyTorch-style RMSProp optimizer."""
    return combine.chain(
        scale_by_rms_pytorch_style(decay=decay, eps=eps, initial_scale=initial_scale),
        scale_by_learning_rate(learning_rate),
        transform.trace(decay=momentum, nesterov=nesterov)
        if momentum is not None
        else base.identity(),
    )
