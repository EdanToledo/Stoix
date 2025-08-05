import time
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from jax._src.pjit import JitWrapped


def scale_gradient(g: chex.Array, scale: float = 1) -> chex.Array:
    """Scales the gradient of `g` by `scale` but keeps the original value unchanged."""
    return g * scale + jax.lax.stop_gradient(g) * (1.0 - scale)


def count_parameters(params: chex.ArrayTree) -> int:
    """Counts the number of parameters in a parameter tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def ndim_at_least(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
    """Check if the number of dimensions of `x` is at least `num_dims`."""
    if not (isinstance(x, jax.Array) or isinstance(x, np.ndarray)):
        x = jnp.asarray(x)
    return x.ndim >= num_dims


def merge_leading_dims(x: chex.Array, num_dims: chex.Numeric) -> chex.Array:
    """Merge leading dimensions.

    Note:
        This implementation is a generic function for merging leading dimensions
        extracted from Haiku.
        For the original implementation, please refer to the following link:
        (https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/basic.py#L207)
    """
    # Don't merge if there aren't dimensions to merge.
    if not ndim_at_least(x, num_dims):
        return x

    new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
    return x.reshape(new_shape)


@jax.jit
def unreplicate_n_dims(x: chex.ArrayTree, unreplicate_depth: int = 2) -> chex.ArrayTree:
    """Unreplicates a pytree by removing the first `unreplicate_depth` axes.

    This function takes a pytree and removes some number of axes, associated with parameter
    duplication for running multiple updates across devices and in parallel with `vmap`.
    This is typically one axis for device replication, and one for the `update batch size`.
    """
    return jax.tree_util.tree_map(lambda x: x[(0,) * unreplicate_depth], x)  # type: ignore


@jax.jit
def unreplicate_batch_dim(x: chex.ArrayTree) -> chex.ArrayTree:
    """Unreplicated just the update batch dimension.
    (The dimension that is vmapped over when acting and learning)

    In stoix's case it is always the second dimension, after the device dimension.
    We simply take element 0 as the params are identical across this dimension.
    """
    return jax.tree_util.tree_map(lambda x: x[:, 0, ...], x)  # type: ignore


def aot_compile(
    fn_to_compile: JitWrapped,
    fn_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Compiles a JAX function ahead-of-time and prints benchmarking information.

    This function generalizes the process of tracing, lowering, and compiling
    a JAX function, making it reusable for different functions like learners,
    evaluators, etc.

    Args:
        fn_to_compile: The jitted or pmapped JAX function to be compiled.
        fn_name: A descriptive name for the function (e.g., "Learner", "Evaluator")
                 used for printing logs.
        *args: Positional arguments to be passed to the function for tracing.
        **kwargs: Keyword arguments to be passed to the function for tracing.

    Returns:
        The compiled function artifact.
    """
    print(f"{Fore.YELLOW}Compiling {fn_name} function ahead of time...{Style.RESET_ALL}")
    start_time = time.time()

    # Use the provided args and kwargs to trace the function
    traced_fn = fn_to_compile.trace(*args, **kwargs)
    lowered_fn = traced_fn.lower()
    compiled_fn = lowered_fn.compile()

    elapsed = time.time() - start_time

    # Extract cost analysis safely
    cost_analysis = compiled_fn.cost_analysis()
    flops_estimate = cost_analysis.get("flops", 0)

    print(
        f"{Fore.GREEN}{Style.BRIGHT}{fn_name} function compiled in "
        f"{elapsed:.2f} seconds.{Style.RESET_ALL}"
    )
    if flops_estimate > 0:
        print(
            f"{Fore.GREEN}{Style.BRIGHT}{fn_name} function FLOPs: "
            f"{flops_estimate / 1e9:.3f} GFlops.{Style.RESET_ALL}"
        )

    return compiled_fn
