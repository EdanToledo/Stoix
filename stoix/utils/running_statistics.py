# flake8: noqa
# type: ignore
"""Utility functions to compute running statistics.
Taken and modified from Acme https://github.com/google-deepmind/acme/blob/master/acme/jax/running_statistics.py"""

import dataclasses
from typing import Any, List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tree

Path = Union[Tuple[Any, ...], chex.ArrayTree]
"""Path in a nested structure.

  A path is a tuple of indices (normally strings for maps and integers for
  arrays and tuples) that uniquely identifies a subtree in the nested structure.
  See
  https://tree.readthedocs.io/en/latest/api.html#tree.map_structure_with_path
  for more details.
"""


def fast_map_structure(func, *structure):
    """Faster map_structure implementation which skips some error checking."""
    flat_structure = (tree.flatten(s) for s in structure)
    entries = zip(*flat_structure)
    # Arbitrarily choose one of the structures of the original sequence (the last)
    # to match the structure for the flattened sequence.
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def fast_map_structure_with_path(func, *structure):
    """Faster map_structure_with_path implementation."""
    head_entries_with_path = tree.flatten_with_path(structure[0])
    if len(structure) > 1:
        tail_entries = (tree.flatten(s) for s in structure[1:])
        entries_with_path = [e[0] + e[1:] for e in zip(head_entries_with_path, *tail_entries)]
    else:
        entries_with_path = head_entries_with_path
    # Arbitrarily choose one of the structures of the original sequence (the last)
    # to match the structure for the flattened sequence.
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries_with_path])


def _is_prefix(a: Path, b: Path) -> bool:
    """Returns whether `a` is a prefix of `b`."""
    return b[: len(a)] == a


def _zeros_like(nest: chex.ArrayTree, dtype: chex.ArrayDType = None) -> chex.ArrayTree:
    return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def _ones_like(nest: chex.ArrayTree, dtype: chex.ArrayDType = None) -> chex.ArrayTree:
    return jax.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


@chex.dataclass(frozen=True)
class NestedMeanStd:
    """A container for running statistics (mean, std) of possibly nested data."""

    mean: chex.ArrayTree
    std: chex.ArrayTree


@chex.dataclass(frozen=True)
class RunningStatisticsState(NestedMeanStd):
    """Full state of running statistics computation."""

    count: Union[int, jnp.ndarray]
    summed_variance: chex.ArrayTree


@dataclasses.dataclass(frozen=True)
class NestStatisticsConfig:
    """Specifies how to compute statistics for Nests with the same structure.

    Attributes:
      paths: A sequence of Nest paths to compute statistics for. If there is a
        collision between paths (one is a prefix of the other), the shorter path
        takes precedence.
    """

    paths: Tuple[Path, ...] = ((),)


def _is_path_included(config: NestStatisticsConfig, path: Path) -> bool:
    """Returns whether the path is included in the config."""
    # A path is included in the config if it corresponds to a tree node that
    # belongs to a subtree rooted at the node corresponding to some path in
    # the config.
    return any(_is_prefix(config_path, path) for config_path in config.paths)


def init_state(nest: chex.ArrayTree) -> RunningStatisticsState:
    """Initializes the running statistics for the given nested structure."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    return RunningStatisticsState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        count=jnp.float32(0.0),
        mean=_zeros_like(nest, dtype=dtype),
        summed_variance=_zeros_like(nest, dtype=dtype),
        # Initialize with ones to make sure normalization works correctly
        # in the initial state.
        std=_ones_like(nest, dtype=dtype),
    )


def _validate_batch_shapes(
    batch: chex.ArrayTree, reference_sample: chex.ArrayTree, batch_dims: Tuple[int, ...]
) -> None:
    """Verifies shapes of the batch leaves against the reference sample.

    Checks that batch dimensions are the same in all leaves in the batch.
    Checks that non-batch dimensions for all leaves in the batch are the same
    as in the reference sample.

    Arguments:
      batch: the nested batch of data to be verified.
      reference_sample: the nested array to check non-batch dimensions.
      batch_dims: a Tuple of indices of batch dimensions in the batch shape.

    Returns:
      None.
    """

    def validate_node_shape(reference_sample: jnp.ndarray, batch: jnp.ndarray) -> None:
        expected_shape = batch_dims + reference_sample.shape
        assert batch.shape == expected_shape, f"{batch.shape} != {expected_shape}"

    fast_map_structure(validate_node_shape, reference_sample, batch)


def update(
    state: RunningStatisticsState,
    batch: chex.ArrayTree,
    *,
    config: NestStatisticsConfig = NestStatisticsConfig(),
    weights: Optional[jnp.ndarray] = None,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
    pmap_axis_name: Optional[Union[str, List[str]]] = None,
    validate_shapes: bool = True,
) -> RunningStatisticsState:
    """Updates the running statistics with the given batch of data.

    Note: data batch and state elements (mean, etc.) must have the same structure.

    Note: by default will use int32 for counts and float32 for accumulated
    variance. This results in an integer overflow after 2^31 data points and
    degrading precision after 2^24 batch updates or even earlier if variance
    updates have large dynamic range.
    To improve precision, consider setting jax_enable_x64 to True, see
    https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

    Arguments:
      state: The running statistics before the update.
      batch: The data to be used to update the running statistics.
      config: The config that specifies which leaves of the nested structure
        should the running statistics be computed for.
      weights: Weights of the batch data. Should match the batch dimensions.
        Passing a weight of 2. should be equivalent to updating on the
        corresponding data point twice.
      std_min_value: Minimum value for the standard deviation.
      std_max_value: Maximum value for the standard deviation.
      pmap_axis_name: Name or list of names of the pmapped axis, if any.
      validate_shapes: If true, the shapes of all leaves of the batch will be
        validated. Enabled by default. Doesn't impact performance when jitted.

    Returns:
      Updated running statistics.
    """
    # We require exactly the same structure to avoid issues when flattened
    # batch and state have different order of elements.
    tree.assert_same_structure(batch, state.mean)
    batch_shape = tree.flatten(batch)[0].shape
    # We assume the batch dimensions always go first.
    batch_dims = batch_shape[: len(batch_shape) - tree.flatten(state.mean)[0].ndim]
    batch_axis = range(len(batch_dims))
    if weights is None:
        step_increment = np.prod(batch_dims)
    else:
        step_increment = jnp.sum(weights)
    if pmap_axis_name is not None:
        if isinstance(pmap_axis_name, str):
            pmap_axis_name = [pmap_axis_name]
        for axis_name in pmap_axis_name:
            step_increment = jax.lax.psum(step_increment, axis_name=axis_name)
    count = state.count + step_increment

    # Validation is important. If the shapes don't match exactly, but are
    # compatible, arrays will be silently broadcasted resulting in incorrect
    # statistics.
    if validate_shapes:
        if weights is not None:
            if weights.shape != batch_dims:
                raise ValueError(f"{weights.shape} != {batch_dims}")
        _validate_batch_shapes(batch, state.mean, batch_dims)

    def _compute_node_statistics(
        path: Path, mean: jnp.ndarray, summed_variance: jnp.ndarray, batch: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        assert isinstance(mean, jnp.ndarray), type(mean)
        assert isinstance(summed_variance, jnp.ndarray), type(summed_variance)
        if not _is_path_included(config, path):
            # Return unchanged.
            return mean, summed_variance
        # The mean and the sum of past variances are updated with Welford's
        # algorithm using batches (see https://stackoverflow.com/q/56402955).
        diff_to_old_mean = batch - mean
        if weights is not None:
            expanded_weights = jnp.reshape(
                weights, list(weights.shape) + [1] * (batch.ndim - weights.ndim)
            )
            diff_to_old_mean = diff_to_old_mean * expanded_weights
        mean_update = jnp.sum(diff_to_old_mean, axis=batch_axis) / count
        if pmap_axis_name is not None:
            for axis_name in pmap_axis_name:
                mean_update = jax.lax.psum(mean_update, axis_name=axis_name)
        mean = mean + mean_update

        diff_to_new_mean = batch - mean
        variance_update = diff_to_old_mean * diff_to_new_mean
        variance_update = jnp.sum(variance_update, axis=batch_axis)
        if pmap_axis_name is not None:
            for axis_name in pmap_axis_name:
                variance_update = jax.lax.psum(variance_update, axis_name=axis_name)
        summed_variance = summed_variance + variance_update
        return mean, summed_variance

    updated_stats = fast_map_structure_with_path(
        _compute_node_statistics, state.mean, state.summed_variance, batch
    )
    # map_structure_up_to is slow, so shortcut if we know the input is not
    # structured.
    if isinstance(state.mean, jnp.ndarray):
        mean, summed_variance = updated_stats
    else:
        # Reshape the updated stats from `nest(mean, summed_variance)` to
        # `nest(mean), nest(summed_variance)`.
        mean, summed_variance = [
            tree.map_structure_up_to(state.mean, lambda s, i=idx: s[i], updated_stats)
            for idx in range(2)
        ]

    def compute_std(path: Path, summed_variance: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(summed_variance, jnp.ndarray)
        if not _is_path_included(config, path):
            return std
        # Summed variance can get negative due to rounding errors.
        summed_variance = jnp.maximum(summed_variance, 0)
        std = jnp.sqrt(summed_variance / count)
        std = jnp.clip(std, std_min_value, std_max_value)
        return std

    std = fast_map_structure_with_path(compute_std, summed_variance, state.std)

    return RunningStatisticsState(count=count, mean=mean, summed_variance=summed_variance, std=std)


def normalize(
    batch: chex.ArrayTree, mean_std: NestedMeanStd, max_abs_value: Optional[float] = None
) -> chex.ArrayTree:
    """Normalizes data using running statistics."""

    def normalize_leaf(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        # Only normalize inexact types.
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        data = (data - mean) / std
        if max_abs_value is not None:
            # TODO(b/124318564): remove pylint directive
            data = jnp.clip(
                data, -max_abs_value, +max_abs_value
            )  # pylint: disable=invalid-unary-operand-type
        return data

    return fast_map_structure(normalize_leaf, batch, mean_std.mean, mean_std.std)


def denormalize(batch: chex.ArrayTree, mean_std: NestedMeanStd) -> chex.ArrayTree:
    """Denormalizes values in a nested structure using the given mean/std.

    Only values of inexact types are denormalized.
    See https://numpy.org/doc/stable/_images/dtype-hierarchy.png for Numpy type
    hierarchy.

    Args:
      batch: a nested structure containing batch of data.
      mean_std: mean and standard deviation used for denormalization.

    Returns:
      Nested structure with denormalized values.
    """

    def denormalize_leaf(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
        # Only denormalize inexact types.
        if not np.issubdtype(data.dtype, np.inexact):
            return data
        return data * std + mean

    return fast_map_structure(denormalize_leaf, batch, mean_std.mean, mean_std.std)


@dataclasses.dataclass(frozen=True)
class NestClippingConfig:
    """Specifies how to clip Nests with the same structure.

    Attributes:
      path_map: A map that specifies how to clip values in Nests with the same
        structure. Keys correspond to paths in the nest. Values are maximum
        absolute values to use for clipping. If there is a collision between paths
        (one path is a prefix of the other), the behavior is undefined.
    """

    path_map: Tuple[Tuple[Path, float], ...] = ()


def get_clip_config_for_path(config: NestClippingConfig, path: Path) -> NestClippingConfig:
    """Returns the config for a subtree from the leaf defined by the path."""
    # Start with an empty config.
    path_map = []
    for map_path, max_abs_value in config.path_map:
        if _is_prefix(map_path, path):
            return NestClippingConfig(path_map=(((), max_abs_value),))
        if _is_prefix(path, map_path):
            path_map.append((map_path[len(path) :], max_abs_value))
    return NestClippingConfig(path_map=tuple(path_map))


def clip(batch: chex.ArrayTree, clipping_config: NestClippingConfig) -> chex.ArrayTree:
    """Clips the batch."""

    def max_abs_value_for_path(path: Path, x: jnp.ndarray) -> Optional[float]:
        del x  # Unused, needed by interface.
        return next(
            (
                max_abs_value
                for clipping_path, max_abs_value in clipping_config.path_map
                if _is_prefix(clipping_path, path)
            ),
            None,
        )

    max_abs_values = fast_map_structure_with_path(max_abs_value_for_path, batch)

    def clip_leaf(data: jnp.ndarray, max_abs_value: Optional[float]) -> jnp.ndarray:
        if max_abs_value is not None:
            # TODO(b/124318564): remove pylint directive
            data = jnp.clip(
                data, -max_abs_value, +max_abs_value
            )  # pylint: disable=invalid-unary-operand-type
        return data

    return fast_map_structure(clip_leaf, batch, max_abs_values)


@dataclasses.dataclass(frozen=True)
class NestNormalizationConfig:
    """Specifies how to normalize Nests with the same structure.

    Attributes:
      stats_config: A config that defines how to compute running statistics to be
        used for normalization.
      clip_config: A config that defines how to clip normalized values.
    """

    stats_config: NestStatisticsConfig = NestStatisticsConfig()
    clip_config: NestClippingConfig = NestClippingConfig()
