"""Utility functions to compute running statistics.
Taken and modified from
Acme https://github.com/google-deepmind/acme/blob/master/acme/jax/running_statistics.py"""

import dataclasses
import types
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import chex
import jax
import jax.numpy as jnp
import numpy as np
import tree
from jax import Array

NestedPath = Union[Tuple[Any, ...], chex.ArrayTree]
"""Path in a nested structure.

  A path is a tuple of indices (normally strings for maps and integers for
  arrays and tuples) that uniquely identifies a subtree in the nested structure.
  See
  https://tree.readthedocs.io/en/latest/api.html#tree.map_structure_with_path
  for more details.
"""

# Type variable for generic tree structures
T = TypeVar("T")


def efficient_map_structure(func: Callable[..., T], *structure: chex.ArrayTree) -> chex.ArrayTree:
    """Faster implementation of tree.map_structure which skips some error checking.

    Args:
        func: The function to apply to each leaf in the structure.
        *structure: The nested structures to map over.

    Returns:
        A new nested structure with the same shape as the input structures but
        with the leaves replaced by the results of applying func to the corresponding
        leaves in the input structures.
    """
    flat_structure = (tree.flatten(s) for s in structure)
    entries = zip(*flat_structure)
    # Arbitrarily choose one of the structures of the original sequence (the last)
    # to match the structure for the flattened sequence.
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def efficient_map_structure_with_path(
    func: Callable[..., T], *structure: chex.ArrayTree
) -> chex.ArrayTree:
    """Faster implementation of tree.map_structure_with_path.

    Args:
        func: The function to apply to each leaf in the structure with its path.
        *structure: The nested structures to map over.

    Returns:
        A new nested structure with the same shape as the input structures but
        with the leaves replaced by the results of applying func to the corresponding
        paths and leaves in the input structures.
    """
    head_entries_with_path = tree.flatten_with_path(structure[0])
    if len(structure) > 1:
        tail_entries = (tree.flatten(s) for s in structure[1:])
        entries_with_path = [e[0] + e[1:] for e in zip(head_entries_with_path, *tail_entries)]
    else:
        entries_with_path = head_entries_with_path
    # Arbitrarily choose one of the structures of the original sequence (the last)
    # to match the structure for the flattened sequence.
    return tree.unflatten_as(structure[-1], [func(*x) for x in entries_with_path])


def _is_prefix(a: NestedPath, b: NestedPath) -> bool:
    """Returns whether `a` is a prefix of `b`."""
    return b[: len(a)] == a


def _zeros_like(
    nest: chex.ArrayTree, dtype: Optional[jax.typing.DTypeLike] = None
) -> chex.ArrayTree:
    """Creates a nested structure of zeros with the same shape as the input.

    Args:
        nest: The nested structure to match.
        dtype: The data type to use for the zeros. If None, uses the dtype of the input.

    Returns:
        A nested structure of zeros with the same shape as the input.
    """
    return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def _ones_like(
    nest: chex.ArrayTree, dtype: Optional[jax.typing.DTypeLike] = None
) -> chex.ArrayTree:
    """Creates a nested structure of ones with the same shape as the input.

    Args:
        nest: The nested structure to match.
        dtype: The data type to use for the ones. If None, uses the dtype of the input.

    Returns:
        A nested structure of ones with the same shape as the input.
    """
    return jax.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


@chex.dataclass(frozen=True)
class RunningStatisticsState:
    """Full state of running statistics computation.

    Attributes:
        mean: The running mean of the data.
        std: The running standard deviation of the data.
        count: The number of samples that have been used to compute the statistics.
        summed_variance: The sum of squared differences from the mean.

    """

    mean: chex.ArrayTree
    std: chex.ArrayTree
    count: Union[int, Array]
    summed_variance: chex.ArrayTree


@dataclasses.dataclass(frozen=True)
class NestedStatisticsConfig:
    """Specifies how to compute statistics for nested structures with the same structure.

    Attributes:
      paths: A sequence of paths in the nested structure to compute statistics for.
        If there is a collision between paths (one is a prefix of the other),
        the shorter path takes precedence.
    """

    paths: Tuple[NestedPath, ...] = ((),)


def _is_path_included(config: NestedStatisticsConfig, path: NestedPath) -> bool:
    """Returns whether the path is included in the config.

    A path is included in the config if it corresponds to a tree node that
    belongs to a subtree rooted at the node corresponding to some path in
    the config.

    Args:
        config: The configuration specifying which paths to include.
        path: The path to check.

    Returns:
        Whether the path is included in the config.
    """
    return any(_is_prefix(config_path, path) for config_path in config.paths)


def initialize_statistics(nest: chex.ArrayTree) -> RunningStatisticsState:
    """Initializes the running statistics for the given nested structure.

    Args:
        nest: The nested structure to initialize statistics for.

    Returns:
        A RunningStatisticsState initialized with zeros for means and ones for standard deviations.
    """
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    return RunningStatisticsState(  # type: ignore
        mean=_zeros_like(nest, dtype=dtype),
        # Initialize with ones to make sure normalization works correctly
        # in the initial state.
        std=_ones_like(nest, dtype=dtype),
        count=jnp.float32(0.0),
        summed_variance=_zeros_like(nest, dtype=dtype),
    )


def _validate_batch_shapes(
    batch: chex.ArrayTree, reference_sample: chex.ArrayTree, batch_dims: Tuple[int, ...]
) -> None:
    """Verifies shapes of the batch leaves against the reference sample.

    Checks that batch dimensions are the same in all leaves in the batch.
    Checks that non-batch dimensions for all leaves in the batch are the same
    as in the reference sample.

    Args:
      batch: the nested batch of data to be verified.
      reference_sample: the nested array to check non-batch dimensions.
      batch_dims: a Tuple of indices of batch dimensions in the batch shape.

    Returns:
      None.
    """

    def validate_node_shape(reference_sample: Array, batch: Array) -> None:
        expected_shape = batch_dims + reference_sample.shape
        assert batch.shape == expected_shape, f"{batch.shape} != {expected_shape}"

    efficient_map_structure(validate_node_shape, reference_sample, batch)


# Create default config as module-level variable to avoid B008
_DEFAULT_NESTED_STATISTICS_CONFIG = NestedStatisticsConfig()


def _compute_node_statistics_step(
    path: NestedPath,
    mean: Array,
    summed_variance: Array,
    batch: Array,
    count: Union[int, Array],
    batch_axis: range,
    weights: Optional[Array],
    config: NestedStatisticsConfig,
    pmap_axis_name: Optional[Union[str, List[str]]],
) -> Tuple[Array, Array]:
    """Helper function to compute statistics for a single node."""
    assert isinstance(mean, Array), type(mean)
    assert isinstance(summed_variance, Array), type(summed_variance)
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


def update_statistics(
    state: RunningStatisticsState,
    batch: chex.ArrayTree,
    *,
    config: Optional[NestedStatisticsConfig] = None,
    weights: Optional[Array] = None,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
    pmap_axis_name: Optional[Union[str, List[str]]] = None,
    validate_shapes: bool = True,
) -> RunningStatisticsState:
    """Updates the running statistics with the given batch of data.

    This function implements Welford's online algorithm for computing running statistics.
    It updates the mean and variance estimates based on the new batch of data.

    Note: data batch and state elements (mean, etc.) must have the same structure.

    Note: by default will use int32 for counts and float32 for accumulated
    variance. This results in an integer overflow after 2^31 data points and
    degrading precision after 2^24 batch updates or even earlier if variance
    updates have large dynamic range.
    To improve precision, consider setting jax_enable_x64 to True, see
    https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

    Args:
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
    if config is None:
        config = _DEFAULT_NESTED_STATISTICS_CONFIG

    # We require exactly the same structure to avoid issues when flattened
    # batch and state have different order of elements.
    tree.assert_same_structure(batch, state.mean)
    batch_shape = tree.flatten(batch)[0].shape
    # We assume the batch dimensions always go first.
    batch_dims = batch_shape[: len(batch_shape) - tree.flatten(state.mean)[0].ndim]
    batch_axis = range(len(batch_dims))
    if weights is None:
        step_increment: Union[int, Array] = np.prod(batch_dims)
    else:
        step_increment = jnp.sum(weights)
    if pmap_axis_name is not None:
        pmap_names = [pmap_axis_name] if isinstance(pmap_axis_name, str) else pmap_axis_name
        for axis_name in pmap_names:
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
        path: NestedPath, mean: Array, summed_variance: Array, batch: Array
    ) -> Tuple[Array, Array]:
        return _compute_node_statistics_step(
            path, mean, summed_variance, batch, count, batch_axis, weights, config, pmap_axis_name
        )

    updated_stats = efficient_map_structure_with_path(
        _compute_node_statistics, state.mean, state.summed_variance, batch
    )
    # map_structure_up_to is slow, so shortcut if we know the input is not
    # structured.
    if isinstance(state.mean, Array):
        mean, summed_variance = updated_stats
    else:
        # Reshape the updated stats from `nest(mean, summed_variance)` to
        # `nest(mean), nest(summed_variance)`.
        mean, summed_variance = [
            tree.map_structure_up_to(state.mean, lambda s, i=idx: s[i], updated_stats)
            for idx in range(2)
        ]

    def compute_std(path: NestedPath, summed_variance: Array, std: Array) -> Array:
        assert isinstance(summed_variance, Array)
        if not _is_path_included(config, path):
            return std
        # Summed variance can get negative due to rounding errors.
        summed_variance = jnp.maximum(summed_variance, 0)
        std = jnp.sqrt(summed_variance / count)
        std = jnp.clip(std, std_min_value, std_max_value)
        return std

    std = efficient_map_structure_with_path(compute_std, summed_variance, state.std)

    return RunningStatisticsState(
        mean=mean, std=std, count=count, summed_variance=summed_variance  # type: ignore
    )


def normalize(
    batch: chex.ArrayTree, mean_std: RunningStatisticsState, max_abs_value: Optional[float] = None
) -> chex.ArrayTree:
    """Normalizes data using running statistics.

    For each value in the batch, computes (value - mean) / std.

    Args:
        batch: The nested structure of data to normalize.
        mean_std: The statistics (mean and std) to use for normalization.
        max_abs_value: If provided, the normalized values will be clipped to
            [-max_abs_value, max_abs_value].

    Returns:
        A nested structure with the same shape as the input, but with normalized values.
    """

    def normalize_leaf(data: Array, mean: Array, std: Array) -> Array:
        # Only normalize inexact types.
        if not jnp.issubdtype(data.dtype, jnp.inexact):
            return data
        data = (data - mean) / std
        if max_abs_value is not None:
            data = jnp.clip(data, -max_abs_value, +max_abs_value)
        return data

    return efficient_map_structure(normalize_leaf, batch, mean_std.mean, mean_std.std)


def denormalize(batch: chex.ArrayTree, mean_std: RunningStatisticsState) -> chex.ArrayTree:
    """Denormalizes values in a nested structure using the given mean/std.

    For each value in the batch, computes value * std + mean, which is the inverse
    of the normalization operation.

    Only values of inexact types are denormalized.
    See https://numpy.org/doc/stable/_images/dtype-hierarchy.png for Numpy type
    hierarchy.

    Args:
      batch: a nested structure containing batch of data.
      mean_std: mean and standard deviation used for denormalization.

    Returns:
      Nested structure with denormalized values.
    """

    def denormalize_leaf(data: Array, mean: Array, std: Array) -> Array:
        # Only denormalize inexact types.
        if not np.issubdtype(data.dtype, np.inexact):
            return data
        return data * std + mean

    return efficient_map_structure(denormalize_leaf, batch, mean_std.mean, mean_std.std)


@dataclasses.dataclass(frozen=True)
class NestedClippingConfig:
    """Specifies how to clip values in nested structures with the same structure.

    Attributes:
      path_map: A map that specifies how to clip values in nested structures.
        Keys correspond to paths in the structure. Values are maximum
        absolute values to use for clipping. If there is a collision between paths
        (one path is a prefix of the other), the behavior is undefined.
    """

    path_map: Tuple[Tuple[NestedPath, float], ...] = ()


def get_clip_config_for_path(
    config: NestedClippingConfig, path: NestedPath
) -> NestedClippingConfig:
    """Returns the clipping config for a subtree defined by the path.

    Args:
        config: The clipping configuration.
        path: The path to get the config for.

    Returns:
        A new clipping config for the subtree.
    """
    # Start with an empty config.
    path_map: List[Tuple[NestedPath, float]] = []
    for map_path, max_abs_value in config.path_map:
        if _is_prefix(map_path, path):
            return NestedClippingConfig(path_map=(((), max_abs_value),))
        if _is_prefix(path, map_path):
            path_map.append((map_path[len(path) :], max_abs_value))
    return NestedClippingConfig(path_map=tuple(path_map))


def clip_values(batch: chex.ArrayTree, clipping_config: NestedClippingConfig) -> chex.ArrayTree:
    """Clips values in a nested structure according to the provided configuration.

    Args:
        batch: The nested structure of data to clip.
        clipping_config: Configuration specifying how to clip values.

    Returns:
        A nested structure with the same shape as the input, but with clipped values.
    """

    def max_abs_value_for_path(path: NestedPath, x: Array) -> Optional[float]:
        del x  # Unused, needed by interface.
        return next(
            (
                max_abs_value
                for clipping_path, max_abs_value in clipping_config.path_map
                if _is_prefix(clipping_path, path)
            ),
            None,
        )

    max_abs_values = efficient_map_structure_with_path(max_abs_value_for_path, batch)

    def clip_leaf(data: Array, max_abs_value: Optional[float]) -> Array:
        if max_abs_value is not None:
            data = jnp.clip(data, -max_abs_value, +max_abs_value)
        return data

    return efficient_map_structure(clip_leaf, batch, max_abs_values)


@dataclasses.dataclass(frozen=True)
class NestedNormalizationConfig:
    """Specifies how to normalize nested structures with the same structure.

    Attributes:
      stats_config: A config that defines how to compute running statistics to be
        used for normalization.
      clip_config: A config that defines how to clip normalized values.
    """

    stats_config: NestedStatisticsConfig = NestedStatisticsConfig()
    clip_config: NestedClippingConfig = NestedClippingConfig()


# Define type variable for NamedTuple subclasses
NT = TypeVar("NT", bound=NamedTuple)


def add_field_to_state(
    base_class: Type[NT], extra_field_name: str, extra_field_type: Type[Any]
) -> Type[NamedTuple]:
    """Create a new NamedTuple class by extending an existing NamedTuple class with an additional field.

    This function creates a specialized NamedTuple class that behaves in a special way:
    1. It includes all fields from the base class plus the extra field
    2. The extra field is accessible as a normal attribute (obj.extra_field_name)
    3. When unpacking the object (a, b, c = obj), only the original fields are included
    4. The _replace method works with all fields including the extra field
    5. copy.deepcopy works correctly

    This approach is useful when you need to attach metadata or state to an existing
    NamedTuple without affecting its unpacking behavior in existing code.

    Args:
        base_class: The NamedTuple class to extend (not an instance).
        extra_field_name: The name of the additional field to add to the new class.
        extra_field_type: The type of the additional field to add to the new class.

    Returns:
        A new NamedTuple class with all fields from base_class plus the extra field.
        When unpacking, only the original fields from base_class will be included.
    """
    # Get field names and annotations from the base class
    fields = getattr(base_class, "_fields", ())
    annotations = getattr(base_class, "__annotations__", {})

    # Create a new class with all original fields plus the new one
    new_fields: Dict[str, Type[Any]] = {field: annotations.get(field, Any) for field in fields}
    new_fields[extra_field_name] = extra_field_type

    # Create the new NamedTuple class with all the fields
    class_name = f"Enhanced{base_class.__name__}"

    # Create the new class
    result_class = types.new_class(
        class_name,
        (NamedTuple,),
        {},
        lambda ns: ns.update(
            {
                "__annotations__": new_fields,
                "__doc__": f"Version of {base_class.__name__} with extra field '{extra_field_name}'.",
                "__module__": base_class.__module__,
            }
        ),
    )

    # Need to preserve the _fields for proper replacement
    all_fields = getattr(result_class, "_fields", ())

    # Override __iter__ to only include base fields when unpacking
    def custom_iter(self: Any) -> Any:
        """Iterates only through the original fields, not the extra field."""
        return iter(getattr(self, field) for field in fields)

    result_class.__iter__ = custom_iter  # type: ignore

    # Fix the _replace method to ensure it still works
    def custom_replace(self: Any, **kwargs: Any) -> Any:
        """Custom replacement that works with the modified structure.

        Allows replacing any field (original or extra) while preserving the
        custom unpacking behavior.
        """
        # Get a dictionary of all current values
        current_values: Dict[str, Any] = {}
        for field in all_fields:
            current_values[field] = getattr(self, field)

        # Update with new values
        current_values.update(kwargs)

        # Create new instance with updated values
        return type(self)(**current_values)

    result_class._replace = custom_replace  # type: ignore

    # Add __getnewargs__ to support copy.deepcopy
    def custom_getnewargs(self: Any) -> tuple:
        """Return all field values for pickling/copying, including the extra field."""
        return tuple(getattr(self, field) for field in all_fields)

    result_class.__getnewargs__ = custom_getnewargs  # type: ignore

    return result_class


def create_with_running_statistics(state: NT, running_statistics: RunningStatisticsState) -> Any:
    """Add running statistics to a state instance.

    This function takes an existing state instance and attaches running statistics
    to it, creating a new version of the state class. The state
    has special unpacking behavior - when unpacked, only the original state fields
    are included, but the running_statistics field can be accessed directly.

    Key behaviors:
    1. The running_statistics field is accessible as state.running_statistics
    2. When unpacking (a, b, c = state), only the original fields are included
    3. All NamedTuple methods like _replace work on all fields
    4. copy.deepcopy works correctly

    Args:
        state: An instance of a NamedTuple state.
        running_statistics: The RunningStatisticsState instance to add to the state.

    Returns:
        A new instance of an enhanced state class that includes the running statistics
        field but excludes it from unpacking operations.
    """
    cls_type = type(state)
    new_cls_type = add_field_to_state(cls_type, "running_statistics", type(running_statistics))
    state_dict = state._asdict()
    state_dict["running_statistics"] = running_statistics
    return new_cls_type(**state_dict)  # type: ignore
