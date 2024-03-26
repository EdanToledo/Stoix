import enum
import functools
from typing import Callable, Sequence, Union

import chex
import flax.linen as nn
import jax

from stoix.networks.utils import parse_activation_fn

InnerOp = Union[nn.Module, Callable[..., chex.Array]]
MakeInnerOp = Callable[..., InnerOp]
NonLinearity = Callable[[chex.Array], chex.Array]


class ResidualBlock(nn.Module):
    make_inner_op: MakeInnerOp
    non_linearity: NonLinearity = jax.nn.relu
    use_layer_norm: bool = False

    def setup(self) -> None:
        self.inner_op1 = self.make_inner_op()
        self.inner_op2 = self.make_inner_op()

        if self.use_layer_norm:
            self.layernorm1 = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)
            self.layernorm2 = nn.LayerNorm(use_scale=True, use_bias=True, epsilon=1e-6)

    def __call__(self, x: chex.Array) -> chex.Array:
        output = x

        # First layer in residual block.
        if self.use_layer_norm:
            output = self.layernorm1(output)
        output = self.non_linearity(output)
        output = self.inner_op1(output)

        # Second layer in residual block.
        if self.use_layer_norm:
            output = self.layernorm2(output)
        output = self.non_linearity(output)
        output = self.inner_op2(output)
        return x + output


class DownsamplingStrategy(enum.Enum):
    AVG_POOL = "avg_pool"
    CONV_MAX = "conv+max"  # Used in IMPALA
    LAYERNORM_RELU_CONV = "layernorm+relu+conv"  # Used in MuZero
    CONV = "conv"


def make_downsampling_layer(
    strategy: Union[str, DownsamplingStrategy],
    output_channels: int,
) -> nn.Module:
    """Returns a sequence of modules corresponding to the desired downsampling."""
    strategy = DownsamplingStrategy(strategy)

    if strategy is DownsamplingStrategy.AVG_POOL:
        return lambda x: nn.avg_pool(x, window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")

    elif strategy is DownsamplingStrategy.CONV:
        return nn.Sequential(
            [
                nn.Conv(
                    features=output_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    kernel_init=nn.initializers.truncated_normal(1e-2),
                ),
            ]
        )

    elif strategy is DownsamplingStrategy.LAYERNORM_RELU_CONV:
        return nn.Sequential(
            [
                nn.LayerNorm(
                    reduction_axes=(-3, -2, -1), use_scale=True, use_bias=True, epsilon=1e-6
                ),
                jax.nn.relu,
                nn.Conv(
                    features=output_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    kernel_init=nn.initializers.truncated_normal(1e-2),
                ),
            ]
        )

    elif strategy is DownsamplingStrategy.CONV_MAX:
        return nn.Sequential(
            [
                nn.Conv(features=output_channels, kernel_size=(3, 3), strides=(1, 1)),
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME"),
            ]
        )
    else:
        raise ValueError(
            "Unrecognized downsampling strategy. Expected one of"
            f" {[strategy.value for strategy in DownsamplingStrategy]}"
            f" but received {strategy}."
        )


class VisualResNetTorso(nn.Module):
    """ResNetTorso for visual inputs, inspired by the IMPALA paper."""

    channels_per_group: Sequence[int] = (16, 32, 32)
    blocks_per_group: Sequence[int] = (2, 2, 2)
    downsampling_strategies: Sequence[DownsamplingStrategy] = (DownsamplingStrategy.CONV,) * 3
    use_layer_norm: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:

        if observation.ndim > 4:
            return nn.batch_apply.BatchApply(self.__call__)(observation)

        assert (
            observation.ndim == 4
        ), f"Expected inputs to have shape [B, H, W, C] but got shape {observation.shape}."

        output = observation
        channels_blocks_strategies = zip(
            self.channels_per_group, self.blocks_per_group, self.downsampling_strategies
        )

        for _, (num_channels, num_blocks, strategy) in enumerate(channels_blocks_strategies):
            output = make_downsampling_layer(strategy, num_channels)(output)

            for _ in range(num_blocks):
                output = ResidualBlock(
                    make_inner_op=functools.partial(
                        nn.Conv, features=num_channels, kernel_size=(3, 3)
                    ),
                    use_layer_norm=self.use_layer_norm,
                    non_linearity=parse_activation_fn(self.activation),
                )(output)

        return output.reshape(*observation.shape[:-3], -1)


class ResNetTorso(nn.Module):
    """ResNetTorso for Vector-based inputs"""

    hidden_units_per_group: Sequence[int] = (256, 256, 256)
    blocks_per_group: Sequence[int] = (2, 2, 2)
    use_layer_norm: bool = False
    activation: str = "relu"

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        output = observation
        hidden_units_blocks = zip(self.hidden_units_per_group, self.blocks_per_group)

        for _, (num_hidden_units, num_blocks) in enumerate(hidden_units_blocks):
            output = nn.Dense(features=num_hidden_units)(output)
            output = parse_activation_fn(self.activation)(output)
            for _ in range(num_blocks):
                output = ResidualBlock(
                    make_inner_op=functools.partial(nn.Dense, features=num_hidden_units),
                    use_layer_norm=self.use_layer_norm,
                    non_linearity=parse_activation_fn(self.activation),
                )(output)

        return output
