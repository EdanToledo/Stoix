from typing import List, Optional, Tuple

import chex
from flax import linen as nn
from jax import numpy as jnp

RecurrentState = chex.Array
Reset = chex.Array
Timestep = chex.Array
InputEmbedding = chex.Array
Inputs = Tuple[InputEmbedding, Reset]
ScanInput = chex.Array
