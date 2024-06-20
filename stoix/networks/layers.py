from typing import List, Tuple

import chex
from flax import linen as nn
from stoix.networks.utils import parse_activation_fn


class StackedRNN(nn.Module):
    """
    A class representing a stacked recurrent neural network (RNN).

    Attributes:
        rnn_size (int): The size of the hidden state for each RNN cell.
        rnn_cls (nn.Module): The class for the RNN cell to be used.
        num_layers (int): The number of RNN layers.
        activation_fn (str): The activation function to use in each RNN cell (default is "tanh").
    """

    rnn_size: int
    rnn_cls: nn.Module
    num_layers: int
    activation_fn: str = "sigmoid"

    def setup(self) -> None:
        """Set up the RNN cells for the stacked RNN."""
        self.cells = [
            self.rnn_cls(
                features=self.rnn_size, activation_fn=parse_activation_fn(self.activation_fn)
            )
            for _ in range(self.num_layers)
        ]

    def __call__(
        self, all_rnn_states: List[chex.ArrayTree], x: chex.Array
    ) -> Tuple[List[chex.ArrayTree], chex.Array]:
        """
        Run the stacked RNN cells on the input.

        Args:
            all_rnn_states (List[chex.ArrayTree]): List of RNN states for each layer.
            x (chex.Array): Input to the RNN.

        Returns:
            Tuple[List[chex.ArrayTree], chex.Array]: A tuple containing the a list of 
                the RNN states of each RNN and the output of the last layer.
        """
        # Ensure all_rnn_states is a list
        if not isinstance(all_rnn_states, list):
            all_rnn_states = [all_rnn_states]
        
        assert len(all_rnn_states) == self.num_layers, (
            f"Expected {self.num_layers} RNN states, but got {len(all_rnn_states)}."
        )

        new_states = []
        for cell, rnn_state in zip(self.cells, all_rnn_states):
            new_rnn_state, x = cell(rnn_state, x)
            new_states.append(new_rnn_state)

        return new_states, x
