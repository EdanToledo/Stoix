import chex
import flax.linen as nn

from stoix.systems.disco_rl.disco_rl_types import AgentOutput


class DiscoAgentNetwork(nn.Module):
    """
    A network for the DiscoRL agent.

    This network has a shared torso and five separate heads, matching
    the architecture required by the DiscoUpdateRule:
    1. logits (Policy)
    2. q (Categorical Value)
    3. y (Auxiliary)
    4. z (Auxiliary)
    5. aux_pi (Auxiliary Policy)
    """

    torso: nn.Module
    logits_head: nn.Module
    q_head: nn.Module
    y_head: nn.Module
    z_head: nn.Module
    aux_pi_head: nn.Module

    def __call__(self, obs: chex.Array) -> AgentOutput:
        """Forward pass."""
        # Run the shared torso
        torso_output = self.torso(obs)

        # Run all five heads
        logits = self.logits_head(torso_output)
        q = self.q_head(torso_output)
        y = self.y_head(torso_output)
        z = self.z_head(torso_output)
        aux_pi = self.aux_pi_head(torso_output)

        return AgentOutput(logits=logits, q=q, y=y, z=z, aux_pi=aux_pi)
