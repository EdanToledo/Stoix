# Soft Actor-Critic (SAC)

An off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework.

## Implementations
- `ff_sac.py`: Feed-forward implementation of SAC

## Algorithm Overview
SAC combines off-policy learning with entropy maximization, encouraging both reward maximization and exploration. The algorithm:
- Uses a replay buffer for sample efficiency
- Incorporates entropy terms into the Q-function
- Can automatically tune the temperature parameter
- Provides stable learning and excellent performance in continuous action spaces

## Key Papers
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) (Haarnoja et al., 2018) - With automatic temperature tuning
