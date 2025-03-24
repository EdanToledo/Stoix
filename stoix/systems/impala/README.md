# IMPALA (Importance Weighted Actor-Learner Architecture)

A highly scalable distributed actor-critic reinforcement learning algorithm.

## Implementations
- `sebulba/ff_impala.py`: Distributed implementation with V-trace correction, multiple actors, and a central learner

## Algorithm Overview
IMPALA decouples acting from learning for efficient distributed training:
- Multiple actors generate experience trajectories
- Central learner processes experiences to update policy
- V-trace correction handles the off-policy learning from lagging actor policies
- Enables efficient scaling with reduced actor-learner coupling

## Key Papers
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) (Espeholt et al., 2018)
