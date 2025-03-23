# Vanilla Policy Gradient (VPG) Algorithms

Fundamental policy-based reinforcement learning methods that directly optimize the policy using gradient ascent.

## Implementations
- `ff_reinforce.py`: Feed-forward REINFORCE for discrete action spaces
- `ff_reinforce_continuous.py`: Feed-forward REINFORCE for continuous action spaces

## Algorithm Overview
VPG methods optimize policies by following the gradient of expected return with respect to policy parameters.

### REINFORCE
A Monte Carlo policy gradient method that:
- Collects complete trajectories from current policy
- Computes full returns for each timestep
- Updates policy to increase probability of actions that led to higher returns

Key characteristics:
- On-policy learning (uses only current policy data)
- Requires complete episodes (Monte Carlo estimation)
- Basic version has no value function baseline
- High variance in gradient estimates

VPG serves as the foundation for more advanced policy gradient methods like PPO, TRPO, and A2C, which address the variance and sample efficiency limitations.

## Key Papers
- [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (Williams, 1992) - REINFORCE
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) (Sutton et al., 2000) - Theory
