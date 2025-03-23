# Proximal Policy Optimization (PPO)

A popular on-policy actor-critic algorithm known for simplicity, effectiveness, and robust performance.

## Implementations
- `anakin/`:
  - `ff_ppo.py`: Standard PPO for discrete actions
  - `ff_ppo_continuous.py`: PPO for continuous actions
  - `ff_ppo_penalty.py`: PPO with KL penalty (discrete)
  - `ff_ppo_penalty_continuous.py`: PPO with KL penalty (continuous)
  - `ff_dpo_continuous.py`: Discovered Policy Optimization (continuous)
  - `rec_ppo.py`: Recurrent PPO for partially observable environments
- `sebulba/`:
  - `ff_ppo.py`: Distributed PPO with parallel actors and central learner

## Algorithm Overview
PPO approximates trust region policy optimization with a simpler, more efficient approach:
- Clipped surrogate objective limits policy update size
- Generalized Advantage Estimation (GAE) balances bias/variance
- Multiple optimization epochs per data batch
- Works well in both discrete and continuous domains
- More stable than standard policy gradient methods while being simpler than TRPO

## Key Papers
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (Schulman et al., 2016) - GAE
