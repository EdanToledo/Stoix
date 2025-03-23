# Deep Deterministic Policy Gradient (DDPG) Algorithms

Off-policy actor-critic algorithms designed primarily for continuous action spaces.

## Implementations
- `ff_ddpg.py`: Feed-forward implementation of DDPG
- `ff_td3.py`: Feed-forward implementation of Twin Delayed DDPG (TD3)
- `ff_d4pg.py`: Feed-forward implementation of Distributed Distributional DDPG (D4PG)

## Algorithm Overview

### DDPG
Combines deterministic policy gradients with DQN techniques:
- Off-policy learning with replay buffer
- Target networks for stability
- Deterministic policy gradient updates

### TD3
Addresses function approximation errors in DDPG through:
- Clipped Double Q-learning (two Q-networks)
- Delayed policy updates
- Target policy smoothing (action noise)

### D4PG
Extends DDPG with distributional RL:
- Models distribution of returns instead of expectation
- Prioritized experience replay
- N-step returns

## Key Papers
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) (Lillicrap et al., 2015) - DDPG
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (Fujimoto et al., 2018) - TD3
- [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617) (Barth-Maron et al., 2018) - D4PG
