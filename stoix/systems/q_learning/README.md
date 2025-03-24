# Q-Learning Algorithms

Value-based deep reinforcement learning algorithms built on Q-learning.

## Implementations
- `ff_dqn.py`: Feed-forward Deep Q-Network (DQN)
- `ff_ddqn.py`: Feed-forward Double DQN
- `ff_dqn_reg.py`: Feed-forward DQN with regularization
- `ff_mdqn.py`: Feed-forward Munchausen DQN
- `ff_c51.py`: Feed-forward Categorical DQN (C51)
- `ff_qr_dqn.py`: Feed-forward Quantile Regression DQN
- `ff_rainbow.py`: Feed-forward Rainbow DQN
- `rec_r2d2.py`: Recurrent R2D2 (Recurrent Replay Distributed DQN)

## Algorithm Overview

### DQN
Combines Q-learning with neural networks:
- Experience replay breaks sample correlations
- Target network stabilizes training

### Double DQN
Addresses overestimation bias by decoupling action selection from evaluation.

### M-DQN
Munchausen DQN adds a KL regularization term to the standard DQN loss:
- Encourages policy consistency through "self-distillation"
- Adds a scaled log-policy term to the reward function

### C51
Models full return distribution using discrete fixed atoms instead of expected value.

### QR-DQN
Estimates return distribution quantiles directly through quantile regression.

### Rainbow
Combines multiple DQN improvements:
- Double Q-learning, prioritized replay, dueling architecture
- Multi-step learning, distributional RL, noisy networks

### R2D2
Extends DQN to partially observable environments with:
- Recurrent networks for temporal dependencies
- Distributed training and prioritized sequence replay

## Key Papers
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013) - DQN
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015) - Double DQN
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) (Bellemare et al., 2017) - C51
- [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) (Dabney et al., 2017) - QR-DQN
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) (Hessel et al., 2018) - Rainbow
- [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX) (Kapturowski et al., 2019) - R2D2
- [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430) (Vieillard et al., 2020) - Munchausen DQN
