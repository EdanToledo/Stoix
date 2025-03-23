# Monte Carlo Tree Search Algorithms

Tree-based planning algorithms for decision making, with implementations inspired by AlphaZero and MuZero.

## Implementations
- `ff_az.py`: Feed-forward AlphaZero-style MCTS
- `ff_mz.py`: Feed-forward MuZero-style MCTS
- `ff_sampled_az.py`: AlphaZero with sampled MCTS
- `ff_sampled_mz.py`: MuZero with sampled MCTS

## Algorithm Overview

### AlphaZero (AZ)
Combines MCTS with deep neural networks for self-play learning:
- Neural networks predict policy and value
- No human knowledge or demonstrations required
- MCTS provides planning and improved decision making

### MuZero (MZ)
Extends AlphaZero by learning a model of environment dynamics:
- Predicts policy, value, rewards, and next state representations
- Works in environments with unknown rules/dynamics
- Enables planning without prior knowledge of environment

### Sampled MCTS
Stochastic variants of AZ/MZ tree search:
- Uses sampling instead of deterministic selection
- Improves exploration in some environments
- Better suited for high branching factor problems

## Key Papers
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) (Silver et al., 2017) - AlphaZero
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) (Schrittwieser et al., 2019) - MuZero
- [Monte-Carlo Tree Search as Regularized Policy Optimization](https://arxiv.org/abs/2007.12509) (Grill et al., 2020) - Theory
