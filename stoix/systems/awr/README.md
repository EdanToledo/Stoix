# Advantage-Weighted Regression (AWR)

A simple off-policy algorithm that optimizes policies via supervised learning with advantage weighting.

## Implementations
- `ff_awr.py`: Feed-forward implementation for discrete action spaces
- `ff_awr_continuous.py`: Feed-forward implementation for continuous action spaces

## Algorithm Overview
AWR optimizes policies through weighted regression using advantages:
- Collects experience with current policy
- Updates policy via supervised learning by weighting state-action pairs by their exponential advantages
- Designed for stability and simplicity while maintaining effectiveness

## Key Papers
- [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://arxiv.org/abs/1910.00177) (Peng et al., 2019)
