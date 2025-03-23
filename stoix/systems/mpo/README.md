# Maximum a Posteriori Policy Optimisation (MPO) Algorithms

Actor-critic algorithms that use constrained policy updates with Expectation-Maximization procedures.

## Implementations
- `ff_mpo.py`: Feed-forward MPO for discrete action spaces
- `ff_mpo_continuous.py`: Feed-forward MPO for continuous action spaces
- `ff_vmpo.py`: Feed-forward V-MPO (on-policy variant) for discrete action spaces
- `ff_vmpo_continuous.py`: Feed-forward V-MPO for continuous action spaces

## Algorithm Overview

### MPO
An off-policy method using a two-step EM procedure:
- E-step: Constructs non-parametric policy that maximizes expected value within trust region
- M-step: Updates parametric policy to match non-parametric policy with KL constraint
- Decouples policy improvement from parameterization for stable learning

### V-MPO
An on-policy adaptation that:
- Uses state-value function instead of action-value function
- Samples from current policy (no replay buffer)
- Selects actions based on advantages
- Maintains EM procedure with trust region constraints

## Key Papers
- [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920) (Abdolmaleki et al., 2018) - MPO
- [Relative Entropy Regularized Policy Iteration](https://arxiv.org/abs/1812.02256) (Abdolmaleki et al., 2018) - Related work
- [V-MPO: On-Policy Maximum a Posteriori Policy Optimization](https://arxiv.org/abs/1909.12238) (Song et al., 2019) - V-MPO
