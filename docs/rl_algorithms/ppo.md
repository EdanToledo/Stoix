# PPO

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Schulman et al. (2017)](https://arxiv.org/abs/1707.06347)*| [`ppo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ppo/ff_ppo.py) |

## Key features

* **Policy Gradient Method**: PPO (Proximal Policy Optimization) is a policy gradient method that optimizes the policy directly by maximizing a surrogate objective function.

* **Clipped Objective**: PPO uses a clipped objective function to ensure that the new policy does not deviate too much from the old policy, thus maintaining stability and reliability during training. The clipped objective function is given by:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \) is the probability ratio, \( \hat{A}_t \) is the estimated advantage at time step \( t \), and \( \epsilon \) is a hyperparameter that controls the clipping range.

* **Multiple Epochs**: PPO performs multiple epochs of optimization on the same batch of data, improving sample efficiency. This allows the policy to be updated several times using the same data.

* **Value Function and Policy Updates**: PPO optimizes both the policy and the value function simultaneously. The value function \( V(s) \) is used to estimate the advantage \( \hat{A}_t \), which guides the policy updates. The loss function for the value function is typically the mean squared error between the predicted and actual returns:

$$
L^{VF}(\theta) = \mathbb{E}_t \left[ (V_\theta(s_t) - R_t)^2 \right]
$$

where \( R_t \) is the discounted return.

* **Entropy Bonus**: To encourage exploration, PPO adds an entropy bonus to the objective function, which promotes exploration by discouraging premature convergence to suboptimal deterministic policies. The entropy bonus is given by:

$$
L^{ENT}(\theta) = \mathbb{E}_t \left[ \beta \mathcal{H}[\pi_\theta(\cdot|s_t)] \right]
$$

where \( \beta \) is a coefficient controlling the strength of the entropy bonus, and \( \mathcal{H} \) is the entropy of the policy.

* **Generalized Advantage Estimation (GAE)**: PPO employs GAE to compute the advantage function, which provides a bias-variance trade-off and helps improve the stability and efficiency of the learning process. The advantage estimation is given by:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where \( \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \) is the temporal-difference error, and \( \lambda \) is a hyperparameter that controls the bias-variance trade-off.

## Algorithm

1. **Initialize**:
    - Initialize the policy network with random weights \( \theta \).
    - Initialize the value function network with random weights.

2. **Collect Data**:
    - Interact with the environment to collect trajectories of states, actions, and rewards.
    - Compute the returns \( R_t \) and advantages \( \hat{A}_t \) using GAE.

3. **Optimize Policy**:
    - For each iteration:
        - Sample a mini-batch of trajectories.
        - Compute the probability ratios \( r_t(\theta) \).
        - Compute the clipped objective \( L^{CLIP}(\theta) \) and perform a gradient ascent step.
        - Optimize the value function using \( L^{VF}(\theta) \).
        - Add the entropy bonus \( L^{ENT}(\theta) \) to the objective function to encourage exploration.

4. **Update Policy**:
    - After several epochs of optimization, update the old policy parameters to the new policy parameters: \( \theta_{old} \leftarrow \theta \).
