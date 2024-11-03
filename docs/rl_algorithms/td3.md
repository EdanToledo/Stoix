# TD3 (Twin Delayed Deep Deterministic Policy Gradient)

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Fujimoto et al. (2018)](https://arxiv.org/abs/1802.09477)*| [`td3.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/td3.py) |

## Key features

* **Addressing Overestimation Bias**: TD3 mitigates the overestimation bias present in standard actor-critic methods by using two critic networks. The minimum value of the two critics is used for the target, reducing overestimation and leading to more stable learning.

* **Clipped Double Q-learning**: TD3 employs clipped double Q-learning by maintaining two Q-networks \( Q_{\theta_1} \) and \( Q_{\theta_2} \). The target value is computed using the minimum of the two Q-values:

$$
    y = r + \gamma \min_{i=1,2} Q_{\theta_i^-}(s', \mu'(s'|\theta^{\mu'}) + \epsilon)
$$

where \( \epsilon \) is noise added to the target policy for smoothing, and \( \gamma \) is the discount factor.

* **Delayed Policy Updates**: The actor (policy) network is updated less frequently than the critic networks. This helps in reducing the variance of the policy updates and stabilizes training.

* **Target Policy Smoothing**: TD3 adds noise to the target action to regularize the Q-value estimates, preventing the policy from exploiting sharp peaks in the Q-value function.

## Algorithm

1. **Initialize**:
    - Initialize the actor network \( \mu(s|\theta^\mu) \) with random weights \( \theta^\mu \).
    - Initialize the critic networks \( Q_{\theta_1}(s, a) \) and \( Q_{\theta_2}(s, a) \) with random weights \( \theta_1 \) and \( \theta_2 \).
    - Initialize the target networks \( \mu' \), \( Q_{\theta_1'} \), and \( Q_{\theta_2'} \) with weights \( \theta^{\mu'} \leftarrow \theta^\mu \), \( \theta_1' \leftarrow \theta_1 \), and \( \theta_2' \leftarrow \theta_2 \).
    - Initialize the replay buffer \( \mathcal{D} \).

2. **Interaction with Environment**:
    - For each episode:
        - For each time step:
            - Select action \( a_t = \mu(s_t|\theta^\mu) + \mathcal{N}_t \), where \( \mathcal{N}_t \) is noise for exploration.
            - Execute action \( a_t \) and observe reward \( r_t \) and next state \( s_{t+1} \).
            - Store transition \( (s_t, a_t, r_t, s_{t+1}) \) in the replay buffer \( \mathcal{D} \).

3. **Training**:
    - Sample a mini-batch of \( N \) transitions \( (s_i, a_i, r_i, s_{i+1}) \) from the replay buffer \( \mathcal{D} \).
    - Add noise \( \epsilon \) to the target action for smoothing:

$$
    a' = \mu'(s_{i+1}|\theta^{\mu'}) + \epsilon
$$

- Compute the target Q-value using the minimum of the two target critics:

$$
    y_i = r_i + \gamma \min_{j=1,2} Q_{\theta_j'}(s_{i+1}, a')
$$

- Update the critic networks by minimizing the loss:

$$
    L(\theta_j) = \frac{1}{N} \sum_i \left( y_i - Q_{\theta_j}(s_i, a_i) \right)^2 \quad \text{for } j = 1, 2
$$

- Delayed Policy Update: Update the actor network and the target networks every \( d \) steps:
        - Update the actor network using the policy gradient:

$$
    \nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q_{\theta_1}(s_i, a|\theta_1) |_{a = \mu(s_i|\theta^\mu)} \nabla_{\theta^\mu} \mu(s_i|\theta^\mu)
$$

- Update the target networks using soft updates:

$$
    \theta^{\mu'} \leftarrow \tau \theta^\mu + (1 - \tau) \theta^{\mu'}
$$
$$
    \theta_1' \leftarrow \tau \theta_1 + (1 - \tau) \theta_1'
$$
$$
    \theta_2' \leftarrow \tau \theta_2 + (1 - \tau) \theta_2'
$$

## Advantages
1. **Reduced Overestimation Bias**: By using the minimum of two Q-networks, TD3 reduces overestimation bias, leading to more accurate value estimates and stable learning.

2. **Stability and Reliability**: Delayed policy updates and target policy smoothing contribute to the stability and reliability of the learning process.

3. **Improved Performance**: TD3 has demonstrated superior performance on various continuous control benchmarks, often outperforming previous algorithms like DDPG.
