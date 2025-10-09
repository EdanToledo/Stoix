# Quantile Regression DQN (QR-DQN)

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Dabney et al. (2018)](https://arxiv.org/abs/1710.10044)*| [`qr_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_qr_dqn.py) |

## Key features

* **Distributional Reinforcement Learning**: QR-DQN models the distribution of the return \( Z(s, a) \) by learning the quantiles of the return distribution, rather than just the expected value. This allows the agent to capture the variability and uncertainty in the environment more effectively.

* **Quantile Regression**: QR-DQN uses quantile regression to estimate the distribution of returns. The quantiles are represented by a set of values \( \theta_i \) for each quantile \( i \), and the loss function minimizes the difference between the predicted and target quantiles:

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \rho_{\hat{\tau}_i} \left( r + \gamma \theta_j(s', a') - \theta_i(s, a) \right)
$$

where \( \rho_{\hat{\tau}_i}(u) = u(\hat{\tau}_i - \mathbb{I}(u < 0)) \) is the quantile Huber loss, \( r \) is the reward, \( \gamma \) is the discount factor, and \( \theta_j(s', a') \) are the target quantiles.

* **Target Distribution**: The target distribution is computed using the Bellman update, but instead of taking the maximum Q-value, QR-DQN uses the quantile values to estimate the distribution of the returns. The target quantiles are given by:

$$
\theta_j' = r + \gamma \theta_j(s', a')
$$

* **Efficient Learning**: By estimating multiple quantiles, QR-DQN can provide a richer representation of the return distribution, improving the agent's learning efficiency and robustness.

## Algorithm

1. **Initialize**:
    - Initialize the Q-network with random weights \( \theta \).
    - Initialize the target Q-network with weights \( \theta^- = \theta \).
    - Initialize the replay buffer \( D \).
    - Define the quantile fractions \( \tau_i = \frac{2i - 1}{2N} \) for \( i = 1, 2, \ldots, N \).

2. **Interaction with Environment**:
    - For each episode:
        - For each time step:
            - With probability \( \epsilon \), select a random action \( a \).
            - Otherwise, select the action \( a = \arg\max \frac{1}{N} \sum_{i=1}^N \theta_i(s, a) \).
            - Execute action \( a \) and observe reward \( r \) and next state \( s' \).
            - Store transition \( (s, a, r, s') \) in the replay buffer \( D \).

3. **Training**:
    - Sample a mini-batch of transitions \( (s, a, r, s') \) from the replay buffer \( D \).
    - Compute the target quantiles:

    $$
    \theta_j' = r + \gamma \theta_j(s', a')
    $$

    - Compute the quantile Huber loss:

    $$
    L(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \rho_{\hat{\tau}_i} \left( r + \gamma \theta_j(s', a') - \theta_i(s, a) \right)
    $$

    - Perform a gradient descent step on the loss function \( L(\theta) \).

4. **Update Target Network**:
    - Every \( C \) steps, update the target network: \( \theta^- = \theta \).
    - Alternatively, use Polyak averaging to update the target network incrementally: \( \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \), where \( \tau \in [0, 1] \) is the Polyak averaging factor.

## Advantages
1. **Better Uncertainty Modeling**: By modeling the distribution of returns through quantiles, QR-DQN provides a more detailed understanding of uncertainty and variability in the environment.

2. **Stability and Robustness**: The quantile-based approach in QR-DQN improves stability and robustness, especially in environments with high variability in rewards.

3. **Improved Performance**: QR-DQN has demonstrated superior performance in various benchmarks, often outperforming traditional DQN and other advanced algorithms.
