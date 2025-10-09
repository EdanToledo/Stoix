# Munchausen DQN (M-DQN)

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Vieillard et al. (2020)](https://arxiv.org/abs/2007.14430)*| [`munchausen_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/munchausen_dqn.py) |

## Key features

* **Munchausen Regularization**: M-DQN introduces a Munchausen term to the Bellman update equation, adding a scaled log-policy term to the immediate reward. This stabilizes learning, promotes exploration, and increases robustness to reward scaling.

* **Modified Bellman Update**: The Q-value update in M-DQN is modified by incorporating the Munchausen term. The target value is computed as follows:

$$
    y = r + \alpha \tau \log \pi(a|s) + \gamma \sum_{a'} \pi(a'|s') [Q_{\theta^-}(s', a') - \tau \log \pi(a'|s')]
$$

where \( r \) is the reward, \( \alpha \) is the Munchausen coefficient, \( \tau \) is the temperature parameter, \( \log \pi(a|s) \) is the log-policy term, \( \gamma \) is the discount factor, and \( Q_{\theta^-}(s', a') \) is the Q-value of the next state-action pair from the target network.

* **Softmax Policy**: The policy \( \pi(a|s) \) is derived from the Q-values using a softmax function:

$$
    \pi(a|s) = \frac{\exp(Q(s, a)/\tau)}{\sum_{a'} \exp(Q(s, a')/\tau)}
$$

where \( \tau \) is a temperature parameter that controls the exploration-exploitation trade-off.

* **Experience Replay**: M-DQN uses experience replay to store experiences $(s, a, r, s')$ in a replay buffer and samples mini-batches of experiences to break the correlation between consecutive updates and stabilize training.

* **Target Network**: M-DQN employs a target network to provide stable target values. The target network is updated periodically or using Polyak averaging.

## Algorithm

1. **Initialize**:
    - Initialize the Q-network with random weights \( \theta \).
    - Initialize the target Q-network with weights \( \theta^- = \theta \).
    - Initialize the replay buffer \( D \).

2. **Interaction with Environment**:
    - For each episode:
        - For each time step:
            - With probability \( \epsilon \), select a random action \( a \).
            - Otherwise, select the action \( a = \arg\max Q(s, a; \theta) \).
            - Execute action \( a \) and observe reward \( r \) and next state \( s' \).
            - Store transition \( (s, a, r, s') \) in the replay buffer \( D \).

3. **Training**:
    - Sample a mini-batch of transitions \( (s, a, r, s') \) from the replay buffer \( D \).

- Compute the target value incorporating the Munchausen term:

$$
    y = r + \alpha \tau \log \pi(a|s) + \gamma \sum_{a'} \pi(a'|s') [Q_{\theta^-}(s', a') - \tau \log \pi(a'|s')]
$$

- Perform a gradient descent step on the loss function:

$$
    L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]
$$

4. **Update Target Network**:
    - Every \( C \) steps, update the target network: \( \theta^- = \theta \).
    - Alternatively, use Polyak averaging to update the target network incrementally: \( \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \), where \( \tau \in [0, 1] \) is the Polyak averaging factor.

## Advantages
1. **Better Performance**: M-DQN has shown to outperform traditional DQN and other advanced algorithms in various environments, especially those with sparse or noisy rewards.
