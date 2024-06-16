# Double DQN

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[van Hasselt et al. (2016)](https://arxiv.org/abs/1509.06461)*| [`ddqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_ddqn.py) |

## Key features

* **Addressing Overestimation Bias**: Double DQN is designed to mitigate the overestimation bias observed in standard DQN by decoupling the action selection and evaluation steps. This leads to more accurate Q-value estimates and improved performance.

* **Double Q-Learning**: Instead of using the maximum Q-value for the next state directly, Double DQN uses the current Q-network to select the action and the target Q-network to evaluate the action. The target value for Double DQN is computed as follows:

$$
    y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta}(s', a'))
$$

where \( \theta \) are the parameters of the current Q-network, \( \theta^- \) are the parameters of the target Q-network, \( r \) is the reward, \( s' \) is the next state, and \( \gamma \) is the discount factor.

* **Experience Replay**: Like DQN, Double DQN utilizes experience replay to store experiences $(s, a, r, s')$ in a replay buffer and samples mini-batches of experiences to break the correlation between consecutive updates and stabilize training.

* **Target Network**: Double DQN employs a target network, which is a delayed copy of the Q-network. This target network is updated periodically or using Polyak averaging to provide stable target values for training.

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
    - Compute the target value using Double Q-learning:

$$
    y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta}(s', a'))
$$

- Perform a gradient descent step on the loss function:

$$
    L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]
$$

4. **Update Target Network**:
    - Every \( C \) steps, update the target network: \( \theta^- = \theta \).
    - Alternatively, use Polyak averaging to update the target network incrementally: \( \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \), where \( \tau \in [0, 1] \) is the Polyak averaging factor.

## Advantages
1. **Reduced Overestimation Bias**: By decoupling action selection and evaluation, Double DQN reduces the overestimation bias inherent in standard DQN, leading to more accurate Q-value estimates.

2. **Better Performance**: Double DQN has been shown to achieve better performance than standard DQN, particularly in environments where overestimation bias can lead to suboptimal policies.
