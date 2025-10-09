# Dueling DQN

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Wang et al. (2016)](https://arxiv.org/abs/1511.06581)*| [`dueling_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dueling_dqn.py) |

## Key features

* **Separate Estimations**: Dueling DQN architecture separates the estimation of the state value function \( V(s) \) and the advantage function \( A(s, a) \). This helps the agent to learn which states are valuable without requiring precise action values.

* **Network Architecture**: The network architecture for Dueling DQN consists of two streams: one for the state value function \( V(s) \) and another for the advantage function \( A(s, a) \). These streams are then combined to produce the Q-values. The combined Q-value function is given by:

$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
$$

where \( \theta \) are the shared parameters, \( \alpha \) and \( \beta \) are the parameters for the advantage and value streams, respectively, and \( |\mathcal{A}| \) is the number of possible actions.

* **Experience Replay**: Dueling DQN uses experience replay to store experiences $(s, a, r, s')$ in a replay buffer. It samples mini-batches from this buffer to break the correlation between consecutive experiences, thereby stabilizing training.

* **Target Network**: Similar to DQN, Dueling DQN employs a target network to provide stable target values. The target network is updated periodically or using Polyak averaging.

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
    - Compute the target value using the target network:

$$
    y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
$$

- Perform a gradient descent step on the loss function:

$$
    L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]
$$

4. **Update Target Network**:
    - Every \( C \) steps, update the target network: \( \theta^- = \theta \).
    - Alternatively, use Polyak averaging to update the target network incrementally: \( \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \), where \( \tau \in [0, 1] \) is the Polyak averaging factor.

## Advantages
1. **Improved Value Estimation**: By separately estimating the state value function and the advantage function, Dueling DQN provides more robust estimates of state values, leading to better policy decisions.

2. **Stability and Reliability**: The architecture helps to stabilize the learning process by reducing the noise in value estimates for less important actions.

3. **Enhanced Performance**: Dueling DQN has been shown to outperform standard DQN in various benchmarks, particularly in scenarios where it is more important to differentiate between good and bad states rather than actions.
