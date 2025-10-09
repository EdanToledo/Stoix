# DQN

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Mnih et al. (2015)](https://www.nature.com/articles/nature14236)*| [`ff_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dqn.py) |

## Key features

* **Q-Learning with Deep Neural Networks**: Deep Q-Networks (DQN) utilize deep neural networks to approximate the Q-value function, $Q(s, a)$, which represents the expected future rewards of taking action $a$ in state $s$.

* **Experience Replay**: DQN uses a replay buffer to store experiences $(s, a, r, s')$. Instead of learning directly from consecutive samples, it randomly samples a mini-batch from the buffer to break the correlation between consecutive experiences and stabilize training.

* **Target Network**: To further stabilize training, DQN uses a target network, $\hat{Q}$, which is a delayed copy of the Q-network. The target network provides the target values for the Q-learning update:

$$
    y = r + \gamma \max_{a'} \hat{Q}(s', a')
$$

where $\gamma$ is the discount factor, and $s'$ is the next state.

* **Q-Learning Update**: The Q-network parameters $\theta$ are updated by minimizing the mean squared error (or huber) loss between the predicted Q-values and the target values:

$$
    L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]
$$

where $D$ is the replay buffer, and $Q(s, a; \theta)$ is the Q-value predicted by the Q-network.

## Algorithm

1. **Initialize**:
    - Initialize the Q-network with random weights $\theta$.
    - Initialize the target network with weights $\theta^- = \theta$.
    - Initialize the replay buffer $D$.

2. **Interaction with Environment**:
    - For each episode:
        - For each time step:
            - With probability $\epsilon$, select a random action $a$.
            - Otherwise, select the action $a = \arg\max Q(s, a; \theta)$.
            - Execute action $a$ and observe reward $r$ and next state $s'$.
            - Store transition $(s, a, r, s')$ in the replay buffer $D$.

3. **Training**:
    - Sample a mini-batch of transitions $(s, a, r, s')$ from the replay buffer $D$.
    - Compute the target value $y = r + \gamma \max_{a'} \hat{Q}(s', a'; \theta^-)$.
    - Perform a gradient descent step on the loss function $L(\theta)$.

4. **Update Target Network**:
    - Every $C$ steps, update the target network: $\theta^- = \theta$.
    - Alternatively, use Polyak averaging to update the target network incrementally: $\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$, where $\tau \in [0, 1]$ is the Polyak averaging factor.
