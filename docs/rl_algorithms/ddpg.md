# DDPG (Deep Deterministic Policy Gradient)

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Lillicrap et al. (2016)](https://arxiv.org/abs/1509.02971)*| [`ddpg.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/ff_ddpg.py) |

## Key features

* **Actor-Critic Architecture**: DDPG uses an actor-critic architecture where the actor learns the policy and the critic learns the Q-value function. The actor network \( \mu(s|\theta^\mu) \) outputs the action directly, while the critic network \( Q(s, a|\theta^Q) \) evaluates the action.

* **Deterministic Policy**: Unlike stochastic policy gradient methods, DDPG uses a deterministic policy, making it suitable for continuous action spaces.

* **Experience Replay**: DDPG employs experience replay to store transitions \((s, a, r, s')\) in a replay buffer. This helps in breaking the correlation between consecutive experiences and stabilizes the training process.

* **Target Networks**: DDPG uses target networks for both the actor and the critic to provide stable target values. The target networks \( \mu' \) and \( Q' \) are soft updates of the original networks:

$$
    \theta^{\mu'} \leftarrow \tau \theta^\mu + (1 - \tau) \theta^{\mu'}
$$
$$
    \theta^{Q'} \leftarrow \tau \theta^Q + (1 - \tau) \theta^{Q'}
$$

where \( \tau \) is a parameter close to 0, ensuring slow updates.

* **Policy Update**: The actor policy is updated using the sampled policy gradient:

$$
    \nabla_{\theta^\mu} J \approx \mathbb{E}_{s \sim \mathcal{D}} \left[ \nabla_a Q(s, a|\theta^Q) |_{a = \mu(s|\theta^\mu)} \nabla_{\theta^\mu} \mu(s|\theta^\mu) \right]
$$

* **Critic Update**: The critic is updated by minimizing the mean squared error loss between the target Q-value and the predicted Q-value:

$$
    L(\theta^Q) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ (r + \gamma Q'(s', \mu'(s'|\theta^{\mu'})|\theta^{Q'}) - Q(s, a|\theta^Q))^2 \right]
$$

## Algorithm

1. **Initialize**:
    - Initialize the actor network \( \mu(s|\theta^\mu) \) with random weights \( \theta^\mu \).
    - Initialize the critic network \( Q(s, a|\theta^Q) \) with random weights \( \theta^Q \).
    - Initialize the target networks \( \mu' \) and \( Q' \) with weights \( \theta^{\mu'} \leftarrow \theta^\mu \) and \( \theta^{Q'} \leftarrow \theta^Q \).
    - Initialize the replay buffer \( \mathcal{D} \).

2. **Interaction with Environment**:
    - For each episode:
        - For each time step:
            - Select action \( a_t = \mu(s_t|\theta^\mu) + \mathcal{N}_t \), where \( \mathcal{N}_t \) is noise for exploration.
            - Execute action \( a_t \) and observe reward \( r_t \) and next state \( s

_{t+1} \).
            - Store transition \( (s_t, a_t, r_t, s_{t+1}) \) in the replay buffer \( \mathcal{D} \).

3. **Training**:
    - Sample a mini-batch of \( N \) transitions \( (s_i, a_i, r_i, s_{i+1}) \) from the replay buffer \( \mathcal{D} \).
    - Compute the target Q-value for each transition:

$$
    y_i = r_i + \gamma Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})
$$

- Update the critic network by minimizing the loss:

$$
    L(\theta^Q) = \frac{1}{N} \sum_i \left( y_i - Q(s_i, a_i|\theta^Q) \right)^2
$$

- Update the actor network using the policy gradient:

$$
    \nabla_{\theta^\mu} J \approx \frac{1}{N} \sum_i \nabla_a Q(s_i, a|\theta^Q) |_{a = \mu(s_i|\theta^\mu)} \nabla_{\theta^\mu} \mu(s_i|\theta^\mu)
$$

4. **Update Target Networks**:
    - Update the target networks using soft updates:

$$
    \theta^{\mu'} \leftarrow \tau \theta^\mu + (1 - \tau) \theta^{\mu'}
$$
$$
    \theta^{Q'} \leftarrow \tau \theta^Q + (1 - \tau) \theta^{Q'}
$$
