# Rainbow DQN

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Hessel et al. (2017)](https://arxiv.org/abs/1710.02298)*| [`ff_rainbow.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_rainbow.py) |

## Key features

Rainbow is a combination of 6 variants of DQN:

1. [**Double DQN**](ddqn.md): Double Q-learning is used to fix the overstimation bias in Q-learning.
2. [**Prioritized Experience Replay**](https://arxiv.org/abs/1511.05952): Experiences are sampled with a probability proportional to the absolute TD error, improving sample efficiency.
3. [**Dueling Networks**](dueling_dqn.md): Dueling DQN separates the estimation of the state value function and the advantage function. This helps the agent to learn which states are valuable without requiring precise action values.
4. [**Multi-step learning** (as in A3C)](https://arxiv.org/abs/1602.01783): Rainbow interacts with multiple environments in parallel, speeding up data collection and reducing sample correlation.
5. [**Categorical DQN** (C51)](c51.md): Instead of estimating the expected return using the Bellman equation, C51 models the distribution of returns. This allows the agent to capture the inherent uncertainty and variability in the rewards.
6. [**Noisy DQN**](noisy_dqn.md): Noisy DQN uses dense layers where weights and biases are perturbed by a parametric function of Gaussian noise. By incorporating noise directly into the weights, the resulting exploration can be more state-dependent, leading to more targeted and efficient exploration strategies.

## Algorithm

1. **Initialize**:
    - Initialize the Q-network with random weights \( \theta \).
    - Initialize the target Q-network with weights \( \theta^- = \theta \).
    - Initialize the replay buffer \( D \).
    - Define the atom support \( z_i \) within the range \([v_{\text{min}}, v_{\text{max}}]\).
    - Initialize parameters for multi-step learning \( n \).
    - Initialize priority exponent \( \alpha \) and importance sampling exponent \( \beta \) for Prioritized Experience Replay (PER).
    - Initialize noise parameter's variance $\sigma_0$ for Noisy Networks.

2. **Interaction with Environment**:
    - For each episode:
        - For each time step:
            - Select action \( a \) using noisy weights: \( a = \arg\max \mathbb{E}[Z(s, a; \theta)] \).
            - Execute action \( a \) and observe reward \( r \) and next state \( s' \).
            - Store transition \( (s, a, r, s') \) in the replay buffer \( D \) with priority \( p \) based on TD-error.

3. **Training**:
    - Sample a mini-batch of transitions \( (s, a, r, s') \) from the replay buffer \( D \) using prioritized sampling.
    - Compute the multi-step return for each transition.
    - Compute the target distribution by projecting the Bellman update onto the fixed support $z$:

$$
\begin{align*}
\hat{T}Z(s, a) &= \Phi_z(R^{(n)}_t + \gamma_t^{(n)}z, p_{\bar \theta}(S_{t+n}, a^\star_{t+n}))
\\ \\
&\begin{cases}
\Phi_z &: L^2\text{-Projection on } z\\
R^{(n)}_{t} &= \sum_{k=0}^{n-1} \gamma^{k}R_{t+k+1}  \\
\gamma_t^{(n)} &= \gamma^{n}
\end{cases}
\end{align*}
$$

- Minimize the KL divergence between the projected target distribution and the estimated distribution using gradient descent:

$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \text{KL}(\hat{T}Z(s, a) \parallel Z(s, a; \theta)) \right]
$$

- Update priorities in the replay buffer based on the new TD-errors.

1. **Update Target Network**:
    - Every \( C \) steps, update the target network: \( \theta^- = \theta \).
    - Alternatively, use Polyak averaging to update the target network incrementally: \( \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \), where \( \tau \in [0, 1] \) is the Polyak averaging factor.
