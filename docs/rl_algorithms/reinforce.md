# REINFORCE with Baseline

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Williams (1992)](https://link.springer.com/article/10.1007/BF00992696)*| [`reinforce_with_baseline.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/vpg/ff_reinforce.py) |

## Key features

* **Policy Gradient Method**: REINFORCE with Baseline is a policy gradient algorithm that optimizes the policy directly by maximizing the expected return. It extends the REINFORCE algorithm by using a baseline to reduce the variance of the gradient estimates.

* **Baseline Function**: The baseline function \( b(s) \) is a state-value function \( V(s) \) that estimates the expected return from a given state \( s \). By subtracting the baseline from the return, the variance of the gradient estimates is reduced without introducing bias.

* **Policy Update**: The policy parameters \( \theta \) are updated using the gradient of the expected return. The gradient estimate is given by:

$$
    \nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \left( G_t - V(s_t) \right) \right]
$$

where \( G_t \) is the return (total discounted reward) from time step \( t \), and \( V(s_t) \) is the baseline value for state \( s_t \).

* **Advantage Function**: The term \( G_t - V(s_t) \) is called the advantage \( A_t \), which measures how much better the taken action \( a_t \) is compared to the expected action.

* **Baseline Estimation**: The baseline can be estimated using a separate neural network trained to minimize the mean squared error between the predicted value and the actual returns:

$$
    L(\phi) = \mathbb{E}_\pi \left[ \left( G_t - V_\phi(s_t) \right)^2 \right]
$$

where \( \phi \) are the parameters of the baseline network.

## Algorithm

1. **Initialize**:
    - Initialize the policy network with random weights \( \theta \).
    - Initialize the baseline network with random weights \( \phi \).

2. **Interaction with Environment**:
    - For each episode:
        - For each time step \( t \):
            - Select an action \( a_t \) according to the policy \( \pi_\theta(a_t|s_t) \).
            - Execute action \( a_t \) and observe reward \( r_t \) and next state \( s_{t+1} \).
            - Store transition \( (s_t, a_t, r_t, s_{t+1}) \).

3. **Compute Returns**:
    - For each time step \( t \) in the episode:
        - Compute the return \( G_t = \sum_{k=t}^T \gamma^{k-t} r_k \), where \( \gamma \) is the discount factor and \( r_k \) is the reward at time step \( k \).

4. **Update Baseline**:
    - For each time step \( t \) in the episode:
        - Update the baseline network \( \phi \) by minimizing the mean squared error loss:

$$
    L(\phi) = \mathbb{E}_\pi \left[ \left( G_t - V_\phi(s_t) \right)^2 \right]
$$

5. **Policy Update**:
    - For each time step \( t \) in the episode:
        - Compute the advantage \( A_t = G_t - V_\phi(s_t) \).
        - Update the policy network \( \theta \) by performing a gradient ascent step on the policy gradient:

$$
    \nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A_t \right]
$$

## Advantages
1. **Reduced Variance**: By using a baseline, the variance of the gradient estimates is significantly reduced, leading to more stable and efficient learning.

2. **Improved Performance**: REINFORCE with Baseline often performs better than the standard REINFORCE algorithm, especially in environments with high variability in returns.

3. **Simple Implementation**: The algorithm is straightforward to implement and can be easily combined with other techniques such as function approximation and neural networks.
