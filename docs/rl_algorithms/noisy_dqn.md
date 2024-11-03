# Noisy DQN

| :material-file-document: Paper      |:material-github: code |
| ----------- | ----------- |
|*[Fortunato et al. (2017)](http://arxiv.org/abs/1706.10295)*| [`noisy_layer`](https://github.com/EdanToledo/Stoix/blob/main/stoix/networks/layers/layers.py) |

## Key features

* NoisyNets are neural networks whose weights and biases are **perturbed** by a **parametric function of the noise**.
* Consider a neural network $y = f_\theta(x)$ parameterized by the vector of noisy parameters:

$$
    \begin{align*}
    \theta &= \mu + \Sigma \odot\epsilon \\
    &\begin{cases}
    \zeta = (\mu, \Sigma): \text{a set of learnable parameters} \\
    \epsilon: \text{zero-mean noise vector with fixed statistics}
    \end{cases}
    \end{align*}
$$

* For a linear network layer $y=wx+b$, the corresponding noisy layer is defined as:

$$
    \begin{align*}
    y&=(\mu^w+\sigma^w \odot \epsilon^w)\cdot x + \mu^b + \sigma^b \odot \epsilon^b \\
    \\
    &\begin{cases}
    w &= \mu^w+\sigma^w \odot \epsilon^w \\
    b &= \mu^b + \sigma^b \odot \epsilon^b \\
    \mu^w,\mu^b, \sigma^w, \sigma^b &: \text{learnable parameters} \\
    \epsilon^w, \epsilon^b &: \text{noise random variables}
    \end{cases}
    \end{align*}
$$

* For performance, the noise is generated using **Factorized Gaussian Noise**. For a linear layer with $m$ inputs and $n$ outputs, a noise matrix of shape $(m\times n)$ is generated from two noise vectors. This methods reduces the number of required random variables from $m\times n$ to $m+n$:

$$ \begin{align*}
\epsilon^w_{m,n} &= f(\epsilon_m)f(\epsilon_n) \\
\epsilon^b_n &= f(\epsilon_n)
\end{align*}
$$

$$\begin{align*}
\begin{cases}
 \text{weight noise}&:\epsilon^w_{m,n}\\
 \text{bias noise}&:\epsilon^b_n\\
 \text{row noise}&:\epsilon^w \sim \mathcal{N}(0, I_m)\\
 \text{column noise}&:\epsilon^b \sim \mathcal{N}(0, I_n) \\
 f(x) &= \text{sgn}(x)\sqrt{|x|}
\end{cases}
\end{align*}$$

## Advantages
1. **Improved Exploration**: Traditional DQN often relies on epsilon-greedy strategies for exploration, which can be inefficient. Noisy DQN, with its inherent noise, provides a more natural and efficient way to explore the action space.

2. **Reduced Hyperparameters**: By integrating noise directly into the network, Noisy DQN reduces the need for manually tuning exploration-related hyperparameters like epsilon in epsilon-greedy strategies.

3. **Better Performance**: Noisy DQN has been shown to perform better than standard DQN in various tasks, particularly in environments where exploration is crucial for learning optimal policies.
