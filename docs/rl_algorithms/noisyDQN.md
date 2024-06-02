# Noisy DQN

Key features:

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

* The usual loss is wrapped by expectation over the noise $\epsilon$: $\bar L(\zeta)=\mathbb E[L(\theta)]$

* For a network layer $y=wx+b$, the corresponding noisy layer is defined as:

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

* As the loss of a noisy network is an expectation over the noise, the gradients are simply given by: $$\nabla \bar L(\zeta) = \nabla \mathbb E[L(\theta)] = \mathbb E[\nabla_{\mu, \Sigma}L(\mu + \Sigma\odot\epsilon)]$$
* In the case of Noisy DQN, the loss is given by: $$\bar L(\zeta) = \mathbb E
  \left[
  \mathbb E_{(s,a,r,s')\sim D}[r + \gamma\max_{b\in A}Q(s',b,\epsilon'; \zeta^-) - Q(s,a,\epsilon; \zeta)]^2
  \right]$$

* ### TODO: detail diff between factored and independent init

  * In DQN, the noise is usually generated using **Factorized Gaussian Noise**:
    * For a weight matrix $W$  of size $(m \times n),$ two vectors of Gaussian noise are sampled:
   $$\begin{align*}

\epsilon^w_{m,n} &= f(\epsilon_m)f(\epsilon_n) \\
\epsilon^b_n(n) &= f(\epsilon_n)
\end{align*}$$
$$\begin{align*}
\begin{cases}
 \text{row noise}&:\epsilon^w \sim \mathcal{N}(0, I_m)\\
 \text{column noise}&:\epsilon^b \sim \mathcal{N}(0, I_n) \\
 f(x) &= \text{sgn}(x)\sqrt{|x|}
\end{cases}
\end{align*}$$

  * This approach **reduces the number of noise samples needed**, making it **more efficient** compared to adding independent noise to each weight.

  * By incorporating noise into the network weights, Noisy DQN inherently **encourages exploration**.
  * The noise is **not static**; it is parameterized and learned during the training process. This allows the network to **adapt the amount of noise**, optimizing the **balance between exploration and exploitation dynamically**.
  * Over time, the network can **learn to ignore the noisy stream**, but will do so at **different rates in different parts of the state space**, allowing **state-conditional exploration** with a form of self-annealing
### Advantages
1. **Improved Exploration**: Traditional DQN often relies on epsilon-greedy strategies for exploration, which can be inefficient. Noisy DQN, with its inherent noise, provides a more natural and efficient way to explore the action space.

2. **Reduced Hyperparameters**: By integrating noise directly into the network, Noisy DQN reduces the need for manually tuning exploration-related hyperparameters like epsilon in epsilon-greedy strategies.

3. **Better Performance**: Noisy DQN has been shown to perform better than standard DQN in various tasks, particularly in environments where exploration is crucial for learning optimal policies.
