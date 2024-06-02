# Stoix - overview

<div align="center"><img src="images/stoix.png" width="30%"> </div>

<div align="center">
<a href="https://www.python.org/doc/versions/">
      <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python Versions">
</a>
<a  href="https://github.com/instadeepai/Mava/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License" />
</a>
<a  href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style" />
</a>
<a  href="http://mypy-lang.org/">
    <img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="MyPy" />
</a>
<a href="https://zenodo.org/doi/10.5281/zenodo.10916257"><img src="https://zenodo.org/badge/758685996.svg" alt="DOI"></a>
</div>

<h2 align="center">
    <p>Distributed Single-Agent Reinforcement Learning End-to-End in JAX</p>
</h2>

<div align="center">

</div>

## Welcome to Stoix! 🏛️

Stoix provides simplified code for quickly iterating on ideas in single-agent reinforcement learning with useful implementations of popular single-agent RL algorithms in JAX allowing for easy parallelisation across devices with JAX's `pmap`. All implementations are fully compilable with JAX's `jit` thus making training and environment execution very fast. However, this requires environments written in JAX. Algorithms and their default hyperparameters have not been hyper-optimised for any specific environment and are useful as a starting point for research and/or for initial baselines.

To join us in these efforts, please feel free to reach out, raise issues or read our [contribution guidelines](#contributing-) (or just star 🌟 to stay up to date with the latest developments)!

Stoix is fully in JAX with substantial speed improvement compared to other popular libraries. We currently provide native support for the [Jumanji][jumanji] environment API and wrappers for popular JAX-based RL environments.

## Code Philosophy 🧘

The current code in Stoix was initially **largely** taken and subsequently adapted from [Mava](mava). As Mava develops, Stoix will hopefully adopt their optimisations that are relevant for single-agent RL. Like Mava, Stoix is not designed to be a highly modular library and is not meant to be imported. Our repository focuses on simplicity and clarity in its implementations while utilising the advantages offered by JAX such as `pmap` and `vmap`, making it an excellent resource for researchers and practitioners to build upon. Stoix follows a similar design philosophy to [CleanRL][cleanrl] and [PureJaxRL][purejaxrl], where we allow for some code duplication to enable readability, easy reuse, and fast adaptation. A notable difference between Stoix and other single-file libraries is that Stoix makes use of abstraction where relevant. It is not intended to be purely educational with research utility as the primary focus. In particular, abstraction is currently used for network architectures, environments, logging, and evaluation.

## Overview 🦜

Stoix currently offers the following building blocks for Single-Agent RL research:

### Implementations of Algorithms 🥑

| Algorithm      | Code | Reference|
| ----------- | -----------   | -----|
| [DQN](https://arxiv.org/abs/1312.5602)  | :material-github: [`dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dqn.py) |  :material-file-document: [paper](/rl_algorithms/dqn) |
| [Double DQN (DDQN)](https://arxiv.org/abs/1509.06461)  | :material-github: [`ddqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_ddqn.py) |  :material-file-document: [paper](/rl_algorithms/ddqn) |
| [Dueling DQN](https://arxiv.org/abs/1511.06581)  | :material-github: [`dueling_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dueling_dqn.py) |  :material-file-document: [paper](/rl_algorithms/dueling_dqn) |
| [Categorical DQN (C51)](https://arxiv.org/abs/1707.06887)  | :material-github: [`c51.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_c51.py) |  :material-file-document: [paper](/rl_algorithms/c51) |
| [Munchausen DQN (M-DQN)](https://arxiv.org/abs/2007.14430)  | :material-github: [`mdqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_mdqn.py) |  :material-file-document: [paper](/rl_algorithms/mdqn) |
| [Quantile Regression DQN (QR-DQN)](https://arxiv.org/abs/1710.10044)  | :material-github: [`qr_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_qr_dqn.py) |  :material-file-document: [paper](/rl_algorithms/qr_dqn) |
| [DQN with Regularized Q-learning (DQN-Reg)](https://arxiv.org/abs/2101.03958)  | :material-github: [`dqn_reg.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dqn_reg.py) |  :material-file-document: [paper](/rl_algorithms/dqn_reg) |
| [REINFORCE With Baseline](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)  | :material-github: [`reinforce_baseline.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/vpg/ff_reinforce.py) |  :material-file-document: [paper](/rl_algorithms/reinforce_baseline) |
| [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)  | :material-github: [`ddpg.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/ff_ddpg.py) |  :material-file-document: [paper](/rl_algorithms/ddpg) |
| [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477)  | :material-github: [`td3.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/ff_td3.py) |  :material-file-document: [paper](/rl_algorithms/td3) |
| [Distributed Distributional DDPG (D4PG)](https://arxiv.org/abs/1804.08617)  | :material-github: [`d4pg.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/d4pg.py) |  :material-file-document: [paper](/rl_algorithms/d4pg) |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)  | :material-github: [`sac.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/sac.py) |  :material-file-document: [paper](/rl_algorithms/sac/ff_sac.py) |
| [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)  | :material-github: [`ppo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ppo/ff_ppo.py) |  :material-file-document: [paper](/rl_algorithms/ppo) |
| [Discovered Policy Optimization (DPO)](https://arxiv.org/abs/2210.05639)  | :material-github: [`dpo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ppo/ff_dpo_continuous.py) |  :material-file-document: [paper](/rl_algorithms/dpo) |
| [Maximum a Posteriori Policy Optimisation (MPO)](https://arxiv.org/abs/1806.06920)  | :material-github: [`mpo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/mpo/ff_mpo.py) |  :material-file-document: [paper](/rl_algorithms/mpo) |
| [On-Policy Maximum a Posteriori Policy Optimisation (V-MPO)](https://arxiv.org/abs/1909.12238)  | :material-github: [`v_mpo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/v_mpo.py) |  :material-file-document: [paper](/rl_algorithms/mpo/ff_vmpo.py) |
| [Advantage-Weighted Regression (AWR)](https://arxiv.org/abs/1910.00177)  | :material-github: [`awr.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/awr.py) |  :material-file-document: [paper](/rl_algorithms/awr) |
| [AlphaZero](https://arxiv.org/abs/1712.01815)  | :material-github: [`alphazero.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/alphazero.py) |  :material-file-document: [paper](/rl_algorithms/search/ff_az.py) |
| [MuZero](https://arxiv.org/abs/1911.08265)  | :material-github: [`muzero.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/muzero.py) |  :material-file-document: [paper](/rl_algorithms/search/ff_mz.py) |
| [Sampled Alpha/Mu-Zero](https://arxiv.org/abs/2104.06303)  | :material-github: [`sampled_alpha_muzero.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/search/ff_sampled_mz.py.py) |  :material-file-document: [paper](/rl_algorithms/sampled_alpha_muzero) |

### Environment Wrappers 🍬

Stoix offers wrappers for [Gymnax][gymnax], [Jumanji][jumanji], [Brax][brax], [XMinigrid][xminigrid], [Craftax][craftax] and even [JAXMarl][jaxmarl] (although using Centralised Controllers).

### Statistically Robust Evaluation 🧪

Stoix natively supports logging to json files which adhere to the standard suggested by [Gorsane et al. (2022)][toward_standard_eval]. This enables easy downstream experiment plotting and aggregation using the tools found in the [MARL-eval][marl_eval] library.

## Performance and Speed 🚀

As the code in Stoix (at the time of creation) was in essence a port of [Mava][mava], for further speed comparisons we point to their repo. Additionally, we refer to the PureJaxRL blog post [here](https://chrislu.page/blog/meta-disco/) where the speed benefits of end-to-end JAX systems are discussed.

Below we provide some plots illustrating that Stoix performs equally to that of [PureJaxRL][purejaxrl] but with the added benefit of the code being already set up for `pmap` distribution over devices as well as the other features provided (algorithm implementations, logging, config system, etc).

<p align="center">
<img src="images/ppo_compare.png" alt="ppo" width="45%"/> <img src="images/dqn_compare.png" alt="dqn" width="45%"/>
</p>

I've also included a plot of the training time for 5e5 steps of PPO as one scales the number of environments. PureJaxRL does not pmap and thus runs on a single a device.

<p align="center">
  <img src="images/env_scaling.png" alt="env_scaling" width="750"/>
</p>

Lastly, please keep in mind for practical use that current networks and hyperparameters for algorithms have not been tuned.

## Roadmap 🛤️

We plan to iteratively expand Stoix in the following increments:

#### 1. 🌴 Support for more environments as they become available

#### 2. 🔁 More robust recurrent systems

- Add recurrent variants of all systems
- Allow easy interchangability of recurrent cells/architecture via config

#### 3. 📊 Benchmarks on more environments

- Create leaderboard of algorithms

#### 4. 🦾 More algorithm implementations

- Muesli - [Paper](https://arxiv.org/abs/2104.06159)
- DreamerV3 - [Paper](https://arxiv.org/abs/2301.04104)
- R2D2 - [Paper](https://openreview.net/pdf?id=r1lyTjAqYX)
- Rainbow - [Paper](https://arxiv.org/abs/1710.02298)

#### 5. 🎮 Self-play 2-player Systems for board games

Please do follow along as we develop this next phase!

## Citing Stoix 📚

If you use Stoix in your work, please cite us:

```bibtex
@software{toledo2024stoix,
author = {Toledo, Edan},
doi = {10.5281/zenodo.10916258},
month = apr,
title = {{Stoix: Distributed Single-Agent Reinforcement Learning End-to-End in JAX}},
url = {https://github.com/EdanToledo/Stoix},
version = {v0.0.1},
year = {2024}
}
```

## Acknowledgements 🙏

We would like to thank the authors and developers of [Mava](mava) as this was essentially a port of their repo at the time of creation. This helped set up a lot of the infracstructure of logging, evaluation and other utilities.

## See Also 🔎

**Related JAX Libraries** In particular, we suggest users check out the following repositories:

- 🦁 [Mava](https://github.com/instadeepai/Mava): Distributed Multi-Agent Reinforcement Learning in JAX.
- 🔌 [OG-MARL](https://github.com/instadeepai/og-marl): datasets with baselines for offline MARL in JAX.
- 🌴 [Jumanji](https://github.com/instadeepai/jumanji): a diverse suite of scalable reinforcement learning environments in JAX.
- 😎 [Matrax](https://github.com/instadeepai/matrax): a collection of matrix games in JAX.
- 🔦 [Flashbax](https://github.com/instadeepai/flashbax): accelerated replay buffers in JAX.
- 📈 [MARL-eval](https://github.com/instadeepai/marl-eval): standardised experiment data aggregation and visualisation for MARL.
- 🦊 [JaxMARL](https://github.com/flairox/jaxmarl): accelerated MARL environments with baselines in JAX.
- 🌀 [DeepMind Anakin][anakin_paper] for the Anakin podracer architecture to train RL agents at scale.
- ♟️ [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
- 🔼 [Minimax](https://github.com/facebookresearch/minimax/): JAX implementations of autocurricula baselines for RL.

[jumanji]: https://github.com/instadeepai/jumanji
[cleanrl]: https://github.com/vwxyzjn/cleanrl
[purejaxrl]: https://github.com/luchris429/purejaxrl
[anakin_paper]: https://arxiv.org/abs/2104.06272
[mava]: https://github.com/instadeepai/Mava
[jaxmarl]: https://github.com/flairox/jaxmarl
[toward_standard_eval]: https://arxiv.org/pdf/2209.10485.pdf
[marl_eval]: https://github.com/instadeepai/marl-eval
[gymnax]: https://github.com/RobertTLange/gymnax/
[brax]: https://github.com/google/brax
[xminigrid]: https://github.com/corl-team/xland-minigrid/
[craftax]: https://github.com/MichaelTMatthews/Craftax

!!! warning

    Disclaimer: This is not an official InstaDeep product nor is any of the work putforward associated with InstaDeep in any official capacity.
