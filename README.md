<p align="center">
    <a href="docs/images/stoix.png">
        <img src="docs/images/stoix.png" alt="Stoix logo" width="30%"/>
    </a>
</p>

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
</div>

<h2 align="center">
    <p>Distributed Single-Agent Reinforcement Learning in JAX</p>
</h2>

<div align="center">

**_stoic - a person who can endure pain or hardship without showing their feelings or complaining._**

</div>

## Welcome to Stoix! 🏛️

Stoix provides simplified code for quickly iterating on ideas in single-agent reinforcement learning with useful implementations of popular single-agent RL algorithms in JAX allowing for easy parallelisation across devices with JAX's `pmap`. All implementations are fully compilable with JAX's `jit` thus making training and environment execution very fast. However, this requires environments written in JAX. Algorithms and their default hyperparameters have not been hyper-optimised for any specific environment and are useful as a starting point for research and/or for initial baselines.

To join us in these efforts, please feel free to reach out, raise issues or read our [contribution guidelines](#contributing-) (or just star 🌟 to stay up to date with the latest developments)!

Stoix is fully in JAX with substantial speed improvement compared to other popular libraries. We currently provide native support for the [Jumanji][jumanji] environment API and wrappers for popular JAX-based RL environments.

## Code Philosophy 🧘

The current code in Stoix is initially **largely** taken and subsequently adapted from [Mava](mava). As Mava develops, Stoix will hopefully adopt their optimisations. Like Mava, Stoix is not designed to be a highly modular library and is not meant to be imported. Our repository focuses on simplicity and clarity in its implementations while utilising the advantages offered by JAX such as `pmap` and `vmap`, making it an excellent resource for researchers and practitioners to build upon. Stoix follows a similar design philosophy to [CleanRL][cleanrl] and [PureJaxRL][purejaxrl], where we allow for some code duplication to enable readability, easy reuse, and fast adaptation.

## Overview 🦜

Stoix currently offers the following building blocks for Single-Agent RL research:

### Implementations of Algorithms 🥑

- **Proximal Policy Optimization (PPO)** - [Paper](https://arxiv.org/abs/1707.06347)
- **Deep Q-Network (DQN)** - [Paper](https://arxiv.org/abs/1312.5602)
- **Categorical DQN (C51)** - [Paper](https://arxiv.org/abs/1707.06887)
- **Munchausen DQN (M-DQN)** [Paper](https://arxiv.org/abs/2007.14430)
- **Quantile Regression DQN (QR-DQN)** - [Paper](https://arxiv.org/abs/1710.10044)
- **DQN with Regularized Q-learning (DQN-Reg)** [Paper](https://arxiv.org/abs/2101.03958)
- **Soft Actor-Critic (SAC)** - [Paper](https://arxiv.org/abs/1801.01290)
- **Deep Deterministic Policy Gradient (DDPG)** - [Paper](https://arxiv.org/abs/1509.02971)
- **Twin Delayed DDPG (TD3)** - [Paper](https://arxiv.org/abs/1802.09477)
- **Distributed Distributional DDPG (D4PG)** - [Paper](https://arxiv.org/abs/1804.08617)
- **AlphaZero** - [Paper](https://arxiv.org/abs/1712.01815)
- **Discovered Policy Optimization (DPO)** [Paper](https://arxiv.org/abs/2210.05639)
- **Maximum a Posteriori Policy Optimisation (MPO)** - [Paper](https://arxiv.org/abs/1806.06920)

### Environment Wrappers 🍬
Stoix offers wrappers for [Gymnax][gymnax], [Jumanji][jumanji], [Brax][brax], [XMinigrid][xminigrid], [Craftax][craftax] and even [JAXMarl][jaxmarl] (although using Centralised Controllers).

### Statistically Robust Evaluation 🧪
Stoix natively supports logging to json files which adhere to the standard suggested by [Gorsane et al. (2022)][toward_standard_eval]. This enables easy downstream experiment plotting and aggregation using the tools found in the [MARL-eval][marl_eval] library.

## Performance and Speed 🚀

As the code in Stoix is in essense a port of [Mava][mava], for speed comparisons we point to their repo. Additionally, please note that current networks and hyperparameters for algorithms have not been tuned.

## Installation 🎬

At the moment Stoix is not meant to be installed as a library, but rather to be used as a research tool.

You can use Stoix by cloning the repo and pip installing as follows:

```bash
git clone https://github.com/EdanToledo/stoix.git
cd stoix
pip install -e .
```

We have tested `Stoix` on Python 3.10. Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version (see the [official installation guide](https://github.com/google/jax#installation)).

## Quickstart ⚡

To get started with training your first Stoix system, simply run one of the system files. e.g.,

```bash
python stoix/systems/ff_ppo.py
```

Stoix makes use of Hydra for config management. In order to see our default system configs please see the `stoix/configs/` directory. A benefit of Hydra is that configs can either be set in config yaml files or overwritten from the terminal on the fly. For an example of running a system on the CartPole environment, the above code can simply be adapted as follows:

```bash
python stoix/systems/ff_ppo.py env=gymnax/cartpole
```

## Contributing 🤝

Please read our [contributing docs](docs/CONTRIBUTING.md) for details on how to submit pull requests, our Contributor License Agreement and community guidelines.

## Roadmap 🛤️

We plan to iteratively expand Stoix in the following increments:

- 🌴 Support for more environments.
- 🔁 More robust recurrent systems.
- 📊 Benchmarks on more environments.
- 🦾 More algorithm implementations.

Please do follow along as we develop this next phase!

## Citing Stoix 📚

If you use Stoix in your work, please cite us:

```bibtex
@misc{toledo2024stoix,
    title={Stoix: Distributed Single-Agent Reinforcement Learning in JAX},
    author={Edan Toledo},
    year={2024},
    url={https://github.com/EdanToledo/Stoix/},
}
```

## Acknowledgements 🙏

We would like to thank the authors and developers of [Mava](mava) as this was essentially a port of their repo at the time of creation.

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

Disclaimer: This is not an official InstaDeep product nor is any of the work putforward associated with InstaDeep in any official capacity.
