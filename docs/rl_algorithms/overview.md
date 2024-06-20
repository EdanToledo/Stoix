# Overview 🦜

Stoix currently offers the following building blocks for Single-Agent RL research:

| Algorithm      | Code | Reference|
| ----------- | -----------   | -----|
| [DQN](https://arxiv.org/abs/1312.5602)  | :material-github: [`dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dqn.py) |  :material-file-document: [docs](/rl_algorithms/dqn) |
| [Double DQN (DDQN)](https://arxiv.org/abs/1509.06461)  | :material-github: [`ddqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_ddqn.py) |  :material-file-document: [docs](/rl_algorithms/ddqn) |
| [Dueling DQN](https://arxiv.org/abs/1511.06581)  | :material-github: [`dueling_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dueling_dqn.py) |  :material-file-document: [docs](/rl_algorithms/dueling_dqn) |
| [Categorical DQN (C51)](https://arxiv.org/abs/1707.06887)  | :material-github: [`c51.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_c51.py) |  :material-file-document: [docs](/rl_algorithms/c51) |
| [Munchausen DQN (M-DQN)](https://arxiv.org/abs/2007.14430)  | :material-github: [`mdqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_mdqn.py) |  :material-file-document: [docs](/rl_algorithms/mdqn) |
| [Quantile Regression DQN (QR-DQN)](https://arxiv.org/abs/1710.10044)  | :material-github: [`qr_dqn.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_qr_dqn.py) |  :material-file-document: [docs](/rl_algorithms/qr_dqn) |
| [DQN with Regularized Q-learning (DQN-Reg)](https://arxiv.org/abs/2101.03958)  | :material-github: [`dqn_reg.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_dqn_reg.py) |  :material-file-document: [docs](/rl_algorithms/dqn_reg) |
| [Rainbow DQN](https://arxiv.org/abs/1710.02298)  | :material-github: [`rainbow.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/q_learning/ff_rainbow.py) |  :material-file-document: [docs](/rl_algorithms/rainbow_dqn) |
| [REINFORCE With Baseline](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)  | :material-github: [`reinforce_baseline.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/vpg/ff_reinforce.py) |  :material-file-document: [docs](/rl_algorithms/reinforce_baseline) |
| [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)  | :material-github: [`ddpg.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/ff_ddpg.py) |  :material-file-document: [docs](/rl_algorithms/ddpg) |
| [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477)  | :material-github: [`td3.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/ff_td3.py) |  :material-file-document: [docs](/rl_algorithms/td3) |
| [Distributed Distributional DDPG (D4PG)](https://arxiv.org/abs/1804.08617)  | :material-github: [`d4pg.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ddpg/d4pg.py) |  :material-file-document: [docs](/rl_algorithms/d4pg) |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)  | :material-github: [`sac.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/sac.py) |  :material-file-document: [docs](/rl_algorithms/sac/ff_sac.py) |
| [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)  | :material-github: [`ppo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ppo/ff_ppo.py) |  :material-file-document: [docs](/rl_algorithms/ppo) |
| [Discovered Policy Optimization (DPO)](https://arxiv.org/abs/2210.05639)  | :material-github: [`dpo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/ppo/ff_dpo_continuous.py) |  :material-file-document: [docs](/rl_algorithms/dpo) |
| [Maximum a Posteriori Policy Optimisation (MPO)](https://arxiv.org/abs/1806.06920)  | :material-github: [`mpo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/mpo/ff_mpo.py) |  :material-file-document: [docs](/rl_algorithms/mpo) |
| [On-Policy Maximum a Posteriori Policy Optimisation (V-MPO)](https://arxiv.org/abs/1909.12238)  | :material-github: [`v_mpo.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/v_mpo.py) |  :material-file-document: [docs](/rl_algorithms/mpo/ff_vmpo.py) |
| [Advantage-Weighted Regression (AWR)](https://arxiv.org/abs/1910.00177)  | :material-github: [`awr.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/awr.py) |  :material-file-document: [docs](/rl_algorithms/awr) |
| [AlphaZero](https://arxiv.org/abs/1712.01815)  | :material-github: [`alphazero.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/alphazero.py) |  :material-file-document: [docs](/rl_algorithms/search/ff_az.py) |
| [MuZero](https://arxiv.org/abs/1911.08265)  | :material-github: [`muzero.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/muzero.py) |  :material-file-document: [docs](/rl_algorithms/search/ff_mz.py) |
| [Sampled Alpha/Mu-Zero](https://arxiv.org/abs/2104.06303)  | :material-github: [`sampled_alpha_muzero.py`](https://github.com/EdanToledo/Stoix/blob/main/stoix/systems/search/ff_sampled_mz.py.py) |  :material-file-document: [docs](/rl_algorithms/sampled_alpha_muzero) |
