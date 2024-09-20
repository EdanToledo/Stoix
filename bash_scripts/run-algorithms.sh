#!/bin/bash

# A simple script to run various algorithms - used for testing purposes

echo "Running All Algorithms..."

# Test a subset of algorithms
python stoix/systems/ppo/anakin/ff_ppo.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/ppo/anakin/ff_ppo_continuous.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/q_learning/ff_dqn.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/q_learning/ff_ddqn.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/q_learning/ff_mdqn.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/q_learning/ff_c51.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/q_learning/ff_qr_dqn.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/sac/ff_sac.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/ddpg/ff_ddpg.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/ddpg/ff_td3.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/vpg/ff_reinforce.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/awr/ff_awr.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/mpo/ff_mpo.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/mpo/ff_vmpo.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/search/ff_az.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8
python stoix/systems/search/ff_mz.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8

# Test a subset of networks
python stoix/systems/ppo/anakin/ff_ppo.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8 env=gymnax/breakout env.wrapper=null network=cnn network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False
python stoix/systems/ppo/anakin/ff_ppo.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8 env=gymnax/breakout env.wrapper=null network=visual_resnet network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False
python stoix/systems/ppo/anakin/ff_ppo.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8 network=mlp_resnet
python stoix/systems/q_learning/ff_dqn.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8 network=mlp_dueling_dqn
python stoix/systems/q_learning/ff_c51.py arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8 network=mlp_dueling_c51
