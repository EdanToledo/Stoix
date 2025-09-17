#!/bin/bash

# Script will exit immediately if any command fails

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
set -o pipefail  # Return exit status of the last command in the pipe that failed

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to run a command with error handling
run_algorithm() {
    local cmd="$1"
    local description="$2"

    print_status "Running: $description"
    echo "Command: $cmd"

    if ! eval "$cmd"; then
        print_error "Failed to run: $description"
        print_error "Command that failed: $cmd"
        exit 1
    fi

    print_status "Successfully completed: $description"
    echo "---"
}

# Common parameters for easier maintenance
COMMON_PARAMS="arch.total_timesteps=256 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=16"
SEBULBA_PARAMS="$COMMON_PARAMS arch.actor.device_ids=[0] arch.actor.actor_per_device=1 arch.learner.device_ids=[0] arch.evaluator_device_id=0"

echo "=========================================="
echo "Starting Algorithm Test Suite"
echo "=========================================="

# Test core algorithms
print_status "Testing Core Algorithms..."

run_algorithm "python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS" "PPO (Discrete)"
run_algorithm "python stoix/systems/ppo/anakin/ff_ppo_continuous.py $COMMON_PARAMS" "PPO (Continuous)"
run_algorithm "python stoix/systems/q_learning/ff_dqn.py $COMMON_PARAMS" "DQN"
run_algorithm "python stoix/systems/q_learning/ff_ddqn.py $COMMON_PARAMS" "Double DQN"
run_algorithm "python stoix/systems/q_learning/ff_mdqn.py $COMMON_PARAMS" "Multi-step DQN"
run_algorithm "python stoix/systems/q_learning/ff_c51.py $COMMON_PARAMS" "C51"
run_algorithm "python stoix/systems/q_learning/ff_qr_dqn.py $COMMON_PARAMS" "QR-DQN"
run_algorithm "python stoix/systems/q_learning/rec_r2d2.py $COMMON_PARAMS system.period=4 system.burn_in_length=4 system.total_buffer_size=10000 system.total_batch_size=32" "R2D2"
run_algorithm "python stoix/systems/sac/ff_sac.py $COMMON_PARAMS" "SAC"
run_algorithm "python stoix/systems/ddpg/ff_ddpg.py $COMMON_PARAMS" "DDPG"
run_algorithm "python stoix/systems/ddpg/ff_td3.py $COMMON_PARAMS" "TD3"
run_algorithm "python stoix/systems/vpg/ff_reinforce.py $COMMON_PARAMS" "REINFORCE"
run_algorithm "python stoix/systems/awr/ff_awr.py $COMMON_PARAMS" "AWR"
run_algorithm "python stoix/systems/mpo/ff_mpo.py $COMMON_PARAMS" "MPO"
run_algorithm "python stoix/systems/mpo/ff_vmpo.py $COMMON_PARAMS" "V-MPO"
run_algorithm "python stoix/systems/search/ff_az.py $COMMON_PARAMS" "AlphaZero"
run_algorithm "python stoix/systems/search/ff_sampled_az.py $COMMON_PARAMS" "Sampled AlphaZero"
run_algorithm "python stoix/systems/search/ff_mz.py $COMMON_PARAMS" "MuZero"
run_algorithm "python stoix/systems/search/ff_sampled_mz.py $COMMON_PARAMS" "Sampled MuZero"
run_algorithm "python stoix/systems/spo/ff_spo.py $COMMON_PARAMS" "SPO (Discrete)"
run_algorithm "python stoix/systems/spo/ff_spo_continuous.py $COMMON_PARAMS" "SPO (Continuous)"
run_algorithm "python stoix/systems/awr/ff_awr_continuous.py $COMMON_PARAMS" "AWR (Continuous)"
run_algorithm "python stoix/systems/ddpg/ff_d4pg.py $COMMON_PARAMS" "D4PG"
run_algorithm "python stoix/systems/mpo/ff_mpo_continuous.py $COMMON_PARAMS" "MPO (Continuous)"
run_algorithm "python stoix/systems/mpo/ff_vmpo_continuous.py $COMMON_PARAMS" "V-MPO (Continuous)"
run_algorithm "python stoix/systems/ppo/anakin/ff_dpo_continuous.py $COMMON_PARAMS" "DPO (Continuous)"
run_algorithm "python stoix/systems/ppo/anakin/ff_ppo_penalty.py $COMMON_PARAMS" "PPO Penalty"
run_algorithm "python stoix/systems/ppo/anakin/ff_ppo_penalty_continuous.py $COMMON_PARAMS" "PPO Penalty (Continuous)"
run_algorithm "python stoix/systems/ppo/anakin/rec_ppo.py $COMMON_PARAMS system.num_minibatches=1" "Recurrent PPO"
run_algorithm "python stoix/systems/q_learning/ff_dqn_reg.py $COMMON_PARAMS" "DQN Reg"
run_algorithm "python stoix/systems/q_learning/ff_pqn.py $COMMON_PARAMS" "PQN"
run_algorithm "python stoix/systems/q_learning/ff_rainbow.py $COMMON_PARAMS" "Rainbow DQN"
run_algorithm "python stoix/systems/vpg/ff_reinforce_continuous.py $COMMON_PARAMS" "REINFORCE (Continuous)"

# Test different network architectures
print_status "Testing Network Architectures..."

run_algorithm "python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS env=gymnax/breakout env.wrapper=null network=cnn network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False" "PPO with CNN (Breakout)"
run_algorithm "python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS env=gymnax/breakout env.wrapper=null network=visual_resnet network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False" "PPO with Visual ResNet (Breakout)"
run_algorithm "python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS network=mlp_resnet" "PPO with MLP ResNet"
run_algorithm "python stoix/systems/q_learning/ff_dqn.py $COMMON_PARAMS network=mlp_dueling_dqn" "DQN with Dueling Network"
run_algorithm "python stoix/systems/q_learning/ff_c51.py $COMMON_PARAMS network=mlp_dueling_c51" "C51 with Dueling Network"

# Test Sebulba algorithms (distributed)
print_status "Testing Sebulba (Distributed) Algorithms..."

run_algorithm "python stoix/systems/ppo/sebulba/ff_ppo.py $SEBULBA_PARAMS" "Sebulba PPO"
run_algorithm "python stoix/systems/impala/sebulba/ff_impala.py $SEBULBA_PARAMS" "Sebulba IMPALA"
run_algorithm "python stoix/systems/impala/sebulba/ff_impala_shared_torso.py $SEBULBA_PARAMS" "Sebulba IMPALA (Shared Torso)"

echo "=========================================="
print_status "All algorithms completed successfully!"
echo "=========================================="
