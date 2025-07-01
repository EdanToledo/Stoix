#!/bin/bash

# A script to run various algorithms in parallel for testing purposes.
# It ensures all tests are run and reports a summary of any failures.

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error.
set -uo pipefail

# --- Configuration ---

# A temporary file to log the commands of failed tests.
FAILURE_LOG=$(mktemp)
trap 'rm -f -- "$FAILURE_LOG"' EXIT

# --- Test Definitions ---

echo "🚀 Launching All Algorithm Tests in Parallel..."

# Base arguments used for most tests.
COMMON_ARGS="arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8"

# Group tests into arrays for easy management.
# The `&` at the end of each command is crucial for background execution.
ALGO_TESTS=(
    "stoix/systems/ppo/anakin/ff_ppo.py ${COMMON_ARGS}"
    "stoix/systems/ppo/anakin/ff_ppo_continuous.py ${COMMON_ARGS}"
    "stoix/systems/q_learning/ff_dqn.py ${COMMON_ARGS}"
    "stoix/systems/q_learning/ff_ddqn.py ${COMMON_ARGS}"
    "stoix/systems/q_learning/ff_mdqn.py ${COMMON_ARGS}"
    "stoix/systems/q_learning/ff_c51.py ${COMMON_ARGS}"
    "stoix/systems/q_learning/ff_qr_dqn.py ${COMMON_ARGS}"
    "stoix/systems/q_learning/rec_r2d2.py ${COMMON_ARGS} system.period=4 system.burn_in_length=4 system.total_buffer_size=10000 system.total_batch_size=32"
    "stoix/systems/sac/ff_sac.py ${COMMON_ARGS}"
    "stoix/systems/ddpg/ff_ddpg.py ${COMMON_ARGS}"
    "stoix/systems/ddpg/ff_td3.py ${COMMON_ARGS}"
    "stoix/systems/vpg/ff_reinforce.py ${COMMON_ARGS}"
    "stoix/systems/awr/ff_awr.py ${COMMON_ARGS}"
    "stoix/systems/mpo/ff_mpo.py ${COMMON_ARGS}"
    "stoix/systems/mpo/ff_vmpo.py ${COMMON_ARGS}"
    "stoix/systems/search/ff_az.py ${COMMON_ARGS}"
    "stoix/systems/search/ff_mz.py ${COMMON_ARGS}"
    "stoix/systems/spo/ff_spo.py ${COMMON_ARGS}"
    "stoix/systems/spo/ff_spo_continuous.py ${COMMON_ARGS}"
)

NETWORK_TESTS=(
    "stoix/systems/ppo/anakin/ff_ppo.py ${COMMON_ARGS} env=gymnax/breakout env.wrapper=null network=cnn network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False"
    "stoix/systems/ppo/anakin/ff_ppo.py ${COMMON_ARGS} env=gymnax/breakout env.wrapper=null network=visual_resnet network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False"
    "stoix/systems/ppo/anakin/ff_ppo.py ${COMMON_ARGS} network=mlp_resnet"
    "stoix/systems/q_learning/ff_dqn.py ${COMMON_ARGS} network=mlp_dueling_dqn"
    "stoix/systems/q_learning/ff_c51.py ${COMMON_ARGS} network=mlp_dueling_c51"
)

SEBULBA_TESTS=(
    "stoix/systems/ppo/sebulba/ff_ppo.py ${COMMON_ARGS} arch.actor.device_ids=[0] arch.actor.actor_per_device=1 arch.learner.device_ids=[0] arch.evaluator_device_id=0"
    "stoix/systems/impala/sebulba/ff_impala.py ${COMMON_ARGS} arch.actor.device_ids=[0] arch.actor.actor_per_device=1 arch.learner.device_ids=[0] arch.evaluator_device_id=0"
)

# Combine all tests into one list
ALL_TESTS=("${ALGO_TESTS[@]}" "${NETWORK_TESTS[@]}" "${SEBULBA_TESTS[@]}")

# --- Test Execution ---

# Associative array to map PIDs to their commands
declare -A pid_to_cmd

# Launch all tests in the background
for test_cmd in "${ALL_TESTS[@]}"; do
    python ${test_cmd} &
    pid=$!
    pid_to_cmd[${pid}]="${test_cmd}"
    echo "Launched test with PID ${pid}: python ${test_cmd}"
done

echo ""
echo "⌛ Waiting for all tests to complete..."

# --- Collect Results ---

FAILURES=0
# Wait for each background job to finish and check its exit code
for pid in "${!pid_to_cmd[@]}"; do
    if ! wait "${pid}"; then
        FAILURES=$((FAILURES + 1))
        failed_cmd="python ${pid_to_cmd[$pid]}"
        echo "❌ FAILED (PID: ${pid}): ${failed_cmd}"
        echo "${failed_cmd}" >> "${FAILURE_LOG}"
    fi
done

# --- Final Summary ---

if [ "${FAILURES}" -gt 0 ]; then
    echo ""
    echo "----------------------------------------"
    echo "🚨 Test run failed. ${FAILURES} error(s) found."
    echo "----------------------------------------"
    # The log file will contain the full command of each failed test
    cat "${FAILURE_LOG}"
    echo "----------------------------------------"
    exit 1
else
    echo ""
    echo "----------------------------------------"
    echo "✅ All tests passed successfully!"
    echo "----------------------------------------"
    exit 0
fi
