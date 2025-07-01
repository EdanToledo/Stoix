#!/bin/bash

# Improved script to run various algorithms for testing purposes
# Exit immediately if any command fails, treat unset variables as errors
set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="${SCRIPT_DIR}/test_logs"
readonly TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
readonly LOG_FILE="${LOG_DIR}/algorithm_test_${TIMESTAMP}.log"

# Common parameters for all tests
readonly COMMON_PARAMS="arch.total_timesteps=300 arch.total_num_envs=8 arch.num_evaluation=1 system.rollout_length=8"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Counters
total_tests=0
passed_tests=0
failed_tests=0

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Function to log messages
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "${LOG_FILE}"
}

# Function to print colored output
print_color() {
    local color="$1"
    shift
    printf "${color}%s${NC}\n" "$*"
}

# Function to run a single test
run_test() {
    local test_name="$1"
    local command="$2"

    ((total_tests++))

    log "INFO" "Starting test: $test_name"
    print_color "$BLUE" "Running: $test_name"

    # Run the command and capture output
    if timeout 1000 bash -c "$command" >> "${LOG_FILE}" 2>&1; then
        ((passed_tests++))
        print_color "$GREEN" "✓ PASSED: $test_name"
        log "INFO" "Test passed: $test_name"
    else
        ((failed_tests++))
        print_color "$RED" "✗ FAILED: $test_name"
        log "ERROR" "Test failed: $test_name"
        # Optionally, you can choose to continue or exit here
        # For CI/CD, you might want to continue to see all failures
    fi
    echo
}

# Function to run algorithm tests
run_algorithm_tests() {
    log "INFO" "Starting algorithm tests"
    print_color "$YELLOW" "=== Running Algorithm Tests ==="

    # Define test cases as associative arrays would be ideal, but for compatibility:
    # Format: "test_name|command"
    local tests=(
        "PPO Feedforward|python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS"
        "PPO Continuous|python stoix/systems/ppo/anakin/ff_ppo_continuous.py $COMMON_PARAMS"
        "DQN|python stoix/systems/q_learning/ff_dqn.py $COMMON_PARAMS"
        "Double DQN|python stoix/systems/q_learning/ff_ddqn.py $COMMON_PARAMS"
        "Munchausen DQN|python stoix/systems/q_learning/ff_mdqn.py $COMMON_PARAMS"
        "C51|python stoix/systems/q_learning/ff_c51.py $COMMON_PARAMS"
        "QR-DQN|python stoix/systems/q_learning/ff_qr_dqn.py $COMMON_PARAMS"
        "R2D2|python stoix/systems/q_learning/rec_r2d2.py $COMMON_PARAMS system.period=4 system.burn_in_length=4 system.total_buffer_size=10000 system.total_batch_size=32"
        "SAC|python stoix/systems/sac/ff_sac.py $COMMON_PARAMS"
        "DDPG|python stoix/systems/ddpg/ff_ddpg.py $COMMON_PARAMS"
        "TD3|python stoix/systems/ddpg/ff_td3.py $COMMON_PARAMS"
        "REINFORCE|python stoix/systems/vpg/ff_reinforce.py $COMMON_PARAMS"
        "AWR|python stoix/systems/awr/ff_awr.py $COMMON_PARAMS"
        "MPO|python stoix/systems/mpo/ff_mpo.py $COMMON_PARAMS"
        "V-MPO|python stoix/systems/mpo/ff_vmpo.py $COMMON_PARAMS"
        "AlphaZero|python stoix/systems/search/ff_az.py $COMMON_PARAMS"
        "MuZero|python stoix/systems/search/ff_mz.py $COMMON_PARAMS"
        "SPO|python stoix/systems/spo/ff_spo.py $COMMON_PARAMS"
        "SPO Continuous|python stoix/systems/spo/ff_spo_continuous.py $COMMON_PARAMS"
    )

    for test in "${tests[@]}"; do
        IFS='|' read -r test_name command <<< "$test"
        run_test "$test_name" "$command"
    done
}

# Function to run network tests
run_network_tests() {
    log "INFO" "Starting network tests"
    print_color "$YELLOW" "=== Running Network Tests ==="

    local network_tests=(
        "CNN Network|python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS env=gymnax/breakout env.wrapper=null network=cnn network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False"
        "Visual ResNet|python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS env=gymnax/breakout env.wrapper=null network=visual_resnet network.actor_network.pre_torso.channel_first=False network.critic_network.pre_torso.channel_first=False"
        "MLP ResNet|python stoix/systems/ppo/anakin/ff_ppo.py $COMMON_PARAMS network=mlp_resnet"
        "Dueling DQN|python stoix/systems/q_learning/ff_dqn.py $COMMON_PARAMS network=mlp_dueling_dqn"
        "Dueling C51|python stoix/systems/q_learning/ff_c51.py $COMMON_PARAMS network=mlp_dueling_c51"
    )

    for test in "${network_tests[@]}"; do
        IFS='|' read -r test_name command <<< "$test"
        run_test "$test_name" "$command"
    done
}

# Function to run Sebulba tests
run_sebulba_tests() {
    log "INFO" "Starting Sebulba tests"
    print_color "$YELLOW" "=== Running Sebulba Tests ==="

    local sebulba_params="arch.actor.device_ids=[0] arch.actor.actor_per_device=1 arch.learner.device_ids=[0] arch.evaluator_device_id=0"
    local sebulba_tests=(
        "Sebulba PPO|python stoix/systems/ppo/sebulba/ff_ppo.py $COMMON_PARAMS $sebulba_params"
        "Sebulba IMPALA|python stoix/systems/impala/sebulba/ff_impala.py $COMMON_PARAMS $sebulba_params"
    )

    for test in "${sebulba_tests[@]}"; do
        IFS='|' read -r test_name command <<< "$test"
        run_test "$test_name" "$command"
    done
}

# Function to print summary
print_summary() {
    log "INFO" "Test run completed"
    print_color "$YELLOW" "=== Test Summary ==="
    echo "Total tests: $total_tests"
    print_color "$GREEN" "Passed: $passed_tests"
    print_color "$RED" "Failed: $failed_tests"
    echo "Log file: $LOG_FILE"

    if [[ $failed_tests -gt 0 ]]; then
        print_color "$RED" "Some tests failed. Check the log file for details."
        return 1
    else
        print_color "$GREEN" "All tests passed!"
        return 0
    fi
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    --algorithms    Run only algorithm tests
    --networks      Run only network tests
    --sebulba       Run only Sebulba tests
    --timeout N     Set timeout for each test (default: 300 seconds)

Examples:
    $0                  # Run all tests
    $0 --algorithms     # Run only algorithm tests
    $0 --verbose        # Run with verbose output
EOF
}

# Parse command line arguments
VERBOSE=false
RUN_ALGORITHMS=true
RUN_NETWORKS=true
RUN_SEBULBA=true
TIMEOUT=1000

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --algorithms)
            RUN_ALGORITHMS=true
            RUN_NETWORKS=false
            RUN_SEBULBA=false
            shift
            ;;
        --networks)
            RUN_ALGORITHMS=false
            RUN_NETWORKS=true
            RUN_SEBULBA=false
            shift
            ;;
        --sebulba)
            RUN_ALGORITHMS=false
            RUN_NETWORKS=false
            RUN_SEBULBA=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log "INFO" "Starting algorithm testing script"
    print_color "$BLUE" "=== Algorithm Testing Suite ==="
    print_color "$BLUE" "Timestamp: $(date)"
    print_color "$BLUE" "Log file: $LOG_FILE"
    echo

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_color "$RED" "Error: Python is not installed or not in PATH"
        exit 1
    fi

    # Run tests based on flags
    if [[ "$RUN_ALGORITHMS" == true ]]; then
        run_algorithm_tests
    fi

    if [[ "$RUN_NETWORKS" == true ]]; then
        run_network_tests
    fi

    if [[ "$RUN_SEBULBA" == true ]]; then
        run_sebulba_tests
    fi

    # Print summary and exit with appropriate code
    if print_summary; then
        exit 0
    else
        exit 1
    fi
}

# Trap to handle script interruption
trap 'print_color "$RED" "Script interrupted!"; exit 130' INT TERM

# Run main function
main "$@"
