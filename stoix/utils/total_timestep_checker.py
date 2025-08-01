from colorama import Fore, Style
from omegaconf import DictConfig

# Anakin and Sebulba are two different distributed RL systems.
# Due to implementation details and architecture differences,
# they require different configurations for handling timesteps and updates.


def check_total_timesteps_anakin(config: DictConfig) -> DictConfig:
    """
    Validate and configure timestep-related parameters for the Anakin distributed RL system.

    This function ensures that environment distribution across devices is valid and calculates
    timesteps and update counts based on the provided configuration.

    Args:
        config: Configuration dictionary containing architecture and system parameters

    Returns:
        Updated configuration with calculated values

    Raises:
        AssertionError: If configuration constraints are violated
    """
    _print_anakin_header()

    # Validate and calculate environment distribution
    _validate_env_distribution_anakin(config)
    _calculate_env_distribution_anakin(config)

    # Handle timesteps vs updates configuration
    _configure_timesteps_and_updates_anakin(config)

    # Calculate and warn about actual timesteps
    _calculate_actual_timesteps_anakin(config)

    return config


def _print_anakin_header() -> None:
    """Print the Anakin system header."""
    print(f"{Fore.YELLOW}{Style.BRIGHT}Using Anakin System!{Style.RESET_ALL}")


def _validate_env_distribution_anakin(config: DictConfig) -> None:
    """Validate that environments can be evenly distributed across devices and batches."""
    divisor = config.num_devices * config.arch.update_batch_size

    if config.arch.total_num_envs % divisor != 0:
        raise AssertionError(
            f"{Fore.RED}{Style.BRIGHT}The total number of environments "
            f"({config.arch.total_num_envs}) must be divisible by "
            f"(num_devices * update_batch_size) = {divisor}!{Style.RESET_ALL}"
        )


def _calculate_env_distribution_anakin(config: DictConfig) -> None:
    """Calculate number of environments per device."""
    config.arch.num_envs = int(
        config.arch.total_num_envs // (config.num_devices * config.arch.update_batch_size)
    )


def _configure_timesteps_and_updates_anakin(config: DictConfig) -> None:
    """Configure either total timesteps or number of updates based on what's provided."""
    if config.arch.total_timesteps is None:
        _calculate_timesteps_from_updates_anakin(config)
    else:
        _calculate_updates_from_timesteps_anakin(config)


def _calculate_timesteps_from_updates_anakin(config: DictConfig) -> None:
    """Calculate total timesteps when num_updates is specified."""
    config.arch.total_timesteps = (
        config.num_devices
        * config.arch.num_updates
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Setting total timesteps to {config.arch.total_timesteps:,}\n"
        f"Note: To train for a specific number of timesteps, set num_updates to None!{Style.RESET_ALL}"
    )


def _calculate_updates_from_timesteps_anakin(config: DictConfig) -> None:
    """Calculate number of updates when total_timesteps is specified."""
    config.arch.num_updates = (
        config.arch.total_timesteps
        // config.system.rollout_length
        // config.arch.update_batch_size
        // config.arch.num_envs
        // config.num_devices
    )

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Setting number of updates to {config.arch.num_updates:,}\n"
        f"Note: To train for a specific number of updates, set total_timesteps to None!{Style.RESET_ALL}"
    )


def _calculate_actual_timesteps_anakin(config: DictConfig) -> None:
    """Calculate the actual number of timesteps that will be run and warn about discrepancies."""
    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Number of updates per evaluation: "
        f"{config.arch.num_updates_per_eval:,}{Style.RESET_ALL}"
    )

    # Calculate timesteps per rollout
    steps_per_rollout = (
        config.num_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Calculate total actual timesteps
    total_actual_timesteps = steps_per_rollout * config.arch.num_evaluation

    # Warn about timestep discrepancy
    timestep_difference = config.arch.total_timesteps - total_actual_timesteps
    if timestep_difference != 0:
        _print_timestep_warning(
            config.arch.total_timesteps, total_actual_timesteps, timestep_difference
        )


def check_total_timesteps_sebulba(config: DictConfig) -> DictConfig:
    """
    Validate and configure timestep-related parameters for the Sebulba distributed RL system.

    This function ensures that environment distribution across actors is valid and calculates
    batch sizes, timesteps, and update counts based on the provided configuration.

    Args:
        config: Configuration dictionary containing architecture and system parameters

    Returns:
        Updated configuration with calculated values

    Raises:
        AssertionError: If configuration constraints are violated
    """
    _print_sebulba_header()

    # Validate and calculate environment distribution
    _validate_env_distribution_sebulba(config)
    _calculate_env_distribution_sebulba(config)

    # Calculate batch sizes
    _calculate_batch_sizes_sebulba(config)

    # Handle timesteps vs updates configuration
    _configure_timesteps_and_updates_sebulba(config)

    # Configure evaluation settings
    _configure_evaluation_sebulba(config)

    # Final validations
    _validate_final_configuration_sebulba(config)

    return config


def _print_sebulba_header() -> None:
    """Print the Sebulba system header."""
    print(f"{Fore.YELLOW}{Style.BRIGHT}Using Sebulba System!{Style.RESET_ALL}")


def _validate_env_distribution_sebulba(config: DictConfig) -> None:
    """Validate that environments can be evenly distributed across actors."""
    total_actors = config.num_actor_devices * config.arch.actor.actor_per_device

    if config.arch.total_num_envs % total_actors != 0:
        raise AssertionError(
            f"{Fore.RED}{Style.BRIGHT}The total number of environments "
            f"({config.arch.total_num_envs}) must be divisible by "
            f"(num_actor_devices * actor_per_device) = {total_actors}!{Style.RESET_ALL}"
        )


def _calculate_env_distribution_sebulba(config: DictConfig) -> None:
    """Calculate how environments are distributed across actors."""
    # Calculate environments per actor device
    num_envs_per_actor_device = config.arch.total_num_envs // config.num_actor_devices

    # Calculate environments per individual actor
    num_envs_per_actor = int(num_envs_per_actor_device // config.arch.actor.actor_per_device)
    config.arch.actor.num_envs_per_actor = num_envs_per_actor


def _calculate_batch_sizes_sebulba(config: DictConfig) -> None:
    """Calculate local and global batch sizes for the learner."""
    # Total environments consumed in parallel by the learner
    config.arch.learner_parallel_env_consumption = (
        config.arch.actor.num_envs_per_actor
        * config.arch.actor.actor_per_device
        * config.num_actor_devices
    )

    # Local batch size per learner
    config.arch.local_batch_size = int(
        config.system.rollout_length * config.arch.learner_parallel_env_consumption
    )

    # Global batch size across all learners
    config.arch.global_batch_size = config.arch.local_batch_size * config.arch.world_size


def _configure_timesteps_and_updates_sebulba(config: DictConfig) -> None:
    """Configure either total timesteps or number of updates based on what's provided."""
    if config.arch.total_timesteps is None:
        _calculate_timesteps_from_updates_sebulba(config)
    else:
        _calculate_updates_from_timesteps_sebulba(config)


def _calculate_timesteps_from_updates_sebulba(config: DictConfig) -> None:
    """Calculate total timesteps when num_updates is specified."""
    config.arch.total_timesteps = config.arch.num_updates * config.arch.global_batch_size

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Setting total timesteps to {config.arch.total_timesteps:,}\n"
        f"Note: To train for a specific number of timesteps, set num_updates to None!{Style.RESET_ALL}"
    )


def _calculate_updates_from_timesteps_sebulba(config: DictConfig) -> None:
    """Calculate number of updates when total_timesteps is specified."""
    config.arch.num_updates = int(config.arch.total_timesteps // config.arch.global_batch_size)

    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Setting number of updates to {config.arch.num_updates:,}\n"
        f"Note: To train for a specific number of updates, set total_timesteps to None!{Style.RESET_ALL}"
    )


def _configure_evaluation_sebulba(config: DictConfig) -> None:
    """Configure evaluation intervals and calculate actual timesteps."""
    # Ensure at least one evaluation
    num_evaluation = max(config.arch.num_evaluation, 1)

    # Calculate updates between evaluations
    config.arch.num_updates_per_eval = int(config.arch.num_updates // num_evaluation)

    # Calculate actual timesteps that will be run
    total_actual_timesteps = (
        config.arch.global_batch_size * config.arch.num_updates_per_eval * num_evaluation
    )

    # Warn about timestep discrepancy
    timestep_difference = config.arch.total_timesteps - total_actual_timesteps
    if timestep_difference != 0:
        _print_timestep_warning(
            config.arch.total_timesteps, total_actual_timesteps, timestep_difference
        )


def _print_timestep_warning(expected: int, actual: int, difference: int) -> None:
    """Print warning about timestep discrepancy."""
    print(
        f"{Fore.RED}{Style.BRIGHT}⚠️ Timestep Discrepancy Warning:\n"
        f"   Expected timesteps: {expected:,}\n"
        f"   Actual timesteps:   {actual:,}\n"
        f"   Difference:         {difference:,}\n"
        f"\n"
        f"   This occurs due to the interaction of rollout length, number of evaluations,\n"
        f"   and other factors. To adjust this, see total_timestep_checker.py in utils/\n"
        f"{Style.RESET_ALL}"
    )


def _validate_final_configuration_sebulba(config: DictConfig) -> None:
    """Perform final validation checks on the configuration."""
    # Check evaluation frequency
    num_evaluation = max(config.arch.num_evaluation, 1)
    if config.arch.num_updates <= num_evaluation:
        raise AssertionError(
            f"Number of updates ({config.arch.num_updates}) must be greater than "
            f"number of evaluations ({num_evaluation})."
        )

    # Check learner device distribution
    if config.arch.learner_parallel_env_consumption % config.num_learner_devices != 0:
        raise AssertionError(
            f"Learner parallel env consumption ({config.arch.learner_parallel_env_consumption}) "
            f"must be divisible by number of learner devices ({config.num_learner_devices})."
        )


def check_total_timesteps(config: DictConfig) -> DictConfig:
    """
    Check and configure timesteps based on the system type.

    Args:
        config: Configuration dictionary

    Returns:
        Updated configuration

    Raises:
        ValueError: If system_type is not recognized
    """

    system_type = "sebulba" if "device_ids" in config.arch.get("learner", {}) else "anakin"

    if system_type.lower() == "sebulba":
        return check_total_timesteps_sebulba(config)
    elif system_type.lower() == "anakin":
        return check_total_timesteps_anakin(config)
    else:
        raise ValueError(f"Unknown system type: {system_type}. Must be 'sebulba' or 'anakin'.")
