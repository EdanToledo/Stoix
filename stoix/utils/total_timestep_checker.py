from colorama import Fore, Style
from omegaconf import DictConfig


def check_total_timesteps(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""

    # If num_devices and update_batch_size are not in the config,
    # usually this means a sebulba config is being used.
    if "num_devices" not in config and "update_batch_size" not in config.arch:
        return check_total_timesteps_sebulba(config)
    else:
        return check_total_timesteps_anakin(config)


def check_total_timesteps_anakin(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""

    print(f"{Fore.YELLOW}{Style.BRIGHT}Using Anakin System!{Style.RESET_ALL}")

    assert config.arch.total_num_envs % (config.num_devices * config.arch.update_batch_size) == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total number of environments "
        + f"should be divisible by the n_devices*update_batch_size!{Style.RESET_ALL}"
    )
    config.arch.num_envs = int(
        config.arch.total_num_envs // (config.num_devices * config.arch.update_batch_size)
    )  # Number of environments per device

    if config.arch.total_timesteps is None:
        config.arch.total_timesteps = (
            config.num_devices
            * config.arch.num_updates
            * config.system.rollout_length
            * config.arch.update_batch_size
            * config.arch.num_envs
        )
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Changing the total number of timesteps "
            + f"to {config.arch.total_timesteps}: If you want to train"
            + " for a specific number of timesteps, please set num_updates to None!"
            + f"{Style.RESET_ALL}"
        )
    else:
        config.arch.num_updates = (
            config.arch.total_timesteps
            // config.system.rollout_length
            // config.arch.update_batch_size
            // config.arch.num_envs
            // config.num_devices
        )
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Changing the number of updates "
            + f"to {config.arch.num_updates}: If you want to train"
            + " for a specific number of updates, please set total_timesteps to None!"
            + f"{Style.RESET_ALL}"
        )

    # Calculate the actual number of timesteps that will be run
    num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        config.num_devices
        * num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )
    total_actual_timesteps = steps_per_rollout * config.arch.num_evaluation
    print(
        f"{Fore.RED}{Style.BRIGHT}Warning: Due to the interaction of various factors such as "
        f"rollout length, number of evaluations, etc... the actual number of timesteps that "
        f"will be run is {total_actual_timesteps}! This is a difference of "
        f"{config.arch.total_timesteps - total_actual_timesteps} timesteps! To change this, "
        f"see total_timestep_checker.py in the utils folder. "
        f"{Style.RESET_ALL}"
    )

    return config

# TODO(Edan): Make this better
def check_total_timesteps_sebulba(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""

    print(f"{Fore.YELLOW}{Style.BRIGHT}Using Sebulba System!{Style.RESET_ALL}")

    assert (
        config.arch.total_num_envs % (config.num_actor_devices * config.arch.actor.actor_per_device)
        == 0
    ), (
        f"{Fore.RED}{Style.BRIGHT}The total number of environments "
        + f"should be divisible by the number of actor devices * actor_per_device!{Style.RESET_ALL}"
    )
    # We first simply take the total number of envs and divide by the number of actor devices
    # to get the number of envs per actor device
    num_envs_per_actor_device = config.arch.total_num_envs // config.num_actor_devices
    # We then divide this by the number of actors per device to get the number of envs per actor
    num_envs_per_actor = int(num_envs_per_actor_device // config.arch.actor.actor_per_device)
    config.arch.actor.num_envs_per_actor = num_envs_per_actor
    config.arch.learner_parallel_env_consumption = config.arch.actor.num_envs_per_actor * config.arch.actor.actor_per_device * config.num_actor_devices
    config.arch.local_batch_size = int(config.system.rollout_length * config.arch.learner_parallel_env_consumption)
    config.arch.global_batch_size = config.arch.local_batch_size * config.arch.world_size
    
    # We base the total number of timesteps based on the number of steps the learner consumes
    if config.arch.total_timesteps is None:
        config.arch.total_timesteps = (
            config.arch.num_updates
            * config.system.rollout_length
            * config.arch.actor.num_envs_per_actor
        )
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Changing the total number of timesteps "
            + f"to {config.arch.total_timesteps}: If you want to train"
            + " for a specific number of timesteps, please set num_updates to None!"
            + f"{Style.RESET_ALL}"
        )
    else:
        config.arch.num_updates = int(config.arch.total_timesteps // config.arch.global_batch_size)
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Changing the number of updates "
            + f"to {config.arch.num_updates}: If you want to train"
            + " for a specific number of updates, please set total_timesteps to None!"
            + f"{Style.RESET_ALL}"
        )

    # Calculate the number of updates per evaluation
    num_evaluation = config.arch.num_evaluation if config.arch.num_evaluation > 0 else 1
    config.arch.num_updates_per_eval = int(config.arch.num_updates // num_evaluation)
    # Get the number of steps consumed by the learner per evaluation
    total_actual_timesteps = config.arch.local_batch_size * config.arch.num_updates_per_eval * num_evaluation
    print(
        f"{Fore.RED}{Style.BRIGHT}Warning: Due to the interaction of various factors such as "
        f"rollout length, number of evaluations, etc... the actual number of timesteps that "
        f"will be run is {total_actual_timesteps}! This is a difference of "
        f"{config.arch.total_timesteps - total_actual_timesteps} timesteps! To change this, "
        f"see total_timestep_checker.py in the utils folder. "
        f"{Style.RESET_ALL}"
    )

    assert (
        config.arch.num_updates > num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    assert config.arch.learner_parallel_env_consumption % config.num_learner_devices == 0

    return config
