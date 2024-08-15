from colorama import Fore, Style
from omegaconf import DictConfig


def check_total_timesteps(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""

    # Check if the number of devices is set in the config
    # If not, it is assumed that the number of devices is 1
    # For the case of using a sebulba config, the number of
    # devices is set to 1 for the calculation
    # of the number of environments per device, etc
    if "num_devices" not in config:
        num_devices = 1
    else:
        num_devices = config.num_devices

    # If update_batch_size is not in the config, usually this means a sebulba config is being used.
    if "update_batch_size" not in config.arch:
        update_batch_size = 1
        print(f"{Fore.YELLOW}{Style.BRIGHT}Using Sebulba System!{Style.RESET_ALL}")
    else:
        update_batch_size = config.arch.update_batch_size
        print(f"{Fore.YELLOW}{Style.BRIGHT}Using Anakin System!{Style.RESET_ALL}")

    assert config.arch.total_num_envs % (num_devices * update_batch_size) == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total number of environments "
        + f"should be divisible by the n_devices*update_batch_size!{Style.RESET_ALL}"
    )
    config.arch.num_envs = int(
        config.arch.total_num_envs // (num_devices * update_batch_size)
    )  # Number of environments per device

    if config.arch.total_timesteps is None:
        config.arch.total_timesteps = (
            num_devices
            * config.arch.num_updates
            * config.system.rollout_length
            * update_batch_size
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
            // update_batch_size
            // config.arch.num_envs
            // num_devices
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
        num_devices
        * num_updates_per_eval
        * config.system.rollout_length
        * update_batch_size
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
