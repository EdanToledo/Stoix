from colorama import Fore, Style
from omegaconf import DictConfig


def check_total_timesteps(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""

    if config.arch.total_timesteps is None:
        config.arch.total_timesteps = (
            config.num_devices
            * config.arch.num_updates
            * config.system.rollout_length
            * config.system.update_batch_size
            * config.arch.total_num_envs
        )
    else:
        config.arch.num_updates = (
            config.arch.total_timesteps
            // config.system.rollout_length
            // config.system.update_batch_size
            // config.arch.total_num_envs
            // config.num_devices
        )
        print(
            f"{Fore.RED}{Style.BRIGHT} Changing the number of updates "
            + f"to {config.arch.num_updates}: If you want to train"
            + " for a specific number of updates, please set total_timesteps to None!"
            + f"{Style.RESET_ALL}"
        )
    assert config.arch.total_num_envs % config.num_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total number of environments "
        + "should be divisible by the number of devices!{Style.RESET_ALL}"
    )
    config.arch.num_envs = (
        config.arch.total_num_envs // config.num_devices
    )  # Number of environments per device
    return config
