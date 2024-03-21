from typing import Callable, Optional, Union

from omegaconf import DictConfig


def make_learning_rate_schedule(
    init_lr: float, num_updates: int, num_epochs: int, num_minibatches: int
) -> Callable:
    """Makes a very simple linear learning rate scheduler.

    Args:
        init_lr: initial learning rate.
        num_updates: number of updates.
        num_epochs: number of epochs.
        num_minibatches: number of minibatches.

    Note:
        We use a simple linear learning rate scheduler based on the suggestions from a blog on PPO
        implementation details which can be viewed at http://tinyurl.com/mr3chs4p
        This function can be extended to have more complex learning rate schedules by adding any
        relevant arguments to the system config and then parsing them accordingly here.
    """

    def linear_scedule(count: int) -> float:
        frac: float = 1.0 - (count // (num_epochs * num_minibatches)) / num_updates
        return init_lr * frac

    return linear_scedule


def make_learning_rate(
    init_lr: float, config: DictConfig, num_epochs: int, num_minibatches: Optional[int] = None
) -> Union[float, Callable]:
    """Returns a constant learning rate or a learning rate schedule.

    Args:
        init_lr: initial learning rate.
        config: system configuration.
        num_epochs: number of epochs.
        num_minibatches: number of minibatches.

    Returns:
        A learning rate schedule or fixed learning rate.
    """
    if num_minibatches is None:
        num_minibatches = 1

    if config.system.decay_learning_rates:
        return make_learning_rate_schedule(
            init_lr, config.arch.num_updates, num_epochs, num_minibatches
        )
    else:
        return init_lr
