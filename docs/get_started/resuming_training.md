# Resuming training and checkpointing 🚦

Checkpointing is essential in reinforcement learning experiments to save and resume training states, ensuring progress is not lost and facilitating model recovery after interruptions. Stoix provides robust checkpointing support using its `Checkpointer` utility.

#### How Checkpointing Works

In Stoix, checkpointing involves periodically saving the learner state, including model parameters and optimizer states.

#### Example Training Script with Checkpointing

The following example illustrates how checkpointing is integrated into a training script in Stoix:

```python
import copy
import time
import jax
import jax.numpy as jnp
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.jax_utils import unreplicate_n_dims

def run_experiment(config):
    # ... (initial setup code)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    for eval_step in range(config.arch.num_evaluation):
        # Train
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log training metrics
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluate
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log evaluation metrics
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        learner_state = learner_output.learner_state

    logger.stop()
    return eval_performance
```

#### Configuration

The `Checkpointer` is configured via Hydra, allowing for flexible adjustments. Below is an example configuration snippet for enabling checkpointing:

```yaml
logger:
  checkpointing:
    save_model: true
    save_args:
      save_interval_steps: 1  # Number of steps between saving checkpoints
      max_to_keep: 1          # Maximum number of checkpoints to keep
    load_model: true
    load_args:
      checkpoint_uid: ""      # Unique identifier for checkpoint to load
```

#### Usage Tips

- **Set Frequent Checkpoints**: Save checkpoints regularly to minimize the risk of data loss.
- **Store Metadata**: Include configuration metadata in checkpoints to ensure reproducibility.
- **Monitor Storage**: Ensure adequate storage space for saving checkpoints, especially for large models.
