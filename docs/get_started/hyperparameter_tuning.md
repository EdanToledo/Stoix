# Hyperparameter Tuning 🚀

In Stoix, hyperparameter tuning is facilitated using the Hydra Optuna plugin. This allows for flexible and efficient exploration of the hyperparameter space to optimize your hyperparameters. Below, we provide an example configuration file and a step-by-step guide on setting up and running hyperparameter sweeps.

#### Example Configuration File

```yaml
# Example of a hyperparameter sweep configuration file

defaults:
  - logger: base_logger
  - arch: anakin
  - system: ff_ppo
  - network: mlp
  - env: gymnax/cartpole
  - override hydra/sweeper: optuna
  - override hydra/sweeper/sampler: tpe
  - _self_

hydra:
  mode: MULTIRUN
  sweeper:
    direction: maximize
    study_name: ${system.system_name}_${env.scenario.task_name}_sweep
    storage: null
    n_trials: 5
    n_jobs: 1
    sampler:
      seed: ${arch.seed}
    params:
      system.clip_eps: range(0.1, 0.3, step=0.1)
      system.gae_lambda: range(0, 1, step=0.05)
      system.epochs: range(1, 10, step=1)
```

#### Configuration Details

- **defaults**: Specifies the default configurations for logger, architecture, system, network, and environment.
- **hydra**: Configures Hydra for multi-run mode.
  - **mode**: Set to `MULTIRUN` for hyperparameter sweeps.
  - **sweeper**: Configures the Optuna sweeper.
    - **direction**: Optimization direction (`maximize` or `minimize`).
    - **study_name**: Name of the study for tracking.
    - **storage**: Database for storing study results (`null` for in-memory storage).
    - **n_trials**: Number of trials to run.
    - **n_jobs**: Number of parallel jobs.
    - **sampler**: Configures the sampler, here using TPE with a specified seed.
    - **params**: Defines the hyperparameter ranges to explore.

#### Running the Hyperparameter Sweep

To run the hyperparameter sweep, use the following command in your terminal and set the config to be the hyperparameter sweep config file:

```bash
python stoix/systems/ppo/ff_ppo.py --config-name=hyperparameter_sweep
```

Replace `ppo` with any of the algorithms. This command will execute multiple runs with different hyperparameter combinations as specified in the configuration file.

#### Tips for Effective Hyperparameter Tuning

1. **Start with a Small Number of Trials**: Begin with a smaller `n_trials` to get a quick sense of the parameter space and to validate all is working okay.
2. **Use Meaningful Ranges**: Ensure the hyperparameter ranges are wide enough to explore diverse configurations but narrow enough to be practical - extremely large multi-dimensional spaces can be hard to explore especially when training runs can take long.
3. **Analyze Results**: After the sweep, analyze the results to identify the best-performing configurations and understand the impact of different hyperparameters.

#### References and Further Reading

For more detailed instructions and advanced configurations, refer to the [Hydra documentation](https://hydra.cc/docs/intro/) and the [Optuna documentation](https://optuna.readthedocs.io/en/stable/).
