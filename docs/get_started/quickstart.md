# Quick start ⚡

To get started with training your first Stoix system, simply run one of the system files. e.g.,

```bash
python stoix/systems/ppo/ff_ppo.py
```

Stoix makes use of Hydra for config management. In order to see our default system configs please see the `stoix/configs/` directory. A benefit of Hydra is that configs can either be set in config yaml files or overwritten from the terminal on the fly. For an example of running a system on the CartPole environment, the above code can simply be adapted as follows:

```bash
python stoix/systems/ppo/ff_ppo.py env=gymnax/cartpole
```
