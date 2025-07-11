# Kinetix
<p align="middle">
  <img src="https://raw.githubusercontent.com/FlairOX/Kinetix/main/images/kinetix_logo.gif" width="500" />
</p>

See the [main Kinetix repo](https://github.com/FLAIROx/Kinetix) for an introduction to this environment.

## Basic Usage

To run PPO with Kinetix, you can use:

```bash
python stoix/systems/ppo/anakin/ff_ppo.py env=kinetix/small arch.num_eval_episodes=720 network=specialised/kinetix_entity
```

This by default runs using the `small` environment size, trains and tests on the [list of small levels](../../stoix/configs/env/kinetix/eval/s.yaml)

To train & eval on medium or large, respectively, replace `env=kinetix/small` in the above command with `env=kinetix/medium` or `env=kinetix/large`.


> [!WARNING]
> `arch.num_eval_episodes` must always be set to a multiple of the number of evaluation levels, which is `10` for small, `24` for medium and `40` for large.

## Customising Behaviour
### Different Observation / Action Spaces
Kinetix has several possible observation and action spaces. By default, the above command uses the `symbolic entity` observation space and `multidiscrete` action space, so we must use the custom `kinetix_entity` network. This consists of a special encoder for the set observation representation, and also a special MultiDiscrete action head.

If you want to use the standard flat symbolic representation, you can do so by setting `env.scenario.observation_type=symbolic_flat`, or `env.scenario.observation_type=symbolic_pixels` for pixels.
Similarly, you can use the continuous action space by setting `env.scenario.action_type=continuous`.

> [!WARNING]
> When changing the action type or the observation type, your network may need to change as well. See `stoix/configs/network/specialised/kinetix_{pixels,flat,entity}.yaml` for example configs for each of the observation spaces and the multi-discrete action space. Similarly, if you use the continuous action space, please run a compatible system, e.g. `ff_ppo_continuous.py`


### Different Train / Eval Environments

You can separately set the train and evaluation environments, for instance setting `env/kinetix/train=s env/kinetix/eval=m`. Note, this works only when the observation space is the same size across different env sizes, i.e., for Pixels or Entity.

Finally, you can set `env/kinetix/env_size=XXX`. This only is used when one of eval or train is set to `random`, in which case the environments will be of the corresponding size.
