# JaxARC

[JaxARC](https://github.com/aadimator/JaxARC) is a hardware-accelerated ARC (Abstraction and Reasoning Corpus) environment built on JAX. It enables massively parallel RL training on ARC puzzle tasks with full JIT compilation support.

## Basic Usage

To run PPO with JaxARC on the MiniARC dataset:

```bash
python stoix/systems/ppo/anakin/ff_ppo.py env=jaxarc/default
```

This uses the `Mini` dataset (70 tasks) with point-based actions — a good starting point for prototyping.

To train on the full ARC-AGI-1 training set (400 tasks):

```bash
python stoix/systems/ppo/anakin/ff_ppo.py env=jaxarc/agi1_train
```

## Available Configs

| Config | Dataset | Tasks | Action Mode | Description |
|--------|---------|-------|-------------|-------------|
| `jaxarc/default` | MiniARC | 70 | point | Small dataset for fast prototyping |
| `jaxarc/agi1_train` | ARC-AGI-1 | 400 | point | Official ARC training benchmark |
| `jaxarc/agi1_bbox` | ARC-AGI-1 | 400 | bbox | Bounding-box actions for structured edits |
| `jaxarc/concept` | ConceptARC | ~40 | point | Tasks grouped by reasoning concept |

## Customising Behaviour

### Datasets

JaxARC supports four ARC datasets. Set the dataset via the `env.scenario.name` config:

- **Mini** (`env.scenario.name=Mini`): 70 simplified tasks from MiniARC. Good for debugging and prototyping.
- **AGI1** (`env.scenario.name=AGI1-train`): 400 official ARC-AGI-1 training tasks. The standard RL benchmark. Use `AGI1-eval` for the evaluation split.
- **AGI2** (`env.scenario.name=AGI2-train`): ARC-AGI-2 competition tasks. Use `AGI2-eval` for the evaluation split.
- **ConceptARC** (`env.scenario.name=Concept-SameDifferent`): Tasks grouped by abstract reasoning concepts. Replace `SameDifferent` with any concept name.

Available ConceptARC concepts: `AboveBelow`, `Center`, `CleanUp`, `CompleteShape`, `Copy`, `Count`, `ExtendToBoundary`, `ExtractObjects`, `FilledNotFilled`, `HorizontalVertical`, `InsideOutside`, `MoveToBoundary`, `Order`, `SameDifferent`, `TopBottom2D`, `TopBottom3D`.

### Action Modes

JaxARC offers three action modes. Override via `env.action.mode`:

```bash
# Point actions (default) — select individual cells
python stoix/systems/ppo/anakin/ff_ppo.py env=jaxarc/default env.action.mode=point

# Bounding-box actions — select rectangular regions
python stoix/systems/ppo/anakin/ff_ppo.py env=jaxarc/default env.action.mode=bbox
```

- **point**: Agent selects one cell at a time (row, col, color). Simple but large action space.
- **bbox**: Agent selects rectangular regions (top-left, bottom-right, color). Enables structured edits.
- **mask**: Full grid mask specification. Most expressive but largest action space.

### Observation Wrappers

Three observation components can be toggled independently:

```yaml
observation_wrappers:
  answer_grid: true   # Include the current answer grid state
  input_grid: true    # Include the input grid (task specification)
  contextual: true    # Include task context (episode step, etc.)
```

## Key Metrics

JaxARC provides domain-specific metrics alongside the standard Stoix `episode_return` and `episode_length`:

| Metric | Description |
|--------|-------------|
| `best_similarity` | Peak grid similarity achieved during the episode (0–1) |
| `solved` | Whether the agent produced a perfect solution |
| `steps_to_solve` | Number of steps taken to first solve (if solved) |
| `final_similarity` | Grid similarity at episode termination (0–1) |
| `was_truncated` | Whether the episode hit the step limit without solving |

These are emitted via JaxARC's `ExtendedMetrics` wrapper and appear in `timestep.extras["episode_metrics"]`.
