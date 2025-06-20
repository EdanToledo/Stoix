# Stoix Performance Testing Framework

This directory contains a framework for running performance tests on reinforcement learning algorithms implemented in the Stoix library. The tests verify that algorithms perform as expected on standard benchmark environments and maintain performance relative to established baselines.

## Directory Structure

```
tests/performance_tests/
├── __init__.py                  # Package initialization
├── main.py                      # Entry point for running tests
├── framework/                   # Test framework components
│   ├── __init__.py
│   ├── registry.py              # Test registry for test discovery
│   ├── runner.py                # Test runner logic
│   └── utils.py                 # Testing utilities
├── algorithms/                  # Tests organized by algorithm
│   ├── __init__.py
│   ├── sac/                     # SAC algorithm tests
│   │   ├── __init__.py
│   │   └── test_sac.py          # SAC tests for different environments
│   └── ...                      # Other algorithms
├── data/                        # Data directory
│   ├── baselines/               # Performance baselines
│   └── reports/                 # Test reports
└── README.md                    # This file
```

## Usage

### Running Tests

You can run the tests using the `main.py` script:

```bash
# Run all tests
python -m tests.performance_tests.main

# Run tests for a specific algorithm
python -m tests.performance_tests.main --algorithm ff_sac

# Run tests for a specific environment
python -m tests.performance_tests.main --environment brax/ant

# Establish new baselines
python -m tests.performance_tests.main --establish-baseline

# List available tests
python -m tests.performance_tests.main --list

# Specify configuration overrides from a JSON file
python -m tests.performance_tests.main --config path/to/config.json

# Specify directory to save reports
python -m tests.performance_tests.main --report-dir path/to/reports

# Enable verbose logging
python -m tests.performance_tests.main --verbose
```

### Adding New Tests

To add a new test for an algorithm:

1. Create a new directory under `algorithms/` if it doesn't exist (e.g., `algorithms/ppo/`)
2. Create a test file with descriptive name (e.g., `test_ppo.py`)
3. Use the `register_test` decorator to register your test functions:

```python
from tests.performance_tests.framework.registry import register_test
from tests.performance_tests.framework.utils import test_algorithm_performance

@register_test(
    algorithm="ff_ppo",
    environment="brax/ant",
    module_path="stoix.systems.ppo.ff_ppo",
    arch="anakin"
)
def test_ppo_ant(establish_baseline=False, config_overrides=None):
    """
    Test PPO performance on the Brax Ant environment.

    Args:
        establish_baseline: If True, save results as new baseline.
        config_overrides: Dictionary of configuration overrides.

    Returns:
        TestResult object with performance metrics and comparison to baseline.
    """
    # Implement your test here
    all_overrides = {
        # Environment-specific configuration overrides
    }

    # Apply user-provided overrides
    if config_overrides:
        all_overrides.update(config_overrides)

    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="brax/ant",
        module_path="stoix.systems.ppo.anakin.ff_ppo",
        arch="anakin",
        establish_baseline=establish_baseline,
        config_overrides=all_overrides
    )
```

### Required Test Registration Parameters

When registering a test, you must provide these parameters:

- `algorithm`: Name of the algorithm being tested (e.g., "ff_sac")
- `environment`: Name of the environment being tested (e.g., "brax/ant")
- `module_path`: Path to the module containing the algorithm implementation (e.g., "stoix.systems.sac.ff_sac")
- `arch`: Architecture identifier used for configuration (e.g., "anakin")

## Baselines

Baseline performance metrics are stored in JSON files in the `data/baselines/` directory. These are used to compare against when running tests. To establish a new baseline, use the `--establish-baseline` flag.

## Reports

Test reports are saved as Markdown files in the `data/reports/` directory. They include summary statistics, detailed metrics, and comparisons to baselines.
