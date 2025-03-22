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

# Set maximum training steps
python -m tests.performance_tests.main --max-steps 1000000

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

@register_test(algorithm="ff_ppo", environment="brax/ant")
def test_ppo_ant(establish_baseline=False, max_steps=None, config_overrides=None):
    # Implement your test here
    return test_algorithm_performance(
        algorithm="ff_ppo",
        environment="brax/ant",
        establish_baseline=establish_baseline,
        max_steps=max_steps,
        config_overrides=config_overrides
    )
```

## Baselines

Baseline performance metrics are stored in JSON files in the `data/baselines/` directory. These are used to compare against when running tests. To establish a new baseline, use the `--establish-baseline` flag.

## Reports

Test reports are saved as Markdown files in the `data/reports/` directory. They include summary statistics, detailed metrics, and comparisons to baselines. 