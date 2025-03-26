# Stoix Testing Framework

This directory contains the testing framework for the Stoix reinforcement learning library. The framework is designed to ensure the reliability, correctness, and performance of the library.

## Test Structure

The tests are organized into two main categories:

1. **Unit Tests**: Located in `unit_tests/`, these test individual components of the library in isolation.
2. **Performance Tests**: Located in `performance_tests/`, these test the performance of algorithms on standard environments using our standardized testing framework.

## Running Tests

### Unit Tests

To run all unit tests:

```bash
pytest stoix/tests/unit_tests -v
```

To run a specific test file:

```bash
pytest stoix/tests/unit_tests/test_file.py -v
```

### Performance Tests

The performance testing system uses a standardized framework that integrates with our Hydra configuration system. This allows tests to use the same configurations as normal training runs with specific overrides for testing purposes.

#### Running Performance Tests

To run all performance tests:

```bash
python -m stoix.tests.performance_tests.main
```

To run tests for a specific algorithm:

```bash
python -m stoix.tests.performance_tests.main --algorithm ff_sac
```

To run tests for a specific environment:

```bash
python -m stoix.tests.performance_tests.main --environment brax/ant
```

For faster testing during development, you can limit the number of training steps:

```bash
python -m stoix.tests.performance_tests.main --max-steps 10000
```

To list all available tests:

```bash
python -m stoix.tests.performance_tests.main --list
```

#### Running Tests in Parallel with SLURM

The testing framework supports parallel execution of tests using SLURM, which can significantly speed up test runs:

```bash
python -m stoix.tests.performance_tests.main --use-slurm
```

You can specify a custom SLURM configuration file:

```bash
python -m stoix.tests.performance_tests.main --use-slurm --slurm-config custom_slurm
```

The SLURM configuration file should be located in `stoix/configs/launcher/` and follow the same format as the main SLURM launcher configuration.

This is particularly useful when running multiple tests or tests with multiple seeds:

```bash
python -m stoix.tests.performance_tests.main --use-slurm --num-seeds 5
```

#### Configuring Performance Tests

You can specify configuration overrides from a JSON file:

```bash
python -m stoix.tests.performance_tests.main --config path/to/config.json
```

#### Establishing Baselines

Before running performance tests, baselines need to be established. This should be done after significant changes to the library or when adding new algorithms/environments.

To establish baselines:

```bash
python -m stoix.tests.performance_tests.main --establish-baseline
```

Or for a specific algorithm/environment:

```bash
python -m stoix.tests.performance_tests.main --algorithm ff_sac --environment brax/ant --establish-baseline
```

#### Test Reports

Performance test reports are generated automatically and include:
- Overall success rate
- Performance comparison against baselines
- Detailed metrics for each test
- Performance regressions or improvements

To specify the directory for saving reports:

```bash
python -m stoix.tests.performance_tests.main --report-dir path/to/reports
```

By default, reports are saved to `stoix/tests/performance_tests/data/reports/`.

## Adding New Tests

### Adding Unit Tests

1. Create a new file in `unit_tests/` named `test_*.py`
2. Import the component to test
3. Write test functions prefixed with `test_`
4. Use assertions to verify expected behavior

### Adding Performance Tests

To add performance tests for a new algorithm:

1. Create a new directory under `performance_tests/algorithms/` if it doesn't exist (e.g., `algorithms/ppo/`)
2. Create a test file with a descriptive name (e.g., `test_ppo.py`)
3. Use the `register_test` decorator to register your test functions:

```python
from stoix.tests.performance_tests.framework.registry import register_test
from stoix.tests.performance_tests.framework.utils import test_algorithm_performance

@register_test(
    algorithm="ff_ppo",
    environment="brax/ant",
    module_path="stoix.systems.ppo.ff_ppo",
    arch="anakin"
)
def test_ppo_ant(establish_baseline=False, config_overrides=None):
    """Test PPO performance on the Brax Ant environment."""
    # Specify algorithm-specific overrides
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

## CI/CD Integration

The tests are automatically run on pull requests to the main branch. Performance tests will compare results against established baselines and fail if performance degrades beyond the acceptable threshold.

To manually trigger a baseline update, use the appropriate workflow in the GitHub Actions tab with the "establish_baseline" input set to true.

## Best Practices

1. **Keep Tests Fast**: Unit tests should run quickly
2. **Use Fixtures**: Reuse common setup code with pytest fixtures
3. **Test Edge Cases**: Include tests for boundary conditions and error handling
4. **Maintain Baselines**: Update baselines when algorithms are improved
5. **Use Fixed Seeds**: Use fixed random seeds for reproducibility
6. **Document Tests**: Include docstrings explaining what each test is verifying
7. **Test Across Environments**: Ensure each algorithm is tested on environments that showcase its strengths
