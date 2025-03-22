# Stoix Testing Framework

This directory contains the testing framework for the Stoix reinforcement learning library. The framework is designed to ensure the reliability, correctness, and performance of the library.

## Test Structure

The tests are organized into three main categories:

1. **Unit Tests**: Located in `unit_tests/`, these test individual components of the library in isolation.
2. **Integration Tests**: Located in `integration_tests/`, these test how components work together.
3. **Performance Tests**: Located in `performance_tests/`, these test the performance of algorithms on standard environments.

## Running Tests

### Unit and Integration Tests

To run all unit and integration tests:

```bash
pytest tests/unit_tests tests/integration_tests -v
```

To run a specific test file:

```bash
pytest tests/unit_tests/test_evaluator.py -v
```

### Performance Tests

Performance tests are designed to compare algorithm performance against established baselines. These tests are more time-consuming and are typically run as part of the CI/CD pipeline.

### New Standardized Testing System

The performance tests now use a standardized system that leverages the existing Hydra configuration framework, allowing tests to:

1. Reuse the same configurations used in normal training
2. Override specific parameters for testing purposes
3. Track performance metrics against established baselines
4. Generate detailed reports

### Running Tests

To run all performance tests:

```bash
python tests/performance_tests/run_performance_tests.py
```

To run tests for a specific algorithm and environment:

```bash
python tests/performance_tests/run_performance_tests.py --algorithm ff_sac --environment brax/ant
```

For faster testing during development, you can limit the number of training steps:

```bash
python tests/performance_tests/run_performance_tests.py --max-steps 10000
```

### Overriding Configurations

A key feature of the testing system is the ability to override existing Hydra configurations without recreating them. You can provide configuration overrides using the `--config-override` argument:

```bash
python tests/performance_tests/run_performance_tests.py --algorithm ff_sac --environment brax/ant --config-override "system.learning_rate=1e-4" --config-override "system.buffer_size=500000"
```

### Establishing Baselines

Before running performance tests, baselines need to be established. This should be done after significant changes to the library or when adding new algorithms/environments.

To establish baselines:

```bash
python tests/performance_tests/run_performance_tests.py --establish-baseline
```

Or for a specific algorithm/environment:

```bash
python tests/performance_tests/run_performance_tests.py --establish-baseline --algorithm ff_sac --environment brax/ant
```

### Test Reports

After running tests, a detailed report is generated showing:
- Overall success rate
- Performance comparison against baselines
- Detailed metrics for each test
- Performance regressions or improvements

Reports are saved to `tests/performance_tests/reports/` by default.

## Adding New Tests

### Adding Unit Tests

1. Create a new file in `unit_tests/` named `test_*.py`
2. Import the component to test
3. Write test functions prefixed with `test_`
4. Use assertions to verify expected behavior

### Adding Integration Tests

1. Create a new file in `integration_tests/` named `test_*.py`
2. Import the components to test
3. Write test functions that verify how components interact
4. Use assertions to verify expected behavior

### Adding Performance Tests for New Algorithms

To add performance tests for a new algorithm, follow these steps:

1. **Create a Test File**: Create `performance_tests/test_<algo>_performance.py`:
   ```python
   # Example for a new algorithm "xyz"
   from tests.performance_tests.run_performance_tests import register_test
   from tests.performance_tests.test_utils import test_algorithm_performance
   
   @register_test(algorithm="ff_xyz", environment="brax/ant")
   def test_xyz_ant(rng_key, establish_baseline=False, max_steps=None, config_overrides=None):
       # You can specify algorithm-specific overrides here
       all_overrides = {
           "system.learning_rate": 3e-4,
           # Other overrides...
       }
       
       if config_overrides:
           all_overrides.update(config_overrides)
       
       return test_algorithm_performance(
           algorithm="ff_xyz",
           environment="brax/ant",
           rng_key=rng_key,
           establish_baseline=establish_baseline,
           max_steps=max_steps,
           config_overrides=all_overrides
       )
   ```

2. **Define Standard Test Parameters**:
   - Include any standard overrides that should be applied for testing
   - Add documentation explaining the specific test setup
   - Make sure parameter names align with what's used in your algorithm's config

That's it! The test will be automatically registered and available in the performance test runner.

## CI/CD Integration

The tests are automatically run on pull requests to the main branch. Performance tests will compare results against established baselines and fail if performance degrades beyond the acceptable threshold.

To manually trigger a baseline update, use the "Performance Tests" workflow in the GitHub Actions tab with the "establish_baseline" input set to true.

The GitHub Actions workflow supports:
- Running all tests or filtering by algorithm/environment
- Setting a maximum number of steps for faster testing
- Establishing new baselines
- Generating reports comparing current performance to baselines

## Best Practices

1. **Keep Tests Fast**: Unit and integration tests should run quickly
2. **Use Fixtures**: Reuse common setup code with pytest fixtures
3. **Test Edge Cases**: Include tests for boundary conditions and error handling
4. **Maintain Baselines**: Update baselines when algorithms are improved
5. **Use Fixed Seeds**: Use fixed random seeds for reproducibility
6. **Document Tests**: Include docstrings explaining what each test is verifying
7. **Time-Limited Tests**: Use the `--max-steps` flag during development for faster iterations
8. **Test Across Environments**: Ensure each algorithm is tested on environments that showcase its strengths 