#!/usr/bin/env python3
"""
Test Registry Module

This module provides a central registry for performance tests,
allowing the test decorator to be imported by test modules and
the registry to be accessed by the test runner.
"""
import logging
from typing import Dict, Callable, Tuple

logger = logging.getLogger(__name__)

# Global registry for test functions
# Maps (algorithm, environment) tuples to test functions
TEST_REGISTRY: Dict[Tuple[str, str, str, str], Callable] = {}

def register_test(algorithm: str, environment: str, module_path: str, arch: str):
    """
    Decorator to register a test function in the registry.
    
    This allows test functions to be automatically discovered and run by the test runner.
    Each test is keyed by its algorithm and environment combination.
    
    Args:
        algorithm: Name of the algorithm being tested (e.g., "ff_sac")
        environment: Name of the environment being tested (e.g., "brax/ant")
        module_path: Path to the file containing the algorithm implementation (e.g., "stoix.systems.sac.ff_sac")
        arch: Name of the architecture being used (e.g., "anakin")
    Returns:
        Decorator function that registers the test function
    """
    
    def decorator(func):
        key = (algorithm, environment, module_path, arch)
        logger.debug(f"Registering test for {algorithm} on {environment} from {module_path} with arch {arch}")
        
        # Store the function in the registry
        TEST_REGISTRY[key] = func
        
        # Return the original function unchanged
        return func
    
    return decorator

def get_registry() -> Dict[Tuple[str, str, str, str], Callable]:
    """
    Get the current test registry.
    
    Returns:
        Dictionary mapping (algorithm, environment, module_path, arch) tuples to test functions
    """
    logger.debug(f"get_registry: TEST_REGISTRY has {len(TEST_REGISTRY)} entries")
    return TEST_REGISTRY 