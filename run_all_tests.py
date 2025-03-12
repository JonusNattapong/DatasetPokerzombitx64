"""
Run all tests for the pokerdata package.
"""

import unittest
import sys
import os


def run_tests():
    """Run all tests in the tests directory."""
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Add the project root to the path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
