"""Setup the tests for the benchmark.

In particular, for each test name TEST_NAME, defining check_TEST_NAME will
allow to skip particular configuration combinations that cannot be run
for some reason (e.g. missing compiled functions, or too long to run in CI).
"""

import pytest


def check_test_dataset_get_data(benchmark, dataset_class):
    if dataset_class.name.lower() == "landau2x2v":
        pytest.skip("Landau2X2V need compiled functions.")
