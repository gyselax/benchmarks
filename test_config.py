"""Setup the tests for the benchmark.

In particular, for each test name TEST_NAME, defining check_TEST_NAME will
allow to skip particular configuration combinations that cannot be run
for some reason (e.g. missing compiled functions, or too long to run in CI).
"""

import os
import shutil
import subprocess

import pytest


def check_test_dataset_get_data(benchmark, dataset_class):
    # Landau2X2V generates its data by running the compiled mini-app inside the
    # gysela-compression docker image. Run the test when that image is
    # available (the default launcher uses it); skip otherwise so CI without
    # docker still passes. Build it with:
    #   docker build -f benchmark_gysela/Dockerfile -t gysela-compression \
    #       gysela-mini-app_io
    if dataset_class.name.lower() == "landau2x2v":
        image = os.environ.get("GYSELA_IMAGE", "gysela-compression:latest")
        if shutil.which("docker") is None:
            pytest.skip("docker not available; skipping Landau2X2V.")
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if inspect.returncode != 0:
            pytest.skip(f"docker image '{image}' not built; skipping "
                        "Landau2X2V (see Dockerfile).")
