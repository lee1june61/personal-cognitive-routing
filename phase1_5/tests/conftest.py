"""pytest config for phase1_5 — verbatim from phase1.

`--run-slow` flag enables tests that require real encoder load (HF download,
GPU). Default skip; CI / Colab pass `--run-slow`.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: integration tests requiring real encoder load (run in Colab)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow", default=False):
        skip = pytest.mark.skip(reason="use --run-slow to run on Colab")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip)


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False)
