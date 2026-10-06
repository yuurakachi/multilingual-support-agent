"""Smoke test: confirms the package is importable and pytest is wired up."""

import support_agent


def test_package_is_importable():
    assert support_agent.__version__
