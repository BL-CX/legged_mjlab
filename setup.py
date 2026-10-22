"""Minimal MJLab dependency bootstrap for Legged MJLab."""

from setuptools import setup


setup(
    name="legged_mjlab",
    version="0.1.0",
    description="Legged robot learning with MJLab",
    python_requires=">=3.10,<3.14",
    # Task packages and assets will be added in subsequent commits.
    packages=[],
    include_package_data=False,
    install_requires=[
        "mjlab==1.2.0",
    ],
)
