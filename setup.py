"""Fallback setup.py, prefer `pip install .` which uses pyproject.toml + scikit-build-core."""
from skbuild import setup

setup(
    name="wefml",
    version="0.1.0",
    packages=["wefml"],
    cmake_source_dir=".",
    cmake_install_dir="wefml",
)
