"""Installation script for the 'lerobot_so101_teleop' python package."""

import os

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# Installation operation
setup(
    name="lerobot_so101_teleop",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    install_requires=INSTALL_REQUIRES,
    author="Lior Ben Horin",
    maintainer="Lior Ben Horin",
    url="https://github.com/liorbenhorin/lerobot_so101_teleop",
    zip_safe=False,
)
