import os
from typing import List

import setuptools
from setuptools import setup


def _parse_requirements(path: str) -> List[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(path)) as f:
        return [line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))]


def _get_version() -> str:
    """Grabs the package version from stoix/version.py."""
    dict_: dict = {}
    with open("stoix/version.py") as f:
        exec(f.read(), dict_)
    return dict_["__version__"]


setup(
    name="stoix",
    version=_get_version(),
    author="EdanToledo",
    description="A Python library for Multi-Agent Reinforcement Learning in JAX.",
    license="Apache 2.0",
    url="https://github.com/EdanToledo/stoix/",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="single-agent reinforcement-learning python jax",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=_parse_requirements("requirements/requirements.txt"),
    extras_require={
        "dev": _parse_requirements("requirements/requirements-dev.txt"),
    },
    package_data={"stoix": ["py.typed"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
    zip_safe=False,
    include_package_data=True,
)
