# Merlin Systems

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/merlin-systems)
[![PyPI version shields.io](https://img.shields.io/pypi/v/merlin-systems.svg)](https://pypi.python.org/pypi/merlin-systems/)
[![Stability Alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://img.shields.io/badge/stability-alpha-f4d03f.svg)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/NVIDIA-Merlin/systems/CPU%20CI)
![GitHub License](https://img.shields.io/github/license/NVIDIA-Merlin/systems)

Merlin Systems provides tools for combining recommendation models with other elements of production recommender systems like feature stores, nearest neighbor search, and exploration strategies into end-to-end recommendation pipelines that can be served with [Triton Inference Server](https://github.com/triton-inference-server/server).

## Installation

Merlin Systems requires Triton Inference Server and Tensorflow. The simplest setup is to use the [Merlin Tensorflow Inference Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference), which has both pre-installed.

### Installing Merlin Systems Using Pip

You can install Merlin Systems with `pip`:

```sh
pip install merlin-systems
```

## Feedback and Support

To report bugs or get help, please [open an issue](https://github.com/NVIDIA-Merlin/NVTabular/issues/new/choose).
