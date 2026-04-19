# QNAS - PyTorch Version

## Introduction

## Convolutional Neural Network


## Environment Configuration

The following steps are used to configure the environment for the project with Astral uv.

- uv installation
- Python and virtual environment setup
- Dependency synchronization

**Notes**: 
- An NVIDIA GPU is required to run the project. 
- The project was tested using three NVIDIA RTX A30 GPUs to run evolutionary search with up to 20 individuals in parallel.
- NVIDIA drivers and the CUDA Toolkit are necessary (tested with CUDA 11.6).
- The following steps have been tested on Ubuntu Linux.

### uv Installation

Install uv following the official guide: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

For Linux/macOS, one option is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python and Environment Setup

Create and pin a Python 3.11 environment for the project:

```bash
uv python install 3.11
uv venv --python 3.11
```

### Dependency Installation

Install all dependencies (including CUDA 12.1 PyTorch wheels) from `pyproject.toml`:

```bash
uv sync
```

### Running Scripts

Use `uv run` to execute project commands inside the uv-managed environment:

```bash
uv run python src/run_evolution.py --help
```

The helper scripts in this repository (`run.sh`, `run_retrain.sh`, `run_resnet.sh`) already use `uv run`.
