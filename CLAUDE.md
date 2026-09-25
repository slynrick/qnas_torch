# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Q-NAS (Quantum-inspired Neural Architecture Search) in PyTorch: a research codebase (PhD thesis) that evolves CNN architectures for CIFAR-10 / MedMNIST / other datasets. "Quantum-inspired" means a probability table per node/op that is sampled and nudged toward good individuals; it runs on plain NVIDIA GPUs (CUDA required). `docs/QNAS_ARCHITECTURE.md` is a thorough walkthrough (pipeline, generation loop, cache, on-disk artifacts, progressive growth appendix) — read it before non-trivial changes to the search.

Tests (`tests/`, pytest) and lint (ruff) are configured in `pyproject.toml`. Tests run on CPU with a fake evaluator, so they need no GPU or dataset; for a real end-to-end check run a small config (e.g. `configs/config_files_cifar/config_progressive_smoketest.yml`).

## Commands

Dependencies are managed with uv (Python 3.11, torch 2.7 cu128 wheels). Always run Python through `uv run`.

```bash
make sync                      # uv sync (includes the dev group: pytest, pytest-cov, ruff)
make help                      # list all make targets

make check                     # ruff + full test suite
make test ARGS="-k ancestor -x"                      # subset / stop on first failure
make test ARGS="tests/test_population.py::TestAncestorDecayUpdate"   # single class or test
make test ARGS="-m 'not e2e'"                        # skip the full-evolve() runs
make test-cov                  # coverage report for src/
make lint                      # ruff check (correctness rules only); make lint-fix applies safe fixes
```

Any test run imports torch, which is slow from `/mnt/d` under WSL (~40s before the first test); prefer one broad run over many single-test invocations. There is no formatter: the existing code is not uniformly formatted, so do not run `ruff format` over it.

```bash

# Full pipeline (evolve -> retrain -> infographic), blocking
make pipeline CONFIG=configs/config_files_cifar/config_progressive.yml \
              EXP=experiment_cifar10_progressive/exp8 ARGS="-M -T -X"
scripts/run_pipeline.sh -e <exp_path> -c <config.yml> -d cifar10   # same thing; see header for all flags

# Individual phases
uv run python src/run_evolution.py --experiment_path <exp> --config_file <cfg> --data_path cifar10_data --dataset cifar10 --network_config default
uv run python src/retrain_model.py --experiment_path <exp> --data_path cifar10_data --dataset cifar10 --network_config default --config_code run1
uv run python src/generate_infographic.py --experiment_path <exp>
```

```bash
make diff-runs A=<exp dir|log_params|yml> B=<...> [EVOLVE_ARGS="--en_pop_crossover ..."]   # effective-config diff
```

Use `diff-runs` before labeling any two runs an A/B: run names and report descriptions have diverged from what actually ran (e.g. `exp4_ancestor_decay` is exp6's config + ancestor_decay). It compares recorded `log_params_evolution.txt` files and can render a `.yml` the way an evolve run would record it — some QNAS settings come from CLI flags (`-X` = `--en_pop_crossover`, `-m` = `--fitness_metric`), so pass them in `EVOLVE_ARGS`.

`run_pipeline.sh` flags: `-S evolve,retrain,infographic` skips steps, `-C <path>` continues a run, `-X` enables population crossover, `-M` AMP during retrain, `-T` retrain early stopping. `run_evolution.py` auto-resumes if `experiment_path` already holds a checkpoint (`log_params_evolution.txt`).

### Experiment queue

Long runs go through a sqlite-backed queue (`src/qnas_queue`, state in `.qnas_queue/`), driven by `scripts/qnas-queue.sh` or `make queue <sub>`:

```bash
make queue-add CONFIG=<yml> EXP=<exp_path> EXTRA="-d cifar10 -M -T -X"   # MODE=pipeline|evolve|retrain
make queue-start | queue-stop | queue-list | queue-status
make queue-logs                # follow current job; queue-logs-summary follows log_QNAS.txt
make queue-cancel|queue-remove|queue-retry ID=<n>
```

`queue-add` freezes a copy of the config under `.qnas_queue/configs/` and the job runs from that copy, so editing a YAML after queueing does not change queued jobs. The queue runner (`src/qnas_queue/runner.py`) builds `uv run python src/...` / `run_pipeline.sh` argv from the job's mode + `EXTRA`. `scripts/path_config.sh` defines `PROJECT_DIR`/`SRC_DIR`/config dirs for all shell scripts.

## Architecture

Entry points live in `src/` and are run as scripts (flat imports like `import qnas`, `import qnas_config as cfg` — `src/` is the import root, not a package).

- **Search loop** — `run_evolution.py` builds `qnas_config.ConfigParameters` (YAML + CLI args), an `evaluation.EvalPopulation`, and a `qnas.QNAS`, then calls `QNAS.evolve()`. Each generation: sample classical individuals from the quantum populations -> decode to `net_list` -> train in parallel worker processes -> `replace_pop` -> `update_quantum` -> save `data_QNAS.pkl`.
- **Two quantum populations** (`population.py` + `chromosome.py`): `QPopulationNetwork` (per-node categorical PMF over ops, a ragged list because nodes can have different op menus after pruning) and `QPopulationParams` (shrinking `[lower, upper]` ranges for hyperparameters). `repetition` classical individuals are sampled per quantum individual.
- **Quantum update engines** (`QNAS.quantum_update_engine` in config): `default` rotates quantum individual *i* toward the classical individual at fitness rank *i*; `ancestor_decay` rotates each quantum individual toward the classical individuals it actually produced (lineage via `classic_ancestor`), weighted by `exp(-quantum_update_age_decay * age)`. It only changes the update weight/target — survivor selection in `replace_pop` is identical, but `replace_pop`/`order_pop` must keep `classic_age`/`classic_ancestor` in lockstep with the population arrays (`extra=` argument).
- **Progressive growth** (P-DARTS-style, `QNAS.progressive` in config): grows network depth and prunes the op menu across stages (`_transition_stage`, `_rank_and_prune_*`, `_nucleus_prune_*`). `mode: deterministic` uses fixed stages by generation; `mode: dynamic` decides transitions from stability streaks (`_dynamic_prune_step`). Growth resizes the PMF (`grow_and_prune_discrete`) and the carried-over classical population; see docs Appendix A before touching it.
- **Evaluation** — `evaluation.py` dispatches one process per individual; `architecture_cache.py` caches fitness by `net_list` (`cache.json`) so identical architectures are not retrained. `cnn/train.py:fitness_calculation` trains one individual; `cnn/model.py` (`NetworkGraph` + op blocks) builds the network from `net_list`; `cnn/input.py` loads data; `cnn/metrics.py` computes params/FLOPs/latency.
- **Retrain / report** — `retrain_model.py` (uses `cnn/train_detailed.py`) retrains the best architecture found in the experiment dir (`ConfigParameters.load_evolved_data` reads it from the best `{gen}_{ind}/training_params.txt`); `generate_infographic.py` renders `infographic.png` from evolve + retrain outputs. `train_resnet.py` and `cnn/fine_tune_cnn.py` are standalone baselines/utilities outside the search loop.
- **Configs** — `configs/config_files_{cifar,atleta,med,medmnist}/*.yml`. A config holds the `QNAS:` block (population sizes, replace/crossover, progressive, quantum update), `params_ranges`, and the `function_dict` op menu (`ConvBlock`, MBConv, `no_op`, ...). The `config_progressive*` files are the current line of experiments; `*_ancestor_decay.yml` variants only switch the quantum update engine.
- **Tests** (`tests/`): `conftest.py` provides `make_qnas` (an initialized `QNAS` wired to `FakeEval`, a stand-in for `EvalPopulation`), which is how `evolve()` is exercised end to end for the plain, deterministic-progressive and dynamic-progressive paths. `test_config.py` loads every YAML under `configs/`; the legacy ones are listed in `STALE_CONFIGS` (non-strict xfail) — a new config that fails to load is a real failure. When you find a defect you are not fixing right away, pin it as a `xfail(strict=True)` test so it turns red (XPASS) the moment it is fixed.
- **Experiments** are written to `experiment_*/expN/` (git-ignored via `exp*/`; run outputs: `data_QNAS.pkl`, `log_QNAS.txt`, per-individual `{gen}_{ind}/`, `retrain_*/`).
- **Reports** are written to `reports/`. Use it in order to register the information of experiments.
