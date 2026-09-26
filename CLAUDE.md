# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Q-NAS (Quantum-inspired Neural Architecture Search) in PyTorch: a research codebase (PhD thesis) that evolves CNN architectures for CIFAR-10 / MedMNIST / other datasets. "Quantum-inspired" means a probability table per node/op that is sampled and nudged toward good individuals; it runs on plain NVIDIA GPUs (CUDA required). `docs/QNAS_ARCHITECTURE.md` is a thorough walkthrough (pipeline, generation loop, cache, on-disk artifacts, progressive growth appendix) — read it before non-trivial changes to the search.

Tests (`tests/`, pytest) and lint (ruff) are configured in `pyproject.toml`. Tests run on CPU with a fake evaluator, so they need no GPU or dataset; for a real end-to-end check run a small config (e.g. `configs/config_files_cifar/00_smoketest.yml`).

**[EXPERIMENTS.md](EXPERIMENTS.md) must always be kept up to date.** It is the index of every CIFAR-10 progressive-growth config in `configs/config_files_cifar/`, in the order the ideas were tried: what each one changes versus its predecessor, what was expected, and the actual result. Whenever you add, rename, or edit a config in that directory, or learn the result of a run (an accuracy, a failure mode, a new exp name), update `EXPERIMENTS.md` in the same change — do not leave it to a follow-up.

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
make pipeline CONFIG=configs/config_files_cifar/01_deterministic_13-8-4.yml \
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
make queue-add CONFIG=<yml> EXP=<exp_path> EXTRA="-d cifar10 -M -T -X" GPU_IDS=0   # MODE=pipeline|evolve|retrain
make queue-start WORKERS=2 | queue-stop | queue-list | queue-status
make queue-logs                # follow current job; queue-logs-summary follows log_QNAS.txt
make queue-logs-running        # follow every running job at once (WORKERS > 1), prefixed by job id
make queue-watch                # live dashboard: generation/best fitness per job updating in place, detail below
make queue-cancel|queue-remove|queue-retry ID=<n>
```

`queue-add` freezes a copy of the config under `.qnas_queue/configs/` and the job runs from that copy, so editing a YAML after queueing does not change queued jobs. The queue runner (`src/qnas_queue/runner.py`) builds `uv run python src/...` / `run_pipeline.sh` argv from the job's mode + `EXTRA`. `scripts/path_config.sh` defines `PROJECT_DIR`/`SRC_DIR`/config dirs for all shell scripts.

**Concurrency and GPU pinning**: `queue-start WORKERS=N` (or `qnas-queue start --workers N`, or `$QNAS_QUEUE_WORKERS`) runs up to N worker processes, each claiming and running at most one queued job at a time (default 1, i.e. the old sequential behavior) - `src/qnas_queue/db.py`'s `workers` table tracks one row per live worker pid, and `claim_next_job` is atomic across them. `queue-add GPU_IDS=0` (or `--gpu-ids "0,1"`) pins that job's subprocess to specific GPU(s) via `CUDA_VISIBLE_DEVICES` (`worker.py::_run_job`) - every CUDA index the job's own code then sees (`evaluation.py`'s per-individual round robin, `cnn/train.py`'s `train.device`) is relative to that narrowed set. Without `GPU_IDS`, a job sees every GPU, as before this option existed - so when running several jobs concurrently (`WORKERS` > 1), either give each a disjoint `GPU_IDS`, or let the workers pick for themselves: `queue-start WORKERS=N GPU_POOL="0,1,2,3"` (or `--gpu-pool`/`$QNAS_QUEUE_GPU_POOL`) has each worker claim a GPU that's currently free in the pool for whatever job it picks up next (`db.claim_next_job`'s `gpu_pool` argument - "free" means not in `gpu_ids` of any other `running` job right now; nothing is reserved ahead of time). A job queued with its own `GPU_IDS` is never touched by the pool either way. If the next queued job needs the pool and it's fully occupied, `claim_next_job` tries the job behind it instead of stalling - only jobs that need the (exhausted) pool are skipped, everything else claims in the usual priority/id order. `queue-logs` without an ID still shows one job at a time (the most recently started running one); with several jobs running at once, use `queue-logs-running` to follow all of them together, each line prefixed `[job <id>]`. `queue-watch` goes further: a live table (per job: generation/max, best fitness, which generation/individual produced it, that fitness's delta vs. the job's FIRST logged generation (not the previous one - most generations don't produce a new best individual, so a gen-over-gen delta reads 0.00000 almost every time), the current generation's own fitness spread, elapsed time, and an ETA extrapolated from that job's own progress so far - all parsed from each job's `log_QNAS.txt`) that updates in place, with every job's detail log tailed below it, tagged by job id - needs a real terminal (exits with an error under a redirect/pipe; use `queue-logs-running` there instead). Each job gets a stable color (cycled by job id, `cli._job_style`) shared by its table row and its "[job N]" detail lines, and those "[job N]" prefixes are padded to a common width per render so lines from different jobs (e.g. job 2 and job 10) stay aligned instead of drifting based on how many digits that job's id has.

## Architecture

Entry points live in `src/` and are run as scripts (flat imports like `import qnas`, `import qnas_config as cfg` — `src/` is the import root, not a package).

- **Search loop** — `run_evolution.py` builds `qnas_config.ConfigParameters` (YAML + CLI args), an `evaluation.EvalPopulation`, and a `qnas.QNAS`, then calls `QNAS.evolve()`. Each generation: sample classical individuals from the quantum populations -> decode to `net_list` -> train in parallel worker processes -> `replace_pop` -> `update_quantum` -> save `data_QNAS.pkl`.
- **Two quantum populations** (`population.py` + `chromosome.py`): `QPopulationNetwork` (per-node categorical PMF over ops, a ragged list because nodes can have different op menus after pruning) and `QPopulationParams` (shrinking `[lower, upper]` ranges for hyperparameters). `repetition` classical individuals are sampled per quantum individual.
- **Quantum update engines** (`QNAS.quantum_update_engine` in config): `default` rotates quantum individual *i* toward the classical individual at fitness rank *i*; `ancestor_decay` rotates each quantum individual toward the classical individuals it actually produced (lineage via `classic_ancestor`), weighted by `exp(-quantum_update_age_decay * age)`. It only changes the update weight/target — survivor selection in `replace_pop` is identical, but `replace_pop`/`order_pop` must keep `classic_age`/`classic_ancestor` in lockstep with the population arrays (`extra=` argument).
- **Progressive growth** (P-DARTS-style, `QNAS.progressive` in config): grows network depth and prunes the op menu across stages (`_transition_stage`, `_rank_and_prune_*`, `_nucleus_prune_*`). `mode: deterministic` uses fixed stages by generation; `mode: dynamic` decides transitions from stability streaks (`_dynamic_prune_step`). Growth resizes the PMF (`grow_and_prune_discrete`) and the carried-over classical population; see docs Appendix A before touching it.
- **Evaluation** — `evaluation.py` dispatches one process per individual; `architecture_cache.py` caches fitness by `net_list` (`cache.json`) so identical architectures are not retrained. `cnn/train.py:fitness_calculation` trains one individual; `cnn/model.py` (`NetworkGraph` + op blocks) builds the network from `net_list`; `cnn/input.py` loads data; `cnn/metrics.py` computes params/FLOPs/latency.
- **Retrain / report** — `retrain_model.py` (uses `cnn/train_detailed.py`) retrains the best architecture found in the experiment dir (`ConfigParameters.load_evolved_data` reads it from the best `{gen}_{ind}/training_params.txt`); `generate_infographic.py` renders `infographic.png` from evolve + retrain outputs. `train_resnet.py` and `cnn/fine_tune_cnn.py` are standalone baselines/utilities outside the search loop.
- **Configs** — `configs/config_files_{cifar,atleta,med,medmnist}/*.yml`. A config holds the `QNAS:` block (population sizes, replace/crossover, progressive, quantum update), `params_ranges`, and the `function_dict` op menu (`ConvBlock`, MBConv, `no_op`, ...). `config_files_cifar/` is the current line of experiments and is numbered in the order the underlying ideas were tried/validated (`00_smoketest.yml`, then `01_...` through `08_...`); each file's header comment says what it changed vs. its predecessor and, where known, the experiment name/accuracy it produced — see [EXPERIMENTS.md](EXPERIMENTS.md) for the same information as a single indexed log; keep it in sync with this directory. `*_ancestor-decay.yml` variants only switch the quantum update engine vs. their non-decay counterpart.
- **Tests** (`tests/`): `conftest.py` provides `make_qnas` (an initialized `QNAS` wired to `FakeEval`, a stand-in for `EvalPopulation`), which is how `evolve()` is exercised end to end for the plain, deterministic-progressive and dynamic-progressive paths. `test_config.py` loads every YAML under `configs/`; the legacy ones are listed in `STALE_CONFIGS` (non-strict xfail) — a new config that fails to load is a real failure. When you find a defect you are not fixing right away, pin it as a `xfail(strict=True)` test so it turns red (XPASS) the moment it is fixed.
- **Experiments** are written to `experiment_*/expN/` (git-ignored via `exp*/`; run outputs: `data_QNAS.pkl`, `log_QNAS.txt`, per-individual `{gen}_{ind}/`, `retrain_*/`).
- **Reports** are written to `reports/`. Use it in order to register the information of experiments.
