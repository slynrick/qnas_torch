# Experiments

Log of the CIFAR-10 progressive-growth configs in
[`configs/config_files_cifar/`](configs/config_files_cifar/), in the order the
underlying ideas were tried. Each entry summarizes what changed versus the
previous step, what was expected from that change, and the actual result
where one is known. Config-level detail (exact keys, rationale for each
value) lives in the header comment of each YAML file; this doc is the
higher-level index across them.

Legacy configs superseded by this line of experiments live in
`configs/config_files_cifar/old/` and are not covered here.

## 00 — `00_smoketest.yml`

Not part of the numbered lineage below. A tiny/fast variant of
`01_deterministic_13-8-4.yml` (2 generations, 6-node cap, ~seconds to run),
used to smoke-test the evolve -> retrain -> infographic pipeline end to end
after code changes, before trusting a real run.

- **Expected:** completes without error; not evaluated for accuracy.
- **Result:** N/A (sanity check only).

## 01 — `01_deterministic_13-8-4.yml`

First entry in the deterministic-progressive-growth line. Discrete Q-NAS with
P-DARTS-style progressive growth: 3 stages pruning the op menu 13 -> 8 -> 4,
`reset_probs_on_stage_change: False` (carries learned quantum probabilities
over at each stage transition instead of resetting to uniform).

- **Expected:** progressively narrowing the op menu while keeping the
  learned probability mass across transitions should let good ops compound
  their advantage stage over stage.
- **Result:** ran as **exp6**. Superseded by `04_deterministic_13-10-8_reset.yml`
  (exp7), which scored higher.

## 02 — `02_deterministic_13-8-4_ancestor-decay.yml`

Same run as 01 (13 -> 8 -> 4, no probability reset), with the only change
being the quantum-population update engine: `quantum_update_engine:
ancestor_decay` instead of the default. `ancestor_decay` rotates each quantum
individual toward the classical individuals it actually produced (weighted
by lineage age), instead of rotating quantum individual *i* toward whichever
classical individual currently ranks *i*-th.

- **Expected:** update the quantum population from architectures it actually
  produced (its own descendants) rather than from whichever individual holds
  a given fitness rank, which may have come from an unrelated lineage.
- **Result:** on disk this run is recorded as
  `experiment_cifar10_progressive/exp4_ancestor_decay`, but that label is a
  misnomer — exp4/exp5 used `reset_probs_on_stage_change: True` (see 03
  below), not `False`. The run actually produced by this file is **exp6 +
  ancestor_decay**.

## 03 — `03_deterministic_13-8-4_reset_ancestor-decay.yml`

Same stage list as 01/02 (13 -> 8 -> 4) but with
`reset_probs_on_stage_change: True` (quantum probabilities reset to uniform
at each stage transition instead of carried over), plus the `ancestor_decay`
engine from 02.

- **Expected:** meant to isolate the effect of `ancestor_decay` on top of the
  exp4/exp5 setup — same run as exp4/exp5, differing only in the update
  engine.
- **Result:** ran as the ancestor_decay twin of **exp4/exp5**. The base
  (non-ancestor_decay) exp4/exp5 config that this pairs against predates the
  current numbering and no longer exists in this directory — this file is
  now the only surviving on-disk record of that setup's parameters.

## 04 — `04_deterministic_13-10-8_reset.yml`

Widens the surviving op menu at each stage compared to 01 — stages 13 -> 10
-> 8 instead of 13 -> 8 -> 4 — and switches
`reset_probs_on_stage_change` to `True` (reset to uniform at each stage
transition instead of carrying probabilities over).

- **Expected:** pruning less aggressively (10 and 8 ops survive, vs. 8 and 4)
  and re-exploring each new, smaller op menu from a clean slate (reset
  instead of carry-over) should avoid prematurely locking in ops that were
  only favored under the wider previous-stage menu.
- **Result:** ran as **exp7** — **77.20% best accuracy**, the best
  deterministic result in this line so far.

## 05 — `05_deterministic_13-10-8_reset_ancestor-decay.yml`

Same run as 04 (13 -> 10 -> 8, reset True), with only the quantum update
engine switched to `ancestor_decay`.

- **Expected:** same rationale as 02/03 — update quantum individuals from
  their own descendants' outcomes rather than by current fitness rank,
  applied to the best-performing deterministic setup (04/exp7) instead of
  exp6.
- **Result:** ancestor_decay counterpart of exp7; no accuracy recorded yet
  in this doc (see the experiment's own `log_params_evolution.txt` /
  `reports/` entry once run).

## 06 — `06_dynamic_v1.yml`

First attempt at **dynamic** progressive growth: instead of a pre-declared
stage list, op pruning and node growth are both decided during the search
(nucleus pruning by cumulative quantum probability mass, growth triggered
once pruning stabilizes). Same base op menu/hyperparameters as 01.

- **Expected:** let the search itself decide when to prune ops and grow
  depth, rather than committing to a fixed generation schedule up front.
- **Result:** ran as **exp_dynamic_1**. Degenerated: the `min_ops: 2` floor
  let nucleus pruning strip every convolution out of every node by
  generation 40, collapsing the op menu to `{avg_pool, max_pool, no_op}`
  with no learnable filters left, and never recovered. **Best accuracy
  plateaued at 72.7%** while the rest of the population's fitness kept
  sliding as depth grew on a conv-less search space. Superseded by
  `07_dynamic_v2.yml`.

## 07 — `07_dynamic_v2.yml`

Tunes the dynamic-mode parameters against the exp_dynamic_1 failure above:
`min_ops` 2 -> 5 (keeps a floor of conv ops alive), `probability_threshold`
0.8 -> 0.92 (less aggressive cuts), `check_every_gen` 10 -> 20 (more
generations for the PMF to converge between cuts), `flatness_epsilon` 0.02 ->
0.06 (wider tie zone, skips more premature cuts), and
`train.early_stopping_patience` 5 -> 9 (gives slower-converging conv ops more
epochs before per-individual early stopping favors faster pooling ops).
`global_op_pruning` left unchanged (`False`) — not part of this tuning pass.

- **Expected:** these five changes together should stop the op menu from
  collapsing to pooling-only, letting convolutions survive long enough to
  compete on fitness rather than being pruned out early on noisy signal.
- **Result:** ran as **exp_dynamic_2** — **75.60% best accuracy**. Fixed the
  collapse from v1, though still below the best deterministic result (exp7,
  77.20%).

## 08 — `08_dynamic_v2_ancestor-decay.yml`

Same run as 07 (dynamic v2 settings), with only the quantum update engine
switched to `ancestor_decay`.

- **Expected:** same ancestor_decay rationale as 02/03/05, applied to the
  fixed dynamic-mode setup instead of the deterministic one.
- **Result:** not yet run / no accuracy recorded in this doc.

## Summary table

| # | File | Mode | Stages / op floor | Reset probs | Update engine | Exp | Best acc. |
|---|------|------|--------------------|--------------|----------------|-----|-----------|
| 00 | `00_smoketest.yml` | deterministic | 4 -> 6 nodes (tiny) | False | default | — | N/A (smoke test) |
| 01 | `01_deterministic_13-8-4.yml` | deterministic | 13 -> 8 -> 4 | False | default | exp6 | — |
| 02 | `02_deterministic_13-8-4_ancestor-decay.yml` | deterministic | 13 -> 8 -> 4 | False | ancestor_decay | exp6 + decay (disk: exp4_ancestor_decay) | — |
| 03 | `03_deterministic_13-8-4_reset_ancestor-decay.yml` | deterministic | 13 -> 8 -> 4 | True | ancestor_decay | exp4/exp5 + decay | — |
| 04 | `04_deterministic_13-10-8_reset.yml` | deterministic | 13 -> 10 -> 8 | True | default | exp7 | **77.20%** |
| 05 | `05_deterministic_13-10-8_reset_ancestor-decay.yml` | deterministic | 13 -> 10 -> 8 | True | ancestor_decay | exp7 + decay | not run yet |
| 06 | `06_dynamic_v1.yml` | dynamic | nucleus, min_ops 2 | — | default | exp_dynamic_1 | 72.7% (degenerated) |
| 07 | `07_dynamic_v2.yml` | dynamic | nucleus, min_ops 5 | — | default | exp_dynamic_2 | **75.60%** |
| 08 | `08_dynamic_v2_ancestor-decay.yml` | dynamic | nucleus, min_ops 5 | — | ancestor_decay | — | not run yet |
