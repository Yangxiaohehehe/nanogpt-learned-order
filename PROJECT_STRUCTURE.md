# Project Structure Guide

This repository currently mixes:
- model/training code
- dataset/config files
- checkpoints
- one-off evaluation outputs
- W&B run artifacts

To avoid breaking existing scripts, the current files are left in place.
This document defines how to interpret the current layout and how new work
should be organized going forward.

## 1. What Each Top-Level Path Means

### Core code
- `AOGPT.py`: AO-GPT model with block-order support and prefix policy head.
- `AOGPT.py`: AO-GPT model with block-order support.
- `train.py`: main training loop for all stages.
- `order_utils.py`: block-order utilities, search helpers, ranking losses, metrics.
- `eval_prefix_pairs.py`: pair-prefix ranking evaluation.
- `signal_benchmark.py`: search-time benchmark for candidate order signals.
- `ar_likeness_benchmark.py`: compare AR-mode and Random-mode trajectory scores under a fixed checkpoint.

### Config
- `config/WikiText103/*.py`: stage-specific experiment configs.
- Files with `_b32.py`: 32-block setup (`block_order_block_len = 8`).
- Files without suffix: original 16-block setup (`block_order_block_len = 16`).

### Data
- `data/wikitext103/`: tokenized dataset binaries.

### Checkpoints
- `out-wikitext103-random/`: main 16-block Random checkpoint and related eval csv.
- `out-wikitext103-ar/`: main 16-block AR checkpoint.
- `out-wikitext103-random-b32/`: 32-block Random checkpoint.
- `out-wikitext103-random_256_test_prefix/`: ad hoc evaluation/export directory.

### Reports
- `Report/`: human-readable experiment outputs and summaries.

### Tracking / temporary
- `wandb/`: run artifacts.
- `__pycache__/`, `.pytest_cache/`: cache directories.

## 2. Current Pain Points

The structure feels messy mainly because three different concepts are stored side by side:

1. canonical training checkpoints
2. temporary evaluation exports
3. benchmark reports generated while exploring signals

This makes it hard to tell:
- which checkpoint is the "main" one
- which report is exploratory vs final
- which outputs should be reused by later stages

## 3. Recommended Mental Model

From now on, treat the repo as having four logical layers:

1. `code`
Meaning:
- model, training, benchmark scripts

2. `configs`
Meaning:
- reproducible experiment definitions

3. `checkpoints`
Meaning:
- stateful training outputs that later stages resume from

4. `reports`
Meaning:
- read-only evaluation/analysis outputs

The physical directory names are not fully normalized yet, but this is the intended interpretation.

## 4. Naming Rules Going Forward

### Config naming
Use:
- `<stage>.py` for the default 16-block setup
- `<stage>_b32.py` for the 32-block setup
- later, if needed:
  - `<stage>_b64.py`

### Checkpoint out_dir naming
Use:
- `out-<dataset>-<mode>`
- `out-<dataset>-<mode>-b32`
- avoid names like `copy`, `test`, or mixed experimental notes in canonical checkpoint directories

Recommended examples:
- `out-wikitext103-random`
- `out-wikitext103-ar`
- `out-wikitext103-random-b32`

### Report directory naming
Use:
- `Report/<topic>/<run_name>`
or
- `Report/<topic>_<variant>`

Examples:
- `Report/ar_likeness/random_only_b200`
- `Report/signal_benchmark/adjacent`
- `Report/signal_benchmark/insert_front`
- `Report/signal_benchmark/insert_anywhere`

For now, old report folders are preserved as-is.

## 5. What To Reuse vs Ignore

### Reuse
- `out-wikitext103-random/ckpt.pt`
- `out-wikitext103-ar/ckpt.pt`
- `out-wikitext103-random-b32/ckpt.pt` if continuing 32-block work
- `config/WikiText103/*.py`
- the scripts in the repo root

### Treat as exploratory / one-off
- `out-wikitext103-random_256_test_prefix/`
- many subfolders under `Report/`
- old `wandb/` runs

## 6. Minimal Cleanup Plan

Without moving any old files, the least risky cleanup strategy is:

1. Keep canonical checkpoints only in `out-*` directories.
2. Put all new analysis outputs under `Report/` with clearer topic names.
3. Avoid writing new evaluation csv/json files into checkpoint directories unless they are directly tied to that checkpoint.
4. Keep all new 32-block work on `_b32.py` configs and `*-b32` checkpoint dirs.
5. Treat `wandb/` as run logs only, not as a source of truth for experiment summaries.

## 7. Suggested Next Refactor

If you later want a real physical reorganization, do it in this order:

1. create `scripts/` and move benchmark/export scripts there
2. create `reports/` naming conventions and migrate only recent report folders
3. rename ad hoc checkpoint folders such as `out-wikitext103-random_256_test_prefix`
4. only after that, consider moving old historical artifacts

That keeps the refactor incremental and avoids breaking active experiments.
