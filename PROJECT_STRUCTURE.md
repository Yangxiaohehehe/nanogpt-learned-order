# nanogpt_learned_order Structure Guide

This repository is now organized around four stable areas:

- root-level core code
- `scripts/` for runnable experiment scripts
- `config/` for reproducible presets
- `Report/` for saved experiment outputs

## 1. Top-Level Layout

### Core training/model code kept at repo root
- `AOGPT.py`
- `train.py`
- `order_utils.py`
- `model.py`
- `sample.py`
- `bench.py`
- `configurator.py`

These are the core files you are most likely to edit when changing model logic or training behavior.

### `scripts/`
All non-core runnable experiment scripts are grouped here.

- `scripts/runner/`
  Multi-stage orchestration scripts
- `scripts/eval/`
  Frozen-checkpoint evaluation scripts
- `scripts/benchmark/`
  Candidate mining, ranking, proposal, and structured-order benchmark scripts
- `scripts/analysis/`
  Higher-level comparison/analysis scripts

### `config/WikiText103/`
Experiment presets grouped by setting.

- `block16/standard/`
- `block32/standard/`
- `block64/standard/`
- `block128/standard/`
- `block_permute/`
- `legacy/`

### `Report/`
Saved outputs, reorganized by experiment type.

### Runtime-only directories
- `out*`
- `wandb/`
- `__pycache__/`
- `.pytest_cache/`

## 2. What Lives In `scripts/`

### `scripts/runner/`
- `segment_curriculum_runner.py`
  Warmup -> benchmark -> resume curriculum pipeline.

### `scripts/eval/`
- `eval_ckpt_modes.py`
- `permuted_ckpt_generate_step_eval.py`
- `block_permutation_sanity_check.py`
- `eval_prefix_pairs.py`

These are the scripts you use when you already have a checkpoint and want trajectories, curves, prefix behavior, or permutation sanity checks.

### `scripts/benchmark/`
- `structured_candidate_benchmark.py`
- `hierarchical_structured_benchmark.py`
- `staged_order_benchmark.py`
- `signal_benchmark.py`
- `metric_candidate_leaderboard.py`
- `metric_rank_benchmark.py`
- `proposal_benchmark_runner.py`
- `trajectory_signature_proposal.py`
- `trajectory_statistical_signature_benchmark.py`

These are the scripts for structured candidate mining, ranking, proposal generation, and benchmark-style comparisons.

### `scripts/analysis/`
- `ar_likeness_benchmark.py`

This is a higher-level comparison script for AR-vs-Random trajectory behavior.

## 3. Config Structure

### Standard block-size presets
Each `blockXX/standard/` folder contains the active presets for that block size:

- `ar.py`
- `random.py`
- `segment_curriculum.py`
- sometimes `segment_curriculum_fixed_order.py`

### Permuted block configs
- `config/WikiText103/block_permute/block32_random_permute.py`
- `config/WikiText103/block_permute/block32_segment_curriculum_permute.py`

### Legacy configs
- `config/WikiText103/legacy/`

Keep this only as historical reference unless you explicitly want an old setup.

## 4. Report Structure

`Report/` is now grouped by topic instead of keeping every experiment directory at the same level.

### `Report/docs/`
Manual documentation and experiment notes.

- `INDEX.md`
- `Readme_chinese.md`
- `Readme_english.md`
- `EXPERIMENT_CHECKLIST.md`
- `prompt.md`

### `Report/analysis/`
Higher-level comparison summaries.

Current example:
- `ar_likeness_benchmark_random_only_b200_mean_b32/`

### `Report/eval/`
Checkpoint evaluation outputs and sanity checks.

Current examples:
- `different_AR_RANDOM/`
- `eval_ckpt_modes_b32_base_permute_block/`
- `eval_pic_b32_permute_block_curriculum/`
- `block_permutation_sanity_b32_curriculum_permute_block/`

### `Report/leaderboards/`
Candidate leaderboard and ranking outputs.

Current examples:
- `metric_candidate_leaderboard_b32_b200_bs64/`
- `metric_candidate_leaderboard_b32_b200_bs64_n256/`
- `metric_candidate_leaderboard_b32_b200_bs64_n256_early_full/`
- `metric_candidate_leaderboard_b32_curriculum_permute_block_n256/`
- `metric_candidate_leaderboard_b32_permute_block_base_n256/`

### `Report/curriculum/`
Stage-wise curriculum outputs across block sizes and variants.

Current examples:
- `segment_curriculum_b16/`
- `segment_curriculum_b32/`
- `segment_curriculum_b32-4090/`
- `segment_curriculum_b32_fixed_order/`
- `segment_curriculum_b32_permute_block/`
- `segment_curriculum_b64/`
- `segment_curriculum_b128/`

Typical files inside a curriculum report:
- `runner_meta.json`
- `stage_01/results.json`, `stage_02/results.json`, ...
- `block_aggregation_trace.json`

### `Report/structured/`
Structured candidate benchmark outputs.

Current examples:
- `structured_candidate_benchmark_b32_pair2/`
- `structured_candidate_benchmark_b32_curriculum_permute_block/`
- `structured_candidate_benchmark_b32_permute_block_base/`

### `Report/staged/`
Staged-order or swap-style outputs.

Current examples:
- `staged_prefix_only_b32_b200_bs64/`
- `swap/`

### `Report/trajectory/`
Trajectory-signature proposal and statistical benchmark outputs.

Current examples:
- `trajectory_signature_proposal/`
- `trajectory_statistical_signature_benchmark/`

Important note:
large raw `*random_avg*.json` files in `Report/trajectory/trajectory_signature_proposal/` are intentionally ignored by Git if they exceed practical repository limits.

## 5. Where To Look For What

### I want to train or resume
Look at:

- `train.py`
- `config/WikiText103/...`

### I want to run curriculum training
Look at:

- `scripts/runner/segment_curriculum_runner.py`
- `config/WikiText103/.../segment_curriculum.py`
- `Report/curriculum/...`

### I want to evaluate a checkpoint
Look at:

- `scripts/eval/...`
- `Report/eval/...`

### I want to benchmark candidate orders or structured proposals
Look at:

- `scripts/benchmark/...`
- `Report/structured/...`
- `Report/leaderboards/...`
- `Report/trajectory/...`

## 6. What Is Source-Of-Truth vs Runtime Artifact

### Source-of-truth code
- root-level core files
- `scripts/`
- `config/`

### Source-of-truth reports
- curated outputs under `Report/`

### Runtime-only / resumable artifacts
- `out*`
- `wandb/`

## 7. Companion Command Guide

For copyable command examples, use:

- `RUN_COMMANDS.md`
