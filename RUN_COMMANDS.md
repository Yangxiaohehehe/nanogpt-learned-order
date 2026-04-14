# nanogpt_learned_order Run Commands

This file groups the active Python scripts by purpose and gives copyable command templates.
All commands assume you run from the repo root:

```bash
cd /home/devbox/project/AOGPT-test-order/nanogpt_learned_order
```

If you use your virtual environment:

```bash
source /home/devbox/project/bin/activate
```

## 1. Training

### Standard single-config training

Block16 random:

```bash
python train.py config/WikiText103/block16/standard/random.py
```

Block32 random:

```bash
python train.py config/WikiText103/block32/standard/random.py
```

Block32 random early-signal scan preset:

```bash
python train.py config/WikiText103/block32/standard/random_early_scan.py
```

Block64 random:

```bash
python train.py config/WikiText103/block64/standard/random.py
```

Block128 random:

```bash
python train.py config/WikiText103/block128/standard/random.py
```

AR baseline:

```bash
python train.py config/WikiText103/block32/standard/ar.py
```

### Resume from an existing out_dir

```bash
python train.py config/WikiText103/block32/standard/random.py --init_from=resume --out_dir=out-wikitext103-random-b32-curriculum
```

## 2. Segment Curriculum Runs

### Run curriculum preset directly

Block16:

```bash
python scripts/runner/segment_curriculum_runner.py config/WikiText103/block16/standard/segment_curriculum.py
```

Block32:

```bash
python scripts/runner/segment_curriculum_runner.py config/WikiText103/block32/standard/segment_curriculum.py
```

Block32 fixed-order variant:

```bash
python scripts/runner/segment_curriculum_runner.py config/WikiText103/block32/standard/segment_curriculum_fixed_order.py
```

Block64:

```bash
python scripts/runner/segment_curriculum_runner.py config/WikiText103/block64/standard/segment_curriculum.py
```

Block128:

```bash
python scripts/runner/segment_curriculum_runner.py config/WikiText103/block128/standard/segment_curriculum.py
```

### Permuted block curriculum

```bash
python scripts/runner/segment_curriculum_runner.py config/WikiText103/block_permute/block32_segment_curriculum_permute.py
```

### Permuted block early-signal scan training

```bash
python train.py config/WikiText103/block_permute/block32_random_permute_early_scan.py
```

## 3. Frozen Checkpoint Evaluation

### AR / Random trajectory evaluation

```bash
python scripts/eval/eval_ckpt_modes.py \
  --ckpt_path out-wikitext103-random-b32-curriculum/ckpt.pt \
  --out_dir Report/eval/eval_ckpt_modes_b32_example
```

### Generate-step evaluation for permuted checkpoints

```bash
python scripts/eval/permuted_ckpt_generate_step_eval.py \
  --ckpt_path out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt \
  --out_dir Report/eval/eval_pic_b32_permute_block_example
```

### Block-permutation sanity check

```bash
python scripts/eval/block_permutation_sanity_check.py \
  --ckpt_path out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt \
  --out_dir Report/eval/block_permutation_sanity_example
```

### Fixed two-block prefix evaluation

```bash
python scripts/eval/eval_prefix_pairs.py \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_csv Report/eval/prefix_pairs_b32.csv
```

## 4. Structured Candidate Benchmarks

### Standard structured candidate benchmark

```bash
python scripts/benchmark/structured_candidate_benchmark.py \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/structured/structured_candidate_benchmark_b32_example
```

### Hierarchical structured benchmark

```bash
python scripts/benchmark/hierarchical_structured_benchmark.py \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/structured/hierarchical_structured_benchmark_b32_example
```

### Staged order benchmark

Prefix-only:

```bash
python scripts/benchmark/staged_order_benchmark.py \
  --experiment prefix_only \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/staged/staged_prefix_only_example
```

Two-stage:

```bash
python scripts/benchmark/staged_order_benchmark.py \
  --experiment two_stage \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/staged/staged_two_stage_example
```

## 5. Signal / Metric Benchmarks

### Signal benchmark

```bash
python scripts/benchmark/signal_benchmark.py \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/analysis/signal_benchmark_example
```

### Candidate metric leaderboard

```bash
python scripts/benchmark/metric_candidate_leaderboard.py \
  --ckpt_path out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/leaderboards/metric_candidate_leaderboard_example
```

### Metric rank benchmark

```bash
python scripts/benchmark/metric_rank_benchmark.py
```

## 6. Proposal Benchmarks

### Proposal benchmark runner

Single checkpoint:

```bash
python scripts/benchmark/proposal_benchmark_runner.py \
  --ckpt out-wikitext103-random-b32-curriculum/ckpt.pt \
  --proposal full_enum,random,attn_early,loo_early \
  --out_dir Report/trajectory/proposal_benchmark_example
```

### Trajectory-signature proposal extraction

```bash
python scripts/benchmark/trajectory_signature_proposal.py \
  --ckpt_path out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt \
  --out_path Report/trajectory/trajectory_signature_proposal/example.json \
  --raw_block_size 32 \
  --num_batches 8 \
  --batch_size 4
```

### Statistical trajectory-signature benchmark

```bash
python scripts/benchmark/trajectory_statistical_signature_benchmark.py \
  --ckpt_path out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt \
  --out_path Report/trajectory/trajectory_statistical_signature_benchmark/example.json \
  --raw_block_size 32 \
  --num_samples 16
```

## 7. AR-Likeness / Analysis

### AR-likeness benchmark

```bash
python scripts/analysis/ar_likeness_benchmark.py \
  --ckpt_paths out/out-wikitext103-random-b32/ckpt.pt \
  --out_dir Report/analysis/ar_likeness_benchmark_example
```

### Early checkpoint signal scan

```bash
python scripts/runner/early_signal_scan.py \
  --checkpoint_dir out-wikitext103-random-b32-early-scan/checkpoints \
  --out_root Report/analysis/early_signal_scan_b32 \
  --save_all_pair_scores \
  --steps 500,1000,1500,2000,3000,4000,5000,6000
```

### Early checkpoint signal scan for permuted block training

```bash
python scripts/runner/early_signal_scan.py \
  --checkpoint_dir out-wikitext103-random-b32-permute-block-early-scan/checkpoints \
  --out_root Report/analysis/early_signal_scan_b32_permute \
  --save_all_pair_scores \
  --steps 500,1000,1500,2000,3000,4000,5000,6000
```

## 8. Useful Output Conventions

### Checkpoints

Common checkpoint locations:

- `out-wikitext103-random-b16-curriculum/ckpt.pt`
- `out-wikitext103-random-b32-curriculum/ckpt.pt`
- `out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt`
- `out-wikitext103-random-b64-curriculum/ckpt.pt`
- `out-wikitext103-random-b128-curriculum/ckpt.pt`
- `out/out-wikitext103-random-b32/ckpt.pt`

### Reports

Most eval / benchmark scripts write only under:

- `Report/<topic_name>/...`
- preferably within:
  `Report/analysis/`, `Report/eval/`, `Report/leaderboards/`, `Report/curriculum/`, `Report/structured/`, `Report/staged/`, or `Report/trajectory/`

That is the preferred place for analysis outputs.

## 9. Recommended Workflow

### Train a new checkpoint

```bash
python train.py config/WikiText103/block32/standard/random.py
```

### Run curriculum

```bash
python scripts/runner/segment_curriculum_runner.py config/WikiText103/block32/standard/segment_curriculum.py
```

### Evaluate the checkpoint

```bash
python scripts/benchmark/structured_candidate_benchmark.py \
  --ckpt_path out-wikitext103-random-b32-curriculum/ckpt.pt \
  --out_dir Report/structured/structured_candidate_benchmark_b32_curriculum
```

### Run trajectory eval

```bash
python scripts/eval/eval_ckpt_modes.py \
  --ckpt_path out-wikitext103-random-b32-curriculum/ckpt.pt \
  --out_dir Report/eval/eval_ckpt_modes_b32_curriculum
```

## 10. Notes

- `train.py` uses config files plus `--key=value` overrides.
- Many benchmark scripts can infer `dataset` from the checkpoint config, so `--dataset` is often optional.
- For permuted-data checkpoints, prefer scripts that already understand block permutation:
  - `scripts/eval/eval_ckpt_modes.py`
  - `scripts/eval/block_permutation_sanity_check.py`
  - `scripts/benchmark/metric_candidate_leaderboard.py`
  - `scripts/benchmark/structured_candidate_benchmark.py`
  - `scripts/benchmark/proposal_benchmark_runner.py`
- Very large raw proposal JSON files should stay local and not be committed to Git.
