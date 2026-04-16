# Preset config for scripts/runner/segment_curriculum_runner.py
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/block128/standard/segment_curriculum.py

config = "config/WikiText103/seq256/non_permute/block128/random.py"

train_out_dir = "out/curriculum/nonpermute/seq256/block128/out-wikitext103-random-b128-curriculum-attn"
benchmark_root = "Report/curriculum/nonpermute/seq256/block128/segment_curriculum_b128_big"

wandb_project = 'AOGPT-order-block'
wandb_run_name = 'seq256-random-b128-curriculum'

# block128 has 128 * 127 ordered pairs, so we keep pair mining selective while
# making the curriculum itself stronger across more stages.
warmup_iters = 6000
stage_iters = 6000
num_curriculum_stages = 4

segment_guided_ratios = "0.3,0.5,0.7,0.8"
segment_max_lens = "4,8,16,32"
segment_max_units_per_order = 4
segment_top_k_pairs = 48

# Benchmark settings
# Pair mining is much heavier than block32, so keep the pool cleaner instead
# of simply expanding it.
benchmark_batch_size = 64
benchmark_num_batches = 200
pair_mining_batches = 16
pair_eval_batch_size = 50
candidate_eval_batch_size = 64

random_pool_size = 64
structured_pool_size = 64
top_pair_pool_size = 96
aggregate_top_k_pairs = 32
prefix_len = 16
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10

pair_mining_mode = "attention_pruned"
attn_top_k = 10
attn_num_batches = 50
attn_batch_size = 64
attn_mode = "Random"
attn_symmetrize = "mean"
