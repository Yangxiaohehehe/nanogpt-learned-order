# Preset config for scripts/runner/segment_curriculum_runner.py
# Goal:
#   Bias later stages toward a small set of stable structured orders.
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/block32/standard/segment_curriculum_fixed_order.py

config = "config/WikiText103/block32/standard/random.py"

train_out_dir = "out-wikitext103-random-b32-curriculum-fixed-order"
benchmark_root = "Report/curriculum/segment_curriculum_b32_fixed_order"

# Warm up with pure Random, then progressively consolidate onto a smaller
# family of structured orders.
warmup_iters = 6000
stage_iters = 6000
num_curriculum_stages = 3

# Later stages rely more heavily on mined structure.
segment_guided_ratios = "0.3,0.5,0.7"

# Allow longer aggregated segments in later stages.
segment_max_lens = "4,6,8"
segment_max_units_per_order = 3
segment_top_k_pairs = 48

# Benchmark settings
benchmark_batch_size = 128
benchmark_num_batches = 200
pair_mining_batches = 24
pair_eval_batch_size = 16
candidate_eval_batch_size = 96

random_pool_size = 64
structured_pool_size = 64
top_pair_pool_size = 96

# Use fewer top pairs for aggregation so the final structure is more concentrated.
aggregate_top_k_pairs = 32
prefix_len = 8
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10

pair_mining_mode = "attention_pruned"
attn_top_k = 4
attn_num_batches = 24
attn_batch_size = 32
attn_mode = "Random"
attn_symmetrize = "mean"
