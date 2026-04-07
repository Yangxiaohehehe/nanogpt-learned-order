# Preset config for segment_curriculum_runner.py
# Usage:
#   python segment_curriculum_runner.py config/WikiText103/block64/standard/segment_curriculum.py

config = "config/WikiText103/block64/standard/random.py"

train_out_dir = "out-wikitext103-random-b64-curriculum"
benchmark_root = "Report/segment_curriculum_b64"

# block64 has 64 * 63 ordered pairs, so keep the first run conservative.
warmup_iters = 7000
stage_iters = 7000
num_curriculum_stages = 2

segment_guided_ratios = "0.3,0.5"
segment_max_lens = "4,6"
segment_max_units_per_order = 2
segment_top_k_pairs = 64

disable_order_head = True

# Benchmark settings
# Pair mining is much heavier than block32, so keep these moderate first.
benchmark_batch_size = 64
benchmark_num_batches = 200
pair_mining_batches = 16
pair_eval_batch_size = 8
candidate_eval_batch_size = 64

random_pool_size = 64
structured_pool_size = 64
top_pair_pool_size = 128
aggregate_top_k_pairs = 64
prefix_len = 16
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10
