# Preset config for segment_curriculum_runner.py
# Usage:
#   python segment_curriculum_runner.py config/WikiText103/block16/standard/segment_curriculum.py

config = "config/WikiText103/block16/standard/random.py"

train_out_dir = "out-wikitext103-random-b16-curriculum"
benchmark_root = "Report/segment_curriculum_b16"

# block16 is the lightest setup here, so we can afford a slightly stronger
# benchmark configuration than block32/block64.
warmup_iters = 7000
stage_iters = 7000
num_curriculum_stages = 2

segment_guided_ratios = "0.3,0.5"
segment_max_lens = "4,6"
segment_max_units_per_order = 2
segment_top_k_pairs = 64

disable_order_head = True

# Benchmark settings
benchmark_batch_size = 128
benchmark_num_batches = 200
pair_mining_batches = 24
pair_eval_batch_size = 16
candidate_eval_batch_size = 96

random_pool_size = 64
structured_pool_size = 64
top_pair_pool_size = 128
aggregate_top_k_pairs = 64
prefix_len = 4
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10
