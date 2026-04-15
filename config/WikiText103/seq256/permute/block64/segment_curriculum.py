# Preset config for scripts/runner/segment_curriculum_runner.py
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/seq256/permute/block64/segment_curriculum.py

config = 'config/WikiText103/seq256/permute/block64/random.py'

train_out_dir = 'out-wikitext103-seq256-random-b64-curriculum-permute-block'
benchmark_root = 'Report/curriculum/seq256/permute/block64'

warmup_iters = 7000
stage_iters = 7000
num_curriculum_stages = 3

segment_guided_ratios = '0.3,0.5,0.7'
segment_max_lens = '4,8,12'
segment_max_units_per_order = 4
segment_top_k_pairs = 48

benchmark_batch_size = 64
benchmark_num_batches = 200
pair_mining_batches = 16
pair_eval_batch_size = 8
candidate_eval_batch_size = 64

random_pool_size = 64
structured_pool_size = 64
top_pair_pool_size = 96
aggregate_top_k_pairs = 32
prefix_len = 16
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_num_batches = 24
attn_batch_size = 32
attn_mode = 'Random'
attn_symmetrize = 'mean'
