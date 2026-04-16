# Preset config for scripts/runner/segment_curriculum_runner.py
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/seq256/permute/block1/segment_curriculum.py

config = 'config/WikiText103/seq256/permute/block1/random.py'

train_out_dir = 'out/curriculum/permute/seq256/block1/out-wikitext103-seq256-random-b1-curriculum-permute-block'
benchmark_root = 'Report/curriculum/permute/seq256/block1/block1'

wandb_project = 'AOGPT-order-block'
wandb_run_name = 'seq256-random-b1-permute-curriculum'

warmup_iters = 7000
stage_iters = 7000
num_curriculum_stages = 2

segment_guided_ratios = '0.2,0.35'
segment_max_lens = '8,12'
segment_max_units_per_order = 2
segment_top_k_pairs = 32

benchmark_batch_size = 32
benchmark_num_batches = 100
pair_mining_batches = 12
pair_eval_batch_size = 4
candidate_eval_batch_size = 32

random_pool_size = 32
structured_pool_size = 32
top_pair_pool_size = 64
aggregate_top_k_pairs = 32
prefix_len = 16
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_num_batches = 12
attn_batch_size = 16
attn_mode = 'Random'
attn_symmetrize = 'mean'
