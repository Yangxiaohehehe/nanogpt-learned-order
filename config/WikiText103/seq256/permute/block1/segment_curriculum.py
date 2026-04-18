# Preset config for scripts/runner/segment_curriculum_runner.py
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/seq256/permute/block1/segment_curriculum.py

config = 'config/WikiText103/seq256/permute/block1/random.py'

train_out_dir = 'out/curriculum/permute/seq256/block1/out-wikitext103-seq256-random-b1-curriculum-permute-block'
benchmark_root = 'Report/curriculum/permute/seq256/block1/block1'

wandb_project = 'AOGPT-order-block'
wandb_run_name = 'seq256-random-b1-permute-curriculum'

warmup_iters = 3000
stage_iters = 5000
num_curriculum_stages = 4

# More aggressive curriculum for token-level (block1) runs:
# - keep early stages conservative so mined segments are stabler
# - expand max segment length over stages instead of jumping too early
# - increase late-stage structure usage once segment quality improves
segment_guided_ratios = '0.3,0.5,0.7,0.9'
segment_max_lens = '6,12,20,32'
segment_max_units_per_order = 6
segment_top_k_pairs = 96
attn_export_type = 'with_none'
benchmark_batch_size = 32
benchmark_num_batches = 64
pair_mining_batches = 20
pair_eval_batch_size = 16
candidate_eval_batch_size = 32

random_pool_size = 32
structured_pool_size = 32
top_pair_pool_size = 64
aggregate_top_k_pairs = 64
prefix_len = 16
skip_candidate_pool_eval = True
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_num_batches = 20
attn_batch_size = 24
attn_mode = 'Random'
attn_symmetrize = 'mean'

# Parameters below are retained for CLI/config compatibility, but when
# skip_candidate_pool_eval=True they do not affect the fast curriculum path:
# - benchmark_num_batches
# - candidate_eval_batch_size
# - random_pool_size
# - structured_pool_size
# - top_pair_pool_size
# - prefix_len
# - benchmark_log_every_batches
