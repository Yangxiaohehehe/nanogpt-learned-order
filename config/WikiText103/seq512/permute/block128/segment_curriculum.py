# Preset config for scripts/runner/segment_curriculum_runner.py
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/seq512/permute/block128/segment_curriculum.py

config = 'config/WikiText103/seq512/permute/block128/random.py'

train_out_dir = 'out/curriculum/permute/seq512/block128/out-wikitext103-seq512-random-b128-curriculum-permute-block'
benchmark_root = 'Report/curriculum/permute/seq512/block128/block128'

wandb_project = 'AOGPT-order-block'
wandb_run_name = 'seq512-random-b128-permute-curriculum'

warmup_iters = 7000
stage_iters = 7000
num_curriculum_stages = 3

segment_guided_ratios = '0.3,0.5,0.7'
segment_max_lens = '4,6,8'
segment_max_units_per_order = 3
segment_top_k_pairs = 48

benchmark_batch_size = 32
pair_mining_batches = 16
pair_eval_batch_size = 8

aggregate_top_k_pairs = 32
skip_candidate_pool_eval = True
pair_score_k = 2
tv_weight = 0.3

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_num_batches = 24
attn_batch_size = 32
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
