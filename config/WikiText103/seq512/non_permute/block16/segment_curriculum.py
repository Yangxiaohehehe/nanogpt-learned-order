# Preset config for scripts/runner/segment_curriculum_runner.py
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/seq512/non_permute/block16/segment_curriculum.py

config = 'config/WikiText103/seq512/non_permute/block16/random.py'

train_out_dir = 'out/curriculum/nonpermute/seq512/block16/out-wikitext103-seq512-random-b16-curriculum'
benchmark_root = 'Report/curriculum/nonpermute/seq512/block16/block16'

wandb_project = 'AOGPT-order-block'
wandb_run_name = 'seq512-random-b16-curriculum'

warmup_iters = 7000
stage_iters = 7000
num_curriculum_stages = 2

segment_guided_ratios = '0.3,0.5'
segment_max_lens = '4,6'
segment_max_units_per_order = 2
segment_top_k_pairs = 64

benchmark_batch_size = 64
pair_mining_batches = 24
pair_eval_batch_size = 16

aggregate_top_k_pairs = 64
skip_candidate_pool_eval = True
pair_score_k = 2
tv_weight = 0.3

pair_mining_mode = 'attention_pruned'
attn_top_k = 4
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
