# Preset config for scripts/runner/segment_curriculum_runner.py
# Usage:
#   python scripts/runner/segment_curriculum_runner.py config/WikiText103/seq512/non_permute/block1/segment_curriculum.py

config = 'config/WikiText103/seq512/non_permute/block1/random.py'

train_out_dir = 'out/curriculum/nonpermute/seq512/block1/out-wikitext103-seq512-random-b1-curriculum'
benchmark_root = 'Report/curriculum/nonpermute/seq512/block1/block1'

wandb_project = 'AOGPT-order-block'
wandb_run_name = 'seq512-random-b1-curriculum'

warmup_iters = 7000
stage_iters = 7000
num_curriculum_stages = 2

segment_guided_ratios = '0.15,0.3'
segment_max_lens = '8,12'
segment_max_units_per_order = 2
segment_top_k_pairs = 24

benchmark_batch_size = 16
benchmark_num_batches = 80
pair_mining_batches = 8
pair_eval_batch_size = 2
candidate_eval_batch_size = 16

random_pool_size = 24
structured_pool_size = 24
top_pair_pool_size = 48
aggregate_top_k_pairs = 24
prefix_len = 16
pair_score_k = 2
tv_weight = 0.3
benchmark_log_every_batches = 10

pair_mining_mode = 'attention_pruned'
attn_top_k = 8
attn_num_batches = 8
attn_batch_size = 8
attn_mode = 'Random'
attn_symmetrize = 'mean'
