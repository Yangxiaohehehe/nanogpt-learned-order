# Early-signal scan preset for block32 random training.
# This keeps iteration-named checkpoints so they can be batch-benchmarked later.

out_dir = 'out-wikitext103-random-b32-early-scan'
eval_interval = 500
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'ao-gpt-experiments-block-order-test-early'
wandb_run_name = 'wikitext103-random-b32-early-scan'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
n_layer = 3
n_head = 8
n_embd = 256
dropout = 0

block_order_block_len = 8
policy_prefix_k = 8
utility_horizon = 8
utility_alpha = 1.0
utility_beta = 1.0
baseline_momentum = 0.95
lambda_list = 0.1
order_temperature = 1.0
order_entropy_temperature = 1.0

learning_rate = 1e-3
max_iters = 6000
lr_decay_iters = 6000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0

always_save_checkpoint = True
save_iter_checkpoints = True
save_iter_checkpoint_dir = 'out-wikitext103-random-b32-early-scan/checkpoints'
save_iter_checkpoint_keep = 0
