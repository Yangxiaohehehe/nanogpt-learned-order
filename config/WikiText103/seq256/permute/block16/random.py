# WikiText103 seq256 permuted RANDOM config for the 16-block setup (block_len=16).

out_dir = 'out-wikitext103-seq256-random-b16-permute-block'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'ao-gpt-experiments-block-order'
wandb_run_name = 'wikitext103-seq256-random-b16-permute-block'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42
permute_mode = 'block'

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
n_layer = 3
n_head = 4
n_embd = 128
dropout = 0

block_order_block_len = 16
policy_prefix_k = 4
utility_horizon = 4
utility_alpha = 1.0
utility_beta = 1.0
baseline_momentum = 0.95
lambda_list = 0.1
order_temperature = 1.0
order_entropy_temperature = 1.0

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
