# Train the backbone with top-k prefix policy orders from a frozen order head in the 32-block setup.

out_dir = 'out-wikitext103-random-b32'
init_from = 'resume'
eval_interval = 250
eval_iters = 100
log_interval = 10

wandb_log = True
wandb_project = 'ao-gpt-experiments-block-order'
wandb_run_name = 'wikitext103-prefix-policy-backbone-b32'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'policy_backbone'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''

n_layer = 3
n_head = 4
n_embd = 128
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
order_head_lr_mult = 1.0

learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
