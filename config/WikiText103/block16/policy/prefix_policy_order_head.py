# Train only the prefix-policy order head on top of a pretrained Random backbone.

out_dir = 'out-wikitext103-random'
init_from = 'resume'
eval_interval = 250
eval_iters = 100
log_interval = 10

wandb_log = True
wandb_project = 'ao-gpt-experiments-block-order'
wandb_run_name = 'wikitext103-prefix-policy-order-head'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'order_head'
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
order_head_lr_mult = 1.0

learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 0
