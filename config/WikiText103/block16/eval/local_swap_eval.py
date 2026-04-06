# Local adjacent-swap search evaluation on a pretrained checkpoint.

out_dir = 'out-wikitext103-random'
init_from = 'resume'
eval_interval = 250
log_interval = 10

wandb_log = True
wandb_project = 'ao-gpt-experiments-block-order'
wandb_run_name = 'wikitext103-local-swap-eval'

dataset = 'wikitext103'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

train_stage = 'local_swap_eval'
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

swap_eval_prefix_k = 2
swap_eval_num_steps = 10
swap_eval_num_batches = 4
swap_eval_batch_size = 8
swap_eval_split = 'val'
swap_eval_csv = 'local_swap_eval.csv'
