# AR training with permuted data. This is kept as a legacy exploratory config.

out_dir = 'out-wikitext103-ar-permute'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'ao-gpt-experiments-128-3'
wandb_run_name = 'wikitext103-ar-permute'

dataset = 'wikitext103'
batch_size = 64
block_size = 128
gradient_accumulation_steps = 2
permute_data = True
permute_seed = 42

model_type = 'aogpt'
aogpt_train_mode = 'AR'
main_eval_mode = 'AR'
generalization_eval_mode = ''
n_layer = 4
n_head = 8
n_embd = 256
dropout = 0

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
