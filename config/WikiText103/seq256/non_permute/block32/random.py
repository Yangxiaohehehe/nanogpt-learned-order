# WikiText103 seq256 non-permuted RANDOM config for the 32-block setup (block_len=8).

out_dir = 'out/base/nonpermute/seq256/block32/out-wikitext103-seq256-random-b32'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-base-block'
wandb_run_name = 'seq256-random-b32'

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

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
