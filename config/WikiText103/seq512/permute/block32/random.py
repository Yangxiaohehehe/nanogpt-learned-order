# WikiText103 seq512 permuted RANDOM config for the 32-block setup (block_len=16).

out_dir = 'out/base/permute/seq512/block32/out-wikitext103-seq512-random-b32-permute-block'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-base-block'
wandb_run_name = 'seq512-random-b32-permute'

dataset = 'wikitext103'
batch_size = 32
block_size = 512
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
n_head = 8
n_embd = 256
dropout = 0

block_order_block_len = 16

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
