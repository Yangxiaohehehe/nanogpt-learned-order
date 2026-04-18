# WikiText103 seq128 non-permuted RANDOM config for the token-level setup (block_len=1).

out_dir = 'out/base/nonpermute/seq128/block1/out-wikitext103-seq128-random-b1'
eval_interval = 250
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = 'AOGPT-base-block'
wandb_run_name = 'seq128-random-b1'

dataset = 'wikitext103'
batch_size = 64
block_size = 128
gradient_accumulation_steps = 2
permute_data = False
permute_seed = 42

model_type = 'aogpt'
train_stage = 'standard'
aogpt_train_mode = 'Random'
main_eval_mode = 'Random'
generalization_eval_mode = ''
order_impl = 'token'
n_layer = 3
n_head = 8
n_embd = 256
dropout = 0

block_order_block_len = 1

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 0
