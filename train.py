"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import sys
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from AOGPT import AOGPTConfig, AOGPT
from order_utils import (
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    sample_random_block_orders,
    token_losses_to_block_losses,
    invert_permutation,
)

os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_MODE"] = "online"
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True 
wandb_project = 'ao-gpt-experiments' # 你的项目名称
wandb_run_name = 'mdm_random_order_run1' # 你的实验运行名称
# data
dataset = 'openwebtext'
permute_data = False
permute_seed = 42
permute_mode = 'block'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# AOGPT-only settings
aogpt_train_mode = 'AR'
main_eval_mode = 'Random'
generalization_eval_mode = ''
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
eval_generate_step_loss_log = True
eval_generate_step_batches = 200
eval_generate_step_loss_filename = 'generate_step_block_loss_latest.png'
train_stage = 'standard'
block_order_block_len = 16
segment_guided_ratio = 0.0
segment_source_json = ''
segment_top_k_pairs = 64
segment_max_len = 4
segment_max_units_per_order = 2
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
config['main_eval_mode'] = main_eval_mode
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
if block_size % block_order_block_len != 0:
    raise ValueError(f"block_size={block_size} must be divisible by block_order_block_len={block_order_block_len}")
num_blocks = block_size // block_order_block_len

np.random.seed(permute_seed)
if permute_data:
    if permute_mode != 'block':
        raise ValueError(f"Unsupported permute_mode={permute_mode!r}. Only 'block' is supported.")
    fixed_block_perm = build_fixed_block_permutation(num_blocks, permute_seed)
    inverse_block_perm = invert_permutation(fixed_block_perm)
    fixed_token_perm = block_permutation_to_token_permutation(
        fixed_block_perm,
        block_len=block_order_block_len,
    )
else:
    fixed_block_perm = None
    inverse_block_perm = None
    fixed_token_perm = None

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split, batch_size_override=None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    local_batch_size = batch_size if batch_size_override is None else int(batch_size_override)
    ix = torch.randint(len(data) - block_size, (local_batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if permute_data:
        perm_idx = fixed_token_perm.to(device)
        x = x[:, perm_idx]
        y = y[:, perm_idx]
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
resume_optimizer_state = True

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  block_order_block_len=block_order_block_len) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = AOGPTConfig(**model_args)
    model = AOGPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'block_order_block_len']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = AOGPTConfig(**model_args)
    model = AOGPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# elif init_from.startswith('gpt2'):
#     print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
#     # initialize from OpenAI GPT-2 weights
#     override_args = dict(dropout=dropout)
#     model = GPT.from_pretrained(init_from, override_args)
#     # read off the created config params, so we can store them into checkpoint correctly
#     for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#         model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

if train_stage != 'standard':
    raise ValueError(f"Unsupported train_stage={train_stage!r}. Only 'standard' is available.")


def _aggregate_top_pairs_to_segments(top_pairs, num_blocks_local, top_k, max_segment_len):
    top_k = max(1, min(int(top_k), len(top_pairs)))
    max_segment_len = max(2, int(max_segment_len))
    next_map = {}
    prev_map = {}

    def trace_start(node):
        while node in prev_map:
            node = prev_map[node]
        return node

    def build_segment(start):
        values = [start]
        seen = {start}
        node = start
        while node in next_map:
            node = next_map[node]
            if node in seen:
                break
            values.append(node)
            seen.add(node)
        return values

    for pair in top_pairs[:top_k]:
        first = int(pair["first"])
        second = int(pair["second"])
        if first == second:
            continue
        if first in next_map or second in prev_map:
            continue
        start_first = trace_start(first)
        start_second = trace_start(second)
        if start_first == start_second:
            continue
        seg_first = build_segment(start_first)
        seg_second = build_segment(start_second)
        if len(seg_first) + len(seg_second) > max_segment_len:
            continue
        next_map[first] = second
        prev_map[second] = first

    starts = [node for node in range(num_blocks_local) if node not in prev_map and node in next_map]
    segments = []
    for start in starts:
        values = build_segment(start)
        if len(values) >= 2:
            segments.append(values)
    segments.sort(key=lambda values: (-len(values), values[0], values[-1]))
    return segments


def _load_segment_library(segment_json_path):
    if not segment_json_path:
        return []
    if not os.path.exists(segment_json_path):
        raise FileNotFoundError(f"segment_source_json not found: {segment_json_path}")
    with open(segment_json_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    raw_segments = payload.get('aggregated_segments')
    if raw_segments:
        segments = [
            [int(v) for v in row["segment"]]
            for row in raw_segments
            if len(row.get("segment", [])) >= 2
        ]
    else:
        top_pairs = payload.get('top_pairs', [])
        segments = _aggregate_top_pairs_to_segments(
            top_pairs,
            num_blocks_local=num_blocks,
            top_k=segment_top_k_pairs,
            max_segment_len=segment_max_len,
        )
    cleaned = []
    for segment in segments:
        if any(v < 0 or v >= num_blocks for v in segment):
            continue
        if len(segment) >= 2:
            cleaned.append(segment)
    return cleaned


segment_library = _load_segment_library(segment_source_json)
if master_process and segment_guided_ratio > 0.0:
    if len(segment_library) == 0:
        print(
            f"warning: segment_guided_ratio={segment_guided_ratio} but no usable segments "
            f"were loaded from {segment_source_json or '[none]'}; falling back to pure random orders."
        )
    else:
        print(
            f"loaded {len(segment_library)} aggregated segments "
            f"for segment-guided random training from {segment_source_json or '[none]'}"
        )


def _sample_mixed_segment_guided_block_orders(batch_size_local, device_local):
    orders = []
    for _ in range(batch_size_local):
        if len(segment_library) == 0 or np.random.random() >= float(segment_guided_ratio):
            orders.append(torch.randperm(num_blocks, device=device_local))
            continue

        shuffled_segment_indices = np.random.permutation(len(segment_library)).tolist()
        chosen_segments = []
        used_blocks = set()
        for seg_idx in shuffled_segment_indices:
            segment = segment_library[seg_idx]
            if len(chosen_segments) >= int(segment_max_units_per_order):
                break
            if any(value in used_blocks for value in segment):
                continue
            chosen_segments.append(segment)
            used_blocks.update(segment)

        units = []
        for segment in chosen_segments:
            units.append(list(segment))
        for block_idx in range(num_blocks):
            if block_idx not in used_blocks:
                units.append([block_idx])

        unit_perm = np.random.permutation(len(units)).tolist()
        order = []
        for unit_idx in unit_perm:
            order.extend(units[unit_idx])
        orders.append(torch.tensor(order, device=device_local, dtype=torch.long))
    return torch.stack(orders, dim=0)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
)
if init_from == 'resume' and resume_optimizer_state:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, mode=main_eval_mode)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        if split == 'val':
            l2r_losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, l2r_loss = model(X, mode='AR')
                l2r_losses[k] = l2r_loss.item()
            out["val_l2r_loss"] = l2r_losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def estimate_eval_generate_step_block_loss_curves(num_batches_override=None):
    model.eval()
    local_num_batches = int(eval_generate_step_batches if num_batches_override is None else num_batches_override)
    ar_block_curves = []
    random_block_curves = []

    for _ in range(local_num_batches):
        X, _ = get_batch('val')
        with ctx:
            _, _, ar_token_losses = model(
                X,
                mode='AR',
                return_token_loss=True,
            )
            _, _, random_token_losses = model(
                X,
                mode='Random',
                return_token_loss=True,
            )
        ar_block_losses = token_losses_to_block_losses(
            ar_token_losses,
            block_len=block_order_block_len,
        )
        random_block_losses = token_losses_to_block_losses(
            random_token_losses,
            block_len=block_order_block_len,
        )
        ar_block_curves.append(ar_block_losses.mean(dim=0).float().cpu())
        random_block_curves.append(random_block_losses.mean(dim=0).float().cpu())

    model.train()
    ar_curve = torch.stack(ar_block_curves, dim=0).mean(dim=0).numpy()
    random_curve = torch.stack(random_block_curves, dim=0).mean(dim=0).numpy()
    return ar_curve, random_curve

def build_generate_step_block_loss_figure(ar_curve, random_curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    steps = np.arange(len(ar_curve))
    ax.plot(steps, ar_curve, label='AR mode', linewidth=2.0)
    ax.plot(steps, random_curve, label='Random mode', linewidth=2.0)
    ax.set_title(f'Val Mean Per-Step Block Loss ({eval_generate_step_batches} batches)')
    ax.set_xlabel('Generate / Reveal Block Step')
    ax.set_ylabel('Mean Block Loss')
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def save_figure_to_out_dir(fig, filename, dpi=200):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return save_path


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
latest_policy_train_stats = None

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * float(param_group.get('lr_scale', 1.0))

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            log_payload = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            if "val_l2r_loss" in losses:
                log_payload["val/l2r_loss"] = losses["val_l2r_loss"]
            if latest_policy_train_stats is not None:
                optional_policy_fields = {
                    "policy/segment_guided_ratio": "segment_guided_ratio",
                    "policy/segment_library_size": "segment_library_size",
                }
                for wandb_key, stats_key in optional_policy_fields.items():
                    if stats_key in latest_policy_train_stats:
                        log_payload[wandb_key] = latest_policy_train_stats[stats_key]
            if eval_generate_step_loss_log and train_stage == 'standard' and aogpt_train_mode == 'Random':
                ar_curve, random_curve = estimate_eval_generate_step_block_loss_curves()
                figure = build_generate_step_block_loss_figure(ar_curve, random_curve)
                latest_plot_path = save_figure_to_out_dir(
                    figure,
                    eval_generate_step_loss_filename,
                )
                log_payload["val/generate_step_block_loss_mean_ar"] = float(np.mean(ar_curve))
                log_payload["val/generate_step_block_loss_mean_random"] = float(np.mean(random_curve))
                log_payload["val/generate_step_block_loss_plot"] = wandb.Image(figure)
                log_payload["val/generate_step_block_loss_plot_path"] = latest_plot_path
                print(f"saved generate_step_block_loss_plot to {latest_plot_path}")
                import matplotlib.pyplot as plt
                plt.close(figure)
            wandb.log(log_payload)
        val_loss_for_ckpt = losses['val']
        if val_loss_for_ckpt < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss_for_ckpt
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if permute_data and fixed_block_perm is not None:
                    checkpoint['data_permutation'] = {
                        'permute_mode': permute_mode,
                        'permute_seed': int(permute_seed),
                        'block_perm': [int(v) for v in fixed_block_perm.tolist()],
                        'inverse_block_perm': [int(v) for v in inverse_block_perm.tolist()],
                    }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if (
                train_stage == 'standard'
                and aogpt_train_mode == 'Random'
                and float(segment_guided_ratio) > 0.0
                and len(segment_library) > 0
            ):
                mixed_block_orders = _sample_mixed_segment_guided_block_orders(
                    X.size(0),
                    X.device,
                )
                mixed_token_orders = expand_block_orders_to_token_orders(
                    mixed_block_orders,
                    block_len=block_order_block_len,
                )
                logits, loss = model(X, mode=None, orders=mixed_token_orders)
                latest_policy_train_stats = {
                    "segment_guided_ratio": float(segment_guided_ratio),
                    "segment_library_size": float(len(segment_library)),
                }
            else:
                logits, loss = model(X, mode=aogpt_train_mode)
                latest_policy_train_stats = None
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
