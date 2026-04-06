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
import csv
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from AOGPT import AOGPTConfig, AOGPT
from order_utils import (
    build_prefix_policy_block_orders,
    compute_order_entropy,
    compute_block_residual_utilities,
    compute_early_shaping_preference_quality,
    compute_prefix_auc,
    expand_block_orders_to_token_orders,
    evaluate_block_order_quality,
    greedy_adjacent_swap_search,
    kendall_tau_to_l2r,
    pairwise_order_preference_loss,
    prefix_position_stats,
    sample_random_block_orders,
    scatter_block_utilities_to_original_positions,
    token_losses_to_block_losses,
    update_stepwise_ema_baseline,
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
order_head_lr_mult = 5.0
train_stage = 'standard' # 'standard', 'order_head', 'policy_backbone', 'joint', 'local_swap_eval'
block_order_block_len = 16
policy_prefix_k = 4
utility_horizon = 4
utility_alpha = 1.0
utility_beta = 1.0
baseline_momentum = 0.95
lambda_list = 0.1
order_temperature = 1.0
order_entropy_temperature = 1.0
preference_num_candidates = 4
preference_early_k = 4
preference_tv_weight = 0.3
preference_margin = 0.0
swap_eval_prefix_k = 2
swap_eval_num_steps = 3
swap_eval_num_batches = 4
swap_eval_batch_size = 8
swap_eval_split = 'val'
swap_eval_csv = 'local_swap_eval.csv'
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
fixed_perm = torch.tensor(np.random.permutation(block_size), dtype=torch.long) if permute_data else None

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
        perm_idx = fixed_perm.to(device)
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
    checkpoint_train_stage = checkpoint.get('config', {}).get('train_stage', 'standard')
    stage_changed = checkpoint_train_stage != train_stage
    if stage_changed:
        # Across training stages we reuse model weights only.
        # The trainable parameter sets and optimizer groups differ between stages,
        # so reloading the old optimizer state is both invalid and misleading.
        resume_optimizer_state = False
        iter_num = 0
        best_val_loss = 1e9
        print(
            f"Loaded model weights from stage '{checkpoint_train_stage}' into "
            f"stage '{train_stage}'. Resetting optimizer state and training counters."
        )
    else:
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

step_mean_baseline = torch.zeros(num_blocks, device=device, dtype=torch.float32)
step_var_baseline = torch.zeros(num_blocks, device=device, dtype=torch.float32)
baseline_initialized = False
if init_from == 'resume' and resume_optimizer_state:
    baseline_state = checkpoint.get('block_order_baselines')
    if baseline_state is not None:
        step_mean_baseline = torch.tensor(baseline_state['step_mean'], device=device, dtype=torch.float32)
        step_var_baseline = torch.tensor(baseline_state['step_var'], device=device, dtype=torch.float32)
        baseline_initialized = bool(baseline_state.get('initialized', True))

def set_train_stage_requires_grad(module, stage):
    for name, param in module.named_parameters():
        if stage == 'order_head':
            param.requires_grad = name.startswith('policy_order_head.')
        elif stage == 'policy_backbone':
            param.requires_grad = not name.startswith('policy_order_head.')
        elif stage == 'joint':
            param.requires_grad = True
        elif stage == 'local_swap_eval':
            param.requires_grad = False
        else:
            param.requires_grad = True

set_train_stage_requires_grad(model, train_stage)

scaler = None
optimizer = None
if train_stage != 'local_swap_eval':
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay,
        learning_rate,
        (beta1, beta2),
        device_type,
        order_head_lr_mult=order_head_lr_mult,
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

def _forward_policy_features(module, idx, need_token_losses=False, track_grad=False):
    random_block_orders = sample_random_block_orders(
        idx.size(0),
        num_blocks,
        idx.device,
    )
    random_token_orders = expand_block_orders_to_token_orders(
        random_block_orders,
        block_len=block_order_block_len,
    )
    forward_ctx = nullcontext() if track_grad else torch.no_grad()
    with forward_ctx:
        with ctx:
            outputs = module(
                idx,
                mode=None,
                orders=random_token_orders,
                return_token_loss=need_token_losses,
                return_hidden=True,
            )
    if need_token_losses:
        _, _, token_losses, hidden_states = outputs
    else:
        _, _, hidden_states = outputs
        token_losses = None
    return random_block_orders, random_token_orders, token_losses, hidden_states

def _score_policy_head(module, idx, hidden_states, detach_inputs):
    return module.score_prefix_policy(idx, hidden_states, detach_inputs=detach_inputs)

def _build_policy_orders_from_batch(module, idx, track_grad=False):
    _, _, _, hidden_states = _forward_policy_features(
        module,
        idx,
        need_token_losses=False,
        track_grad=track_grad,
    )
    scores = _score_policy_head(
        module,
        idx,
        hidden_states,
        detach_inputs=not track_grad,
    )
    block_orders = build_prefix_policy_block_orders(scores, policy_prefix_k)
    token_orders = expand_block_orders_to_token_orders(
        block_orders,
        block_len=block_order_block_len,
    )
    return scores, block_orders, token_orders, hidden_states

def _policy_stats(scores, block_orders, block_losses=None, token_losses=None, residual_step_utilities=None):
    stats = {
        "order_entropy": float(compute_order_entropy(scores, temperature=order_entropy_temperature).item()),
        "kendall_tau": float(kendall_tau_to_l2r(block_orders).item()),
    }
    prefix_stats = prefix_position_stats(block_orders, policy_prefix_k)
    stats.update({
        "prefix_mean_index": float(prefix_stats["prefix_mean_index"].item()),
        "prefix_std_index": float(prefix_stats["prefix_std_index"].item()),
        "prefix_min_index": float(prefix_stats["prefix_min_index"].item()),
        "prefix_max_index": float(prefix_stats["prefix_max_index"].item()),
    })
    if block_losses is not None:
        stats["prefix_auc"] = float(compute_prefix_auc(block_losses, policy_prefix_k).item())
    if token_losses is not None:
        stats["token_prefix_auc"] = float(compute_prefix_auc(token_losses, policy_prefix_k * block_order_block_len).item())
    if residual_step_utilities is not None:
        stats["residual_mean"] = float(residual_step_utilities.mean().item())
        stats["residual_std"] = float(residual_step_utilities.std(unbiased=False).item())
    return stats


def _compute_residual_block_targets(block_losses, update_baseline=True):
    global step_mean_baseline, step_var_baseline, baseline_initialized

    (
        step_utilities,
        future_means,
        future_vars,
        residual_means,
        residual_vars,
    ) = compute_block_residual_utilities(
        block_losses,
        step_mean_baseline,
        step_var_baseline,
        horizon=utility_horizon,
        alpha=utility_alpha,
        beta=utility_beta,
    )
    batch_future_mean = future_means.mean(dim=0)
    batch_future_var = future_vars.mean(dim=0)
    if update_baseline:
        step_mean_baseline = update_stepwise_ema_baseline(
            step_mean_baseline,
            batch_future_mean,
            momentum=baseline_momentum,
            initialized=baseline_initialized,
        )
        step_var_baseline = update_stepwise_ema_baseline(
            step_var_baseline,
            batch_future_var,
            momentum=baseline_momentum,
            initialized=baseline_initialized,
        )
        baseline_initialized = True
    return {
        "step_utilities": step_utilities,
        "future_means": future_means,
        "future_vars": future_vars,
        "residual_means": residual_means,
        "residual_vars": residual_vars,
    }


@torch.no_grad()
def _sample_order_preference_targets(module, idx):
    num_candidates = max(2, int(preference_num_candidates))
    candidate_orders = sample_random_block_orders(
        idx.size(0) * num_candidates,
        num_blocks,
        idx.device,
    ).view(idx.size(0), num_candidates, num_blocks)
    flat_orders = candidate_orders.reshape(idx.size(0) * num_candidates, num_blocks)
    flat_idx = idx.unsqueeze(1).expand(idx.size(0), num_candidates, idx.size(1)).reshape(
        idx.size(0) * num_candidates,
        idx.size(1),
    )
    candidate_metrics = evaluate_block_order_quality(
        module,
        flat_idx,
        flat_orders,
        prefix_k=max(2, min(preference_early_k, num_blocks)),
        block_len=block_order_block_len,
        autocast_context=ctx,
    )
    candidate_block_losses = candidate_metrics["block_losses"].view(idx.size(0), num_candidates, num_blocks)
    candidate_quality = compute_early_shaping_preference_quality(
        candidate_block_losses.view(idx.size(0) * num_candidates, num_blocks),
        early_k=preference_early_k,
        tv_weight=preference_tv_weight,
    ).view(idx.size(0), num_candidates)
    best_idx = candidate_quality.argmax(dim=-1)
    worst_idx = candidate_quality.argmin(dim=-1)
    batch_indices = torch.arange(idx.size(0), device=idx.device)
    preferred_orders = candidate_orders[batch_indices, best_idx]
    other_orders = candidate_orders[batch_indices, worst_idx]
    preferred_quality = candidate_quality[batch_indices, best_idx]
    other_quality = candidate_quality[batch_indices, worst_idx]
    return {
        "preferred_orders": preferred_orders,
        "other_orders": other_orders,
        "preferred_quality": preferred_quality,
        "other_quality": other_quality,
        "quality_gap": preferred_quality - other_quality,
    }


@torch.no_grad()
def run_local_swap_eval(module):
    """
    Greedy adjacent-swap local-search experiment.

    This is an evaluation-only diagnostic that starts from random block orders
    and applies local adjacent swaps using prefix AUC as the quality function.
    It is not a training signal and does not use l2r as supervision.
    """
    module.eval()
    csv_path = os.path.join(out_dir, swap_eval_csv)
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    rows = []
    initial_prefix = []
    final_prefix = []
    initial_full = []
    final_full = []
    initial_kendall = []
    final_kendall = []
    improved_prefix_count = 0
    improved_kendall_count = 0
    total_samples = 0

    for batch_idx in range(int(swap_eval_num_batches)):
        X, _ = get_batch(swap_eval_split, batch_size_override=swap_eval_batch_size)
        init_block_orders = sample_random_block_orders(
            X.size(0),
            num_blocks,
            X.device,
        )
        initial_metrics = evaluate_block_order_quality(
            module,
            X,
            init_block_orders,
            prefix_k=swap_eval_prefix_k,
            block_len=block_order_block_len,
            autocast_context=ctx,
        )
        search_result = greedy_adjacent_swap_search(
            module,
            X,
            init_block_orders,
            num_steps=swap_eval_num_steps,
            prefix_k=swap_eval_prefix_k,
            block_len=block_order_block_len,
            autocast_context=ctx,
        )

        history = search_result["history"]
        batch_size_local = init_block_orders.size(0)
        total_samples += batch_size_local
        final_prefix_step = history[-1]["prefix_auc_per_sample"]
        final_full_step = history[-1]["full_loss_per_sample"]
        final_kendall_step = history[-1]["kendall_per_sample"]
        initial_prefix.extend(initial_metrics["prefix_auc_per_sample"].detach().cpu().tolist())
        final_prefix.extend(final_prefix_step.detach().cpu().tolist())
        initial_full.extend(initial_metrics["full_loss_per_sample"].detach().cpu().tolist())
        final_full.extend(final_full_step.detach().cpu().tolist())
        initial_kendall.extend(initial_metrics["kendall_per_sample"].detach().cpu().tolist())
        final_kendall.extend(final_kendall_step.detach().cpu().tolist())
        improved_prefix_count += int((final_prefix_step < initial_metrics["prefix_auc_per_sample"]).sum().item())
        improved_kendall_count += int((final_kendall_step > initial_metrics["kendall_per_sample"]).sum().item())

        for sample_idx in range(batch_size_local):
            print(f"[local_swap_eval] batch={batch_idx} sample={sample_idx}")
            for state in history:
                step_idx = int(state["step"])
                swap_idx = int(state["swap_idx"][sample_idx].item())
                prefix_auc = float(state["prefix_auc_per_sample"][sample_idx].item())
                full_loss = float(state["full_loss_per_sample"][sample_idx].item())
                kendall = float(state["kendall_per_sample"][sample_idx].item())
                improvement = float(state["improvement"][sample_idx].item())
                orders = [int(v) for v in state["orders"][sample_idx].detach().cpu().tolist()]
                if step_idx == 0:
                    print(
                        f"  step=0 init prefix_auc={prefix_auc:.6f} "
                        f"full_loss={full_loss:.6f} kendall={kendall:.6f} order={orders}"
                    )
                else:
                    print(
                        f"  step={step_idx} swap_idx={swap_idx} prefix_auc={prefix_auc:.6f} "
                        f"full_loss={full_loss:.6f} kendall={kendall:.6f} "
                        f"delta_quality={improvement:.6f} order={orders}"
                    )
                rows.append(
                    {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "step": step_idx,
                        "swap_idx": swap_idx,
                        "improved": int(state["improved_mask"][sample_idx].item()),
                        "prefix_auc": prefix_auc,
                        "full_loss": full_loss,
                        "kendall_tau_l2r": kendall,
                        "order": " ".join(str(v) for v in orders),
                    }
                )

    mean_prefix_improvement = float(np.mean(np.asarray(initial_prefix) - np.asarray(final_prefix)))
    mean_full_improvement = float(np.mean(np.asarray(initial_full) - np.asarray(final_full)))
    mean_kendall_improvement = float(np.mean(np.asarray(final_kendall) - np.asarray(initial_kendall)))

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_idx",
                "sample_idx",
                "step",
                "swap_idx",
                "improved",
                "prefix_auc",
                "full_loss",
                "kendall_tau_l2r",
                "order",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("[local_swap_eval] summary")
    print(f"  csv: {csv_path}")
    print(f"  samples: {total_samples}")
    print(f"  mean prefix_auc improvement: {mean_prefix_improvement:.6f}")
    print(f"  mean full_loss improvement: {mean_full_improvement:.6f}")
    print(f"  mean kendall improvement: {mean_kendall_improvement:.6f}")
    print(f"  prefix improved samples: {improved_prefix_count}/{total_samples}")
    print(f"  kendall improved samples: {improved_kendall_count}/{total_samples}")
    return {
        "csv_path": csv_path,
        "num_samples": total_samples,
        "mean_prefix_auc_improvement": mean_prefix_improvement,
        "mean_full_loss_improvement": mean_full_improvement,
        "mean_kendall_improvement": mean_kendall_improvement,
        "prefix_improved_samples": improved_prefix_count,
        "kendall_improved_samples": improved_kendall_count,
    }

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if train_stage == 'order_head':
            entropies = torch.zeros(eval_iters)
            taus = torch.zeros(eval_iters)
            prefix_means = torch.zeros(eval_iters)
            preference_losses = torch.zeros(eval_iters)
            preference_accs = torch.zeros(eval_iters)
            preference_gaps = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                _, _, _, hidden_states = _forward_policy_features(
                    raw_model,
                    X,
                    need_token_losses=False,
                    track_grad=False,
                )
                scores = _score_policy_head(raw_model, X, hidden_states, detach_inputs=True)
                preference_targets = _sample_order_preference_targets(raw_model, X)
                loss, pref_acc, _, _ = pairwise_order_preference_loss(
                    scores,
                    preference_targets["preferred_orders"],
                    preference_targets["other_orders"],
                    prefix_k=policy_prefix_k,
                    margin=preference_margin,
                )
                policy_block_orders = build_prefix_policy_block_orders(scores, policy_prefix_k)
                losses[k] = loss.item()
                entropies[k] = compute_order_entropy(scores, temperature=order_entropy_temperature).item()
                taus[k] = kendall_tau_to_l2r(policy_block_orders).item()
                prefix_means[k] = policy_block_orders[:, :max(1, min(policy_prefix_k, scores.size(1)))].float().mean().item()
                preference_losses[k] = loss.item()
                preference_accs[k] = pref_acc.item()
                preference_gaps[k] = preference_targets["quality_gap"].mean().item()
            out[split] = losses.mean().item()
            out[f"{split}_order_entropy"] = entropies.mean().item()
            out[f"{split}_kendall_tau"] = taus.mean().item()
            out[f"{split}_prefix_mean_index"] = prefix_means.mean().item()
            out[f"{split}_preference_loss"] = preference_losses.mean().item()
            out[f"{split}_preference_accuracy"] = preference_accs.mean().item()
            out[f"{split}_preference_gap"] = preference_gaps.mean().item()
        elif train_stage in ('policy_backbone', 'joint'):
            prefix_auc = torch.zeros(eval_iters)
            entropies = torch.zeros(eval_iters)
            taus = torch.zeros(eval_iters)
            preference_losses = torch.zeros(eval_iters) if train_stage == 'joint' else None
            preference_accs = torch.zeros(eval_iters) if train_stage == 'joint' else None
            preference_gaps = torch.zeros(eval_iters) if train_stage == 'joint' else None
            ao_losses = torch.zeros(eval_iters) if train_stage == 'joint' else None
            total_losses = torch.zeros(eval_iters) if train_stage == 'joint' else None
            for k in range(eval_iters):
                X, Y = get_batch(split)
                scores_for_policy, policy_block_orders, policy_token_orders, _ = _build_policy_orders_from_batch(raw_model, X, track_grad=False)
                with ctx:
                    _, ao_loss, token_losses = model(
                        X,
                        mode=None,
                        orders=policy_token_orders,
                        return_token_loss=True,
                    )
                block_losses = token_losses_to_block_losses(
                    token_losses,
                    block_len=block_order_block_len,
                )
                losses[k] = ao_loss.item()
                prefix_auc[k] = compute_prefix_auc(block_losses, policy_prefix_k).item()
                entropies[k] = compute_order_entropy(scores_for_policy, temperature=order_entropy_temperature).item()
                taus[k] = kendall_tau_to_l2r(policy_block_orders).item()
                if train_stage == 'joint':
                    _, _, _, hidden_states = _forward_policy_features(
                        raw_model,
                        X,
                        need_token_losses=False,
                        track_grad=False,
                    )
                    scores_for_pref = _score_policy_head(
                        raw_model,
                        X,
                        hidden_states,
                        detach_inputs=True,
                    )
                    preference_targets = _sample_order_preference_targets(raw_model, X)
                    pref_loss, pref_acc, _, _ = pairwise_order_preference_loss(
                        scores_for_pref,
                        preference_targets["preferred_orders"],
                        preference_targets["other_orders"],
                        prefix_k=policy_prefix_k,
                        margin=preference_margin,
                    )
                    preference_losses[k] = pref_loss.item()
                    preference_accs[k] = pref_acc.item()
                    preference_gaps[k] = preference_targets["quality_gap"].mean().item()
                    ao_losses[k] = ao_loss.item()
                    total_losses[k] = (ao_loss.item() + lambda_list * preference_losses[k].item())
            out[split] = losses.mean().item()
            out[f"{split}_prefix_auc"] = prefix_auc.mean().item()
            out[f"{split}_order_entropy"] = entropies.mean().item()
            out[f"{split}_kendall_tau"] = taus.mean().item()
            if preference_losses is not None:
                out[f"{split}_preference_loss"] = preference_losses.mean().item()
                out[f"{split}_preference_accuracy"] = preference_accs.mean().item()
                out[f"{split}_preference_gap"] = preference_gaps.mean().item()
                out[f"{split}_ao_loss"] = ao_losses.mean().item()
                out[f"{split}_total_loss"] = total_losses.mean().item()
        else:
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, mode=main_eval_mode)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
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


def build_block_score_table(avg_score_per_position):
    import wandb

    avg_scores = np.asarray(avg_score_per_position, dtype=np.float32)
    sorted_origin = np.argsort(-avg_scores)
    table = wandb.Table(columns=["rank", "origin_block_index", "avg_score"])
    for rank_idx, origin_pos in enumerate(sorted_origin, start=1):
        table.add_data(
            int(rank_idx),
            int(origin_pos),
            float(avg_scores[origin_pos]),
        )
    return table

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

if train_stage == 'local_swap_eval':
    local_swap_summary = run_local_swap_eval(raw_model)
    if wandb_log and master_process:
        wandb.log({
            "local_swap/mean_prefix_auc_improvement": local_swap_summary["mean_prefix_auc_improvement"],
            "local_swap/mean_full_loss_improvement": local_swap_summary["mean_full_loss_improvement"],
            "local_swap/mean_kendall_improvement": local_swap_summary["mean_kendall_improvement"],
            "local_swap/prefix_improved_samples": local_swap_summary["prefix_improved_samples"],
            "local_swap/kendall_improved_samples": local_swap_summary["kendall_improved_samples"],
        })
    if ddp:
        destroy_process_group()
    sys.exit(0)

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * float(param_group.get('lr_scale', 1.0))

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if train_stage == 'order_head':
            print(
                f"  order entropy: train {losses['train_order_entropy']:.4f}, "
                f"val {losses['val_order_entropy']:.4f}"
            )
            print(
                f"  kendall tau to l2r: train {losses['train_kendall_tau']:.4f}, "
                f"val {losses['val_kendall_tau']:.4f}"
            )
            print(
                f"  preference loss/acc/gap: train {losses['train_preference_loss']:.4f}/{losses['train_preference_accuracy']:.4f}/{losses['train_preference_gap']:.4f}, "
                f"val {losses['val_preference_loss']:.4f}/{losses['val_preference_accuracy']:.4f}/{losses['val_preference_gap']:.4f}"
            )
        elif train_stage in ('policy_backbone', 'joint'):
            print(
                f"  prefix_auc: train {losses['train_prefix_auc']:.4f}, "
                f"val {losses['val_prefix_auc']:.4f}"
            )
            print(
                f"  kendall tau to l2r: train {losses['train_kendall_tau']:.4f}, "
                f"val {losses['val_kendall_tau']:.4f}"
            )
            if train_stage == 'joint':
                print(
                    f"  ao/pref/total: train {losses['train_ao_loss']:.4f}/{losses['train_preference_loss']:.4f}/{losses['train_total_loss']:.4f}, "
                    f"val {losses['val_ao_loss']:.4f}/{losses['val_preference_loss']:.4f}/{losses['val_total_loss']:.4f}"
                )
        if wandb_log:
            log_payload = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            }
            if train_stage == 'order_head':
                log_payload.update({
                    "train/order_entropy": losses["train_order_entropy"],
                    "val/order_entropy": losses["val_order_entropy"],
                    "train/kendall_tau_l2r": losses["train_kendall_tau"],
                    "val/kendall_tau_l2r": losses["val_kendall_tau"],
                    "train/prefix_mean_index": losses["train_prefix_mean_index"],
                    "val/prefix_mean_index": losses["val_prefix_mean_index"],
                    "train/preference_loss": losses["train_preference_loss"],
                    "val/preference_loss": losses["val_preference_loss"],
                    "train/preference_accuracy": losses["train_preference_accuracy"],
                    "val/preference_accuracy": losses["val_preference_accuracy"],
                    "train/preference_gap": losses["train_preference_gap"],
                    "val/preference_gap": losses["val_preference_gap"],
                })
                if latest_policy_train_stats is not None and "avg_score_per_position" in latest_policy_train_stats:
                    log_payload["order_head/origin_score_table"] = build_block_score_table(
                        latest_policy_train_stats["avg_score_per_position"]
                    )
            if train_stage in ('policy_backbone', 'joint'):
                log_payload.update({
                    "train/prefix_auc": losses["train_prefix_auc"],
                    "val/prefix_auc": losses["val_prefix_auc"],
                    "train/order_entropy": losses["train_order_entropy"],
                    "val/order_entropy": losses["val_order_entropy"],
                    "train/kendall_tau_l2r": losses["train_kendall_tau"],
                    "val/kendall_tau_l2r": losses["val_kendall_tau"],
                })
                if train_stage == 'joint':
                    log_payload["train/preference_loss"] = losses["train_preference_loss"]
                    log_payload["val/preference_loss"] = losses["val_preference_loss"]
                    log_payload["train/preference_accuracy"] = losses["train_preference_accuracy"]
                    log_payload["val/preference_accuracy"] = losses["val_preference_accuracy"]
                    log_payload["train/preference_gap"] = losses["train_preference_gap"]
                    log_payload["val/preference_gap"] = losses["val_preference_gap"]
                    log_payload["train/ao_loss"] = losses["train_ao_loss"]
                    log_payload["val/ao_loss"] = losses["val_ao_loss"]
                    log_payload["train/total_loss"] = losses["train_total_loss"]
                    log_payload["val/total_loss"] = losses["val_total_loss"]
            if latest_policy_train_stats is not None:
                log_payload.update({
                    "policy/order_entropy": latest_policy_train_stats["order_entropy"],
                    "policy/kendall_tau_l2r": latest_policy_train_stats["kendall_tau"],
                    "policy/prefix_mean_index": latest_policy_train_stats["prefix_mean_index"],
                    "policy/prefix_std_index": latest_policy_train_stats["prefix_std_index"],
                    "policy/prefix_min_index": latest_policy_train_stats["prefix_min_index"],
                    "policy/prefix_max_index": latest_policy_train_stats["prefix_max_index"],
                })
                if "prefix_auc" in latest_policy_train_stats:
                    log_payload["policy/prefix_auc"] = latest_policy_train_stats["prefix_auc"]
                if "preference_loss" in latest_policy_train_stats:
                    log_payload["policy/preference_loss"] = latest_policy_train_stats["preference_loss"]
                if "preference_accuracy" in latest_policy_train_stats:
                    log_payload["policy/preference_accuracy"] = latest_policy_train_stats["preference_accuracy"]
                if "preference_gap" in latest_policy_train_stats:
                    log_payload["policy/preference_gap"] = latest_policy_train_stats["preference_gap"]
                if "ao_loss" in latest_policy_train_stats:
                    log_payload["policy/ao_loss"] = latest_policy_train_stats["ao_loss"]
                if "total_loss" in latest_policy_train_stats:
                    log_payload["policy/total_loss"] = latest_policy_train_stats["total_loss"]
                if "token_prefix_auc" in latest_policy_train_stats:
                    log_payload["policy/token_prefix_auc"] = latest_policy_train_stats["token_prefix_auc"]
            if train_stage == 'joint' and latest_policy_train_stats is not None:
                log_payload.update({
                    "joint/ao_loss": latest_policy_train_stats["ao_loss"],
                    "joint/preference_loss": latest_policy_train_stats["preference_loss"],
                    "joint/total_loss": latest_policy_train_stats["total_loss"],
                    "joint/order_entropy": latest_policy_train_stats["order_entropy"],
                    "joint/prefix_mean_index": latest_policy_train_stats["prefix_mean_index"],
                    "joint/kendall_to_l2r": latest_policy_train_stats["kendall_tau"],
                })
                if "prefix_auc" in latest_policy_train_stats:
                    log_payload["joint/prefix_auc"] = latest_policy_train_stats["prefix_auc"]
            log_payload["baseline/step_mean_avg"] = float(step_mean_baseline.mean().item())
            log_payload["baseline/step_var_avg"] = float(step_var_baseline.mean().item())
            for step_idx in range(num_blocks):
                log_payload[f"baseline/step_mean_{step_idx:02d}"] = float(step_mean_baseline[step_idx].item())
                log_payload[f"baseline/step_var_{step_idx:02d}"] = float(step_var_baseline[step_idx].item())
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
                    'block_order_baselines': {
                        'step_mean': step_mean_baseline.detach().cpu().tolist(),
                        'step_var': step_var_baseline.detach().cpu().tolist(),
                        'initialized': baseline_initialized,
                    },
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
            if train_stage == 'order_head':
                _, _, _, hidden_states = _forward_policy_features(
                    raw_model,
                    X,
                    need_token_losses=False,
                    track_grad=False,
                )
                scores = _score_policy_head(raw_model, X, hidden_states, detach_inputs=True)
                preference_targets = _sample_order_preference_targets(raw_model, X)
                logits = None
                loss, pref_acc, _, _ = pairwise_order_preference_loss(
                    scores,
                    preference_targets["preferred_orders"],
                    preference_targets["other_orders"],
                    prefix_k=policy_prefix_k,
                    margin=preference_margin,
                )
                latest_policy_train_stats = _policy_stats(
                    scores.detach(),
                    build_prefix_policy_block_orders(scores.detach(), policy_prefix_k),
                )
                latest_policy_train_stats["avg_score_per_position"] = [
                    float(v) for v in scores.detach().mean(dim=0).cpu().tolist()
                ]
                latest_policy_train_stats["preference_loss"] = float(loss.detach().item())
                latest_policy_train_stats["preference_accuracy"] = float(pref_acc.detach().item())
                latest_policy_train_stats["preference_gap"] = float(preference_targets["quality_gap"].mean().item())
            elif train_stage == 'policy_backbone':
                scores, policy_block_orders, policy_token_orders, _ = _build_policy_orders_from_batch(
                    raw_model,
                    X,
                    track_grad=False,
                )
                logits, loss, token_losses = model(
                    X,
                    mode=None,
                    orders=policy_token_orders,
                    return_token_loss=True,
                )
                block_losses = token_losses_to_block_losses(
                    token_losses,
                    block_len=block_order_block_len,
                )
                latest_policy_train_stats = _policy_stats(
                    scores.detach(),
                    policy_block_orders.detach(),
                    block_losses=block_losses,
                    token_losses=token_losses,
                )
                latest_policy_train_stats["ao_loss"] = float(loss.detach().item())
            elif train_stage == 'joint':
                # Weak joint training with preference-style supervision for the order head.
                # Part A trains the order head to prefer better sampled orders.
                # Part B trains the backbone under the discrete prefix policy rollout.
                _, _, _, hidden_states_for_list = _forward_policy_features(
                    raw_model,
                    X,
                    need_token_losses=False,
                    track_grad=True,
                )
                scores_for_list = _score_policy_head(
                    raw_model,
                    X,
                    hidden_states_for_list,
                    detach_inputs=False,
                )
                preference_targets = _sample_order_preference_targets(raw_model, X)
                preference_loss, pref_acc, _, _ = pairwise_order_preference_loss(
                    scores_for_list,
                    preference_targets["preferred_orders"],
                    preference_targets["other_orders"],
                    prefix_k=policy_prefix_k,
                    margin=preference_margin,
                )

                # Separate forward path for the learned block policy rollout.
                # The hard block order is explicitly detached before expansion into token orders,
                # which makes the weak-joint semantics unambiguous.
                _, _, _, hidden_states_for_policy = _forward_policy_features(
                    raw_model,
                    X,
                    need_token_losses=False,
                    track_grad=True,
                )
                scores_for_policy = _score_policy_head(
                    raw_model,
                    X,
                    hidden_states_for_policy,
                    detach_inputs=False,
                )
                scores_for_policy_detached = scores_for_policy.detach()
                policy_block_orders = build_prefix_policy_block_orders(
                    scores_for_policy_detached,
                    policy_prefix_k,
                )
                policy_token_orders = expand_block_orders_to_token_orders(
                    policy_block_orders,
                    block_len=block_order_block_len,
                )
                logits, ao_loss, token_losses = model(
                    X,
                    mode=None,
                    orders=policy_token_orders,
                    return_token_loss=True,
                )
                policy_block_losses = token_losses_to_block_losses(
                    token_losses,
                    block_len=block_order_block_len,
                )
                total_loss = ao_loss + lambda_list * preference_loss
                loss = total_loss
                latest_policy_train_stats = _policy_stats(
                    scores_for_policy_detached,
                    policy_block_orders.detach(),
                    block_losses=policy_block_losses,
                    token_losses=token_losses,
                )
                latest_policy_train_stats["preference_loss"] = float(preference_loss.detach().item())
                latest_policy_train_stats["preference_accuracy"] = float(pref_acc.detach().item())
                latest_policy_train_stats["preference_gap"] = float(preference_targets["quality_gap"].mean().item())
                latest_policy_train_stats["ao_loss"] = float(ao_loss.detach().item())
                latest_policy_train_stats["total_loss"] = float(total_loss.detach().item())
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
