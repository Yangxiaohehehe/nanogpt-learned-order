from contextlib import nullcontext

import torch
from torch.nn import functional as F


def _validate_block_layout(length, block_len):
    if length % block_len != 0:
        raise ValueError(f"Length {length} is not divisible by block_len={block_len}.")
    return length // block_len


def sample_random_block_orders(batch_size, num_blocks, device, generator=None):
    return torch.stack(
        [torch.randperm(num_blocks, generator=generator, device=device) for _ in range(batch_size)]
    )


def build_ascending_block_orders(batch_size, num_blocks, device):
    return torch.arange(num_blocks, device=device).unsqueeze(0).expand(batch_size, -1)


def expand_block_orders_to_token_orders(block_orders, block_len=16):
    """
    Expand block orders into token orders while keeping l2r order within each block.

    Args:
        block_orders: (B, num_blocks)
    Returns:
        token_orders: (B, num_blocks * block_len)
    """
    batch_size, num_blocks = block_orders.shape
    block_offsets = torch.arange(block_len, device=block_orders.device).view(1, 1, block_len)
    token_orders = block_orders.unsqueeze(-1) * block_len + block_offsets
    token_orders = token_orders.reshape(batch_size, num_blocks * block_len)
    expected = torch.arange(num_blocks * block_len, device=block_orders.device).unsqueeze(0).expand(batch_size, -1)
    if not torch.equal(torch.sort(token_orders, dim=-1).values, expected):
        raise ValueError("expand_block_orders_to_token_orders produced an invalid token permutation")
    return token_orders


def token_losses_to_block_losses(token_losses, block_len=16):
    """
    Aggregate token losses into block losses along the reveal trajectory.

    Args:
        token_losses: (B, T) ordered by reveal token step.
    Returns:
        block_losses: (B, num_blocks) ordered by reveal block step.
    """
    batch_size, seq_len = token_losses.shape
    num_blocks = _validate_block_layout(seq_len, block_len)
    return token_losses.float().view(batch_size, num_blocks, block_len).mean(dim=-1)


def compute_future_block_stats(block_losses, horizon=4):
    """
    Compute future-window mean and variance at each reveal step.

    Args:
        block_losses: (B, num_blocks) ordered by reveal block step.
    Returns:
        future_means: (B, num_blocks)
        future_vars: (B, num_blocks)
    """
    batch_size, num_blocks = block_losses.shape
    future_means = torch.zeros_like(block_losses, dtype=torch.float32)
    future_vars = torch.zeros_like(block_losses, dtype=torch.float32)
    float_losses = block_losses.float()

    for step in range(num_blocks):
        window_end = min(num_blocks, step + int(horizon))
        window = float_losses[:, step:window_end]
        future_means[:, step] = window.mean(dim=-1)
        future_vars[:, step] = window.var(dim=-1, unbiased=False)
    return future_means, future_vars


def update_stepwise_ema_baseline(baseline, current_batch_stat, momentum=0.95, initialized=False):
    """
    EMA update for step-wise random-order baselines.
    """
    if not initialized:
        return current_batch_stat.detach().clone()
    return momentum * baseline + (1.0 - momentum) * current_batch_stat.detach()


def compute_block_residual_utilities(
    block_losses,
    step_mean_baseline,
    step_var_baseline,
    horizon=4,
    alpha=1.0,
    beta=1.0,
):
    """
    Build reveal-step utilities from de-trended future mean/variance statistics.

    Returns:
        step_utilities: (B, num_blocks)
        future_means: (B, num_blocks)
        future_vars: (B, num_blocks)
        residual_means: (B, num_blocks)
        residual_vars: (B, num_blocks)
    """
    future_means, future_vars = compute_future_block_stats(block_losses, horizon=horizon)
    residual_means = future_means - step_mean_baseline.view(1, -1)
    residual_vars = future_vars - step_var_baseline.view(1, -1)
    step_utilities = -float(alpha) * residual_means - float(beta) * residual_vars
    return step_utilities, future_means, future_vars, residual_means, residual_vars


def scatter_block_utilities_to_original_positions(step_utilities, block_orders):
    """
    Scatter reveal-step utilities back to original block indices.
    """
    batch_size, num_blocks = step_utilities.shape
    original = torch.zeros_like(step_utilities, dtype=torch.float32)
    original.scatter_(1, block_orders, step_utilities.float())
    expected = torch.arange(num_blocks, device=block_orders.device).unsqueeze(0).expand(batch_size, -1)
    if not torch.equal(torch.sort(block_orders, dim=-1).values, expected):
        raise ValueError("block_orders is not a valid block permutation")
    return original


def listwise_block_order_loss(scores, block_utilities, temperature=1.0, eps=1e-8):
    """
    Block-level listwise ranking loss driven by residual utilities, not l2r labels.
    """
    scaled_scores = scores / max(float(temperature), eps)
    scaled_utilities = block_utilities.detach() / max(float(temperature), eps)
    log_p = F.log_softmax(scaled_scores, dim=-1)
    q = F.softmax(scaled_utilities, dim=-1)
    return -(q * log_p).sum(dim=-1).mean()


def build_prefix_policy_block_orders(scores, k, generator=None):
    """
    Build block reveal orders from top-k scored prefix + random suffix.
    """
    batch_size, num_blocks = scores.shape
    k = max(0, min(int(k), num_blocks))
    _, sorted_idx = torch.sort(scores, dim=-1, descending=True, stable=True)
    prefix = sorted_idx[:, :k]

    random_keys = torch.rand(batch_size, num_blocks, device=scores.device, generator=generator)
    random_perm = torch.argsort(random_keys, dim=-1)
    prefix_mask = torch.zeros(batch_size, num_blocks, dtype=torch.bool, device=scores.device)
    if k > 0:
        prefix_mask.scatter_(1, prefix, True)
    suffix_mask = ~prefix_mask.gather(1, random_perm)
    suffix = random_perm[suffix_mask].view(batch_size, num_blocks - k)
    block_orders = torch.cat([prefix, suffix], dim=-1)

    expected = torch.arange(num_blocks, device=scores.device).unsqueeze(0).expand(batch_size, -1)
    if not torch.equal(torch.sort(block_orders, dim=-1).values, expected):
        raise ValueError("build_prefix_policy_block_orders produced an invalid block permutation")
    return block_orders


def compute_prefix_auc(step_losses, k):
    k = max(1, min(int(k), step_losses.size(1)))
    return step_losses[:, :k].mean()


def compute_order_entropy(scores, temperature=1.0, eps=1e-8):
    probs = F.softmax(scores / max(float(temperature), eps), dim=-1)
    return -(probs * (probs.clamp_min(eps).log())).sum(dim=-1).mean()


def kendall_tau_to_l2r(orders):
    batch_size, seq_len = orders.shape
    if seq_len < 2:
        return orders.new_tensor(1.0, dtype=torch.float32)

    taus = []
    total_pairs = seq_len * (seq_len - 1) / 2.0
    for row in orders.detach().cpu():
        inversions = 0
        values = row.tolist()
        for i in range(seq_len):
            vi = values[i]
            for j in range(i + 1, seq_len):
                if vi > values[j]:
                    inversions += 1
        taus.append(1.0 - (2.0 * inversions / total_pairs))
    return torch.tensor(taus, dtype=torch.float32).mean()


def prefix_position_stats(orders, k):
    k = max(1, min(int(k), orders.size(1)))
    prefix = orders[:, :k].float()
    return {
        "prefix_mean_index": prefix.mean(),
        "prefix_std_index": prefix.std(unbiased=False),
        "prefix_min_index": prefix.min(),
        "prefix_max_index": prefix.max(),
    }


def kendall_tau_to_l2r_per_sample(orders):
    batch_size, seq_len = orders.shape
    if seq_len < 2:
        return torch.ones(batch_size, dtype=torch.float32, device=orders.device)

    taus = []
    total_pairs = seq_len * (seq_len - 1) / 2.0
    for row in orders.detach().cpu():
        inversions = 0
        values = row.tolist()
        for i in range(seq_len):
            vi = values[i]
            for j in range(i + 1, seq_len):
                if vi > values[j]:
                    inversions += 1
        taus.append(1.0 - (2.0 * inversions / total_pairs))
    return torch.tensor(taus, dtype=torch.float32, device=orders.device)


def generate_adjacent_swap_candidates(block_orders):
    """
    Enumerate all adjacent-swap block-order candidates.

    Args:
        block_orders: (B, num_blocks)
    Returns:
        candidates: (B, num_blocks - 1, num_blocks)
    """
    batch_size, num_blocks = block_orders.shape
    if num_blocks < 2:
        return block_orders.unsqueeze(1)

    candidates = block_orders.unsqueeze(1).expand(batch_size, num_blocks - 1, num_blocks).clone()
    swap_idx = torch.arange(num_blocks - 1, device=block_orders.device)
    left_vals = candidates[:, swap_idx, swap_idx].clone()
    right_vals = candidates[:, swap_idx, swap_idx + 1].clone()
    candidates[:, swap_idx, swap_idx] = right_vals
    candidates[:, swap_idx, swap_idx + 1] = left_vals
    return candidates


@torch.no_grad()
def evaluate_block_order_quality(model, idx, block_orders, prefix_k=2, block_len=16, autocast_context=None):
    """
    Evaluate a block order using the current AO-GPT checkpoint.

    This is a local-search diagnostic utility, not a training signal and not
    an l2r supervision mechanism.
    """
    if autocast_context is None:
        autocast_context = nullcontext()

    token_orders = expand_block_orders_to_token_orders(block_orders, block_len=block_len)
    with autocast_context:
        _, _, token_losses = model(
            idx,
            mode=None,
            orders=token_orders,
            return_token_loss=True,
        )
    block_losses = token_losses_to_block_losses(token_losses, block_len=block_len)
    prefix_k = max(1, min(int(prefix_k), block_losses.size(1)))
    prefix_auc_per_sample = block_losses[:, :prefix_k].mean(dim=-1)
    full_loss_per_sample = token_losses.float().mean(dim=-1)
    kendall_per_sample = kendall_tau_to_l2r_per_sample(block_orders)
    return {
        "block_losses": block_losses,
        "token_losses": token_losses,
        "prefix_auc_per_sample": prefix_auc_per_sample,
        "full_loss_per_sample": full_loss_per_sample,
        "kendall_per_sample": kendall_per_sample,
        "prefix_auc": float(prefix_auc_per_sample.mean().item()),
        "full_loss": float(full_loss_per_sample.mean().item()),
        "kendall_tau": float(kendall_per_sample.mean().item()),
    }


@torch.no_grad()
def greedy_adjacent_swap_step(model, idx, block_orders, prefix_k=2, block_len=16, autocast_context=None):
    """
    One greedy local-search step over all adjacent swaps.

    Quality is defined as -prefix_auc_k, so lower prefix AUC is better.
    """
    current_metrics = evaluate_block_order_quality(
        model,
        idx,
        block_orders,
        prefix_k=prefix_k,
        block_len=block_len,
        autocast_context=autocast_context,
    )
    candidates = generate_adjacent_swap_candidates(block_orders)
    batch_size, num_candidates, num_blocks = candidates.shape
    flat_candidates = candidates.reshape(batch_size * num_candidates, num_blocks)
    flat_idx = idx.unsqueeze(1).expand(batch_size, num_candidates, idx.size(1)).reshape(
        batch_size * num_candidates,
        idx.size(1),
    )
    candidate_metrics = evaluate_block_order_quality(
        model,
        flat_idx,
        flat_candidates,
        prefix_k=prefix_k,
        block_len=block_len,
        autocast_context=autocast_context,
    )
    candidate_prefix = candidate_metrics["prefix_auc_per_sample"].view(batch_size, num_candidates)
    candidate_full = candidate_metrics["full_loss_per_sample"].view(batch_size, num_candidates)
    candidate_kendall = candidate_metrics["kendall_per_sample"].view(batch_size, num_candidates)
    current_quality = -current_metrics["prefix_auc_per_sample"]
    candidate_quality = -candidate_prefix
    best_candidate_idx = candidate_quality.argmax(dim=-1)
    best_quality = candidate_quality.gather(1, best_candidate_idx.unsqueeze(-1)).squeeze(-1)
    improvement = best_quality - current_quality
    improved_mask = improvement > 0

    batch_indices = torch.arange(batch_size, device=block_orders.device)
    best_orders = candidates[batch_indices, best_candidate_idx]
    best_metrics = {
        "prefix_auc_per_sample": candidate_prefix[batch_indices, best_candidate_idx],
        "full_loss_per_sample": candidate_full[batch_indices, best_candidate_idx],
        "kendall_per_sample": candidate_kendall[batch_indices, best_candidate_idx],
    }
    best_metrics["prefix_auc"] = float(best_metrics["prefix_auc_per_sample"].mean().item())
    best_metrics["full_loss"] = float(best_metrics["full_loss_per_sample"].mean().item())
    best_metrics["kendall_tau"] = float(best_metrics["kendall_per_sample"].mean().item())

    return {
        "best_orders": best_orders,
        "best_swap_idx": best_candidate_idx,
        "current_metrics": current_metrics,
        "best_metrics": best_metrics,
        "improvement": improvement,
        "improved_mask": improved_mask,
    }


@torch.no_grad()
def greedy_adjacent_swap_search(
    model,
    idx,
    init_block_orders,
    num_steps=3,
    prefix_k=2,
    block_len=16,
    autocast_context=None,
):
    """
    Greedy adjacent-swap local search from an initial block order.

    This is an evaluation experiment only; it does not provide supervision.
    """
    history = []
    current_orders = init_block_orders.clone()
    current_metrics = evaluate_block_order_quality(
        model,
        idx,
        current_orders,
        prefix_k=prefix_k,
        block_len=block_len,
        autocast_context=autocast_context,
    )
    history.append(
        {
            "step": 0,
            "orders": current_orders.clone(),
            "prefix_auc_per_sample": current_metrics["prefix_auc_per_sample"].clone(),
            "full_loss_per_sample": current_metrics["full_loss_per_sample"].clone(),
            "kendall_per_sample": current_metrics["kendall_per_sample"].clone(),
            "swap_idx": torch.full(
                (current_orders.size(0),),
                -1,
                dtype=torch.long,
                device=current_orders.device,
            ),
            "improvement": torch.zeros(current_orders.size(0), dtype=torch.float32, device=current_orders.device),
            "improved_mask": torch.zeros(current_orders.size(0), dtype=torch.bool, device=current_orders.device),
        }
    )

    for step_idx in range(1, int(num_steps) + 1):
        step_result = greedy_adjacent_swap_step(
            model,
            idx,
            current_orders,
            prefix_k=prefix_k,
            block_len=block_len,
            autocast_context=autocast_context,
        )
        improved_mask = step_result["improved_mask"]
        current_orders = torch.where(
            improved_mask.unsqueeze(-1),
            step_result["best_orders"],
            current_orders,
        )
        current_metrics = evaluate_block_order_quality(
            model,
            idx,
            current_orders,
            prefix_k=prefix_k,
            block_len=block_len,
            autocast_context=autocast_context,
        )
        history.append(
            {
                "step": step_idx,
                "orders": current_orders.clone(),
                "prefix_auc_per_sample": current_metrics["prefix_auc_per_sample"].clone(),
                "full_loss_per_sample": current_metrics["full_loss_per_sample"].clone(),
                "kendall_per_sample": current_metrics["kendall_per_sample"].clone(),
                "swap_idx": torch.where(
                    improved_mask,
                    step_result["best_swap_idx"],
                    torch.full_like(step_result["best_swap_idx"], -1),
                ),
                "improvement": torch.where(
                    improved_mask,
                    step_result["improvement"],
                    torch.zeros_like(step_result["improvement"]),
                ),
                "improved_mask": improved_mask.clone(),
            }
        )
        if not improved_mask.any():
            break

    return {
        "history": history,
        "final_orders": history[-1]["orders"],
        "final_prefix_auc_per_sample": history[-1]["prefix_auc_per_sample"],
        "final_full_loss_per_sample": history[-1]["full_loss_per_sample"],
        "final_kendall_per_sample": history[-1]["kendall_per_sample"],
    }


# Deprecated token-level helpers kept only so older imports fail less abruptly.
def compute_token_utilities(*args, **kwargs):
    raise RuntimeError("compute_token_utilities is deprecated; use block-level residual utilities instead.")


def listwise_order_loss(*args, **kwargs):
    raise RuntimeError("listwise_order_loss is deprecated; use listwise_block_order_loss instead.")


def build_prefix_policy_orders(*args, **kwargs):
    raise RuntimeError("build_prefix_policy_orders is deprecated; use build_prefix_policy_block_orders instead.")
