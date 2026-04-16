from contextlib import nullcontext

import torch


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


def build_fixed_block_permutation(num_blocks, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(num_blocks, generator=generator, device="cpu")


def invert_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def block_permutation_to_token_permutation(block_perm, block_len):
    token_orders = expand_block_orders_to_token_orders(
        block_perm.view(1, -1).to(dtype=torch.long),
        block_len=block_len,
    )
    return token_orders.squeeze(0).to(device="cpu", dtype=torch.long)


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


def compute_prefix_auc(step_losses, k):
    k = max(1, min(int(k), step_losses.size(1)))
    return step_losses[:, :k].mean()


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
