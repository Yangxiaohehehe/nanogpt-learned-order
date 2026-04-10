import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    evaluate_block_order_quality,
    invert_permutation,
)


def load_runner_config_from_argv(argv):
    config_ns = {}
    filtered = []
    config_path = None
    for arg in argv:
        if "=" not in arg and not arg.startswith("--") and config_path is None:
            config_path = Path(arg)
        else:
            filtered.append(arg)
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as handle:
            exec(handle.read(), config_ns)
    return config_ns, filtered


def parse_csv_list(raw_value, cast_fn=str):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [cast_fn(item) for item in values]


def parse_args():
    config_ns, filtered_argv = load_runner_config_from_argv(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description=(
            "Proposal benchmark on frozen checkpoints. "
            "Generate candidate predecessor->target edges with different proposal methods, "
            "then validate them with the existing pair2 area+tv logic."
        )
    )
    parser.add_argument("--ckpt", action="append", default=[], help="Checkpoint path. Can be provided multiple times.")
    parser.add_argument("--dataset", type=str, default=config_ns.get("dataset", None))
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default=str(config_ns.get("split", "val")), choices=["train", "val"])
    parser.add_argument(
        "--proposal",
        type=str,
        default=str(config_ns.get("proposal", "random")),
        help="Comma-separated list from: full_enum, random, attn_early, loo_early",
    )
    parser.add_argument("--num-targets", type=int, default=int(config_ns.get("num_targets", 0)))
    parser.add_argument(
        "--num-candidates-per-target",
        type=int,
        default=int(config_ns.get("num_candidates_per_target", 8)),
    )
    parser.add_argument("--proposal-num-batches", type=int, default=int(config_ns.get("proposal_num_batches", 4)))
    parser.add_argument("--proposal-batch-size", type=int, default=int(config_ns.get("proposal_batch_size", 16)))
    parser.add_argument("--pair-mining-batches", type=int, default=int(config_ns.get("pair_mining_batches", 50)))
    parser.add_argument("--pair-eval-batch-size", type=int, default=int(config_ns.get("pair_eval_batch_size", 64)))
    parser.add_argument("--candidate-eval-batch-size", type=int, default=int(config_ns.get("candidate_eval_batch_size", 64)))
    parser.add_argument("--pair-score-k", type=int, default=int(config_ns.get("pair_score_k", 2)))
    parser.add_argument("--tv-weight", type=float, default=float(config_ns.get("tv_weight", 0.3)))
    parser.add_argument("--segment-len", type=int, default=int(config_ns.get("segment_len", 4)))
    parser.add_argument("--probe-prefix-len", type=int, default=int(config_ns.get("probe_prefix_len", 6)))
    parser.add_argument("--target-probe-index", type=int, default=int(config_ns.get("target_probe_index", 3)))
    parser.add_argument("--seed", type=int, default=int(config_ns.get("seed", 12345)))
    parser.add_argument("--out_dir", type=Path, default=Path(config_ns.get("out_dir", "Report/trajectory/proposal_benchmark")))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["float32", "float16", "bfloat16"],
    )
    args = parser.parse_args(filtered_argv)
    if not args.ckpt:
        raise ValueError("At least one --ckpt must be provided.")
    if not (0 < int(args.target_probe_index) < int(args.probe_prefix_len)):
        raise ValueError("Require 0 < target_probe_index < probe_prefix_len.")
    return args


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in device or dtype == "float32":
        return nullcontext()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint(ckpt_path: Path, device: str):
    return torch.load(ckpt_path, map_location=device)


def build_model(checkpoint, device: str):
    model = AOGPT(AOGPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def resolve_data_dir(dataset_arg, data_dir_arg, checkpoint):
    if data_dir_arg is not None:
        return data_dir_arg
    dataset = dataset_arg or checkpoint.get("config", {}).get("dataset")
    if dataset is None:
        raise ValueError("Could not infer dataset. Pass config, --dataset, or --data_dir.")
    return REPO_ROOT / "data" / dataset


def load_tokens(data_dir: Path, split: str):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def load_block_permutation_from_checkpoint(checkpoint, num_blocks, block_len):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    perm_state = checkpoint.get("data_permutation")
    if perm_state is not None and perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(block_perm, block_len=block_len)
    return {
        "permute_mode": "block",
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str, token_perm=None):
    max_start = len(tokens) - block_size
    starts = rng.integers(0, max_start, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    ).to(device)
    if token_perm is not None:
        batch = batch[:, token_perm.to(device)]
    return batch


def build_random_suffix_orders_for_pairs(pairs, batch_size, num_blocks, device, generator):
    blocks = torch.arange(num_blocks, device=device)
    orders = []
    for first, second in pairs:
        local = []
        mask = (blocks != first) & (blocks != second)
        remaining = blocks[mask]
        for _ in range(batch_size):
            perm = torch.randperm(remaining.numel(), generator=generator, device=device)
            local.append(torch.cat([torch.tensor([first, second], device=device), remaining[perm]], dim=0))
        orders.append(torch.stack(local, dim=0))
    return torch.stack(orders, dim=1)


def compute_window_area_plus_tv(block_losses, window_k, tv_weight):
    window_k = max(1, min(int(window_k), block_losses.size(1)))
    window = block_losses[:, :window_k].float()
    area = -window.mean(dim=-1)
    if window.size(1) < 2:
        total_variation = torch.zeros_like(area)
    else:
        total_variation = -(window[:, 1:] - window[:, :-1]).abs().sum(dim=-1)
    return area + float(tv_weight) * total_variation


def validate_candidate_pairs(
    model,
    tokens,
    candidate_pairs,
    batch_size,
    num_batches,
    pair_score_k,
    tv_weight,
    device,
    autocast_context,
    seed,
    eval_batch_size,
    token_perm=None,
):
    num_blocks = model.num_blocks
    score_sums = torch.zeros(len(candidate_pairs), dtype=torch.float64)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cuda" if "cuda" in device else "cpu")
    generator.manual_seed(seed)
    chunk_size = max(1, int(eval_batch_size))

    for _ in range(int(num_batches)):
        idx = sample_batch(tokens, batch_size, model.config.block_size, rng, device, token_perm=token_perm)
        for start in range(0, len(candidate_pairs), chunk_size):
            chunk_pairs = candidate_pairs[start : start + chunk_size]
            pair_orders = build_random_suffix_orders_for_pairs(
                chunk_pairs,
                batch_size=idx.size(0),
                num_blocks=num_blocks,
                device=device,
                generator=generator,
            )
            flat_orders = pair_orders.reshape(idx.size(0) * len(chunk_pairs), num_blocks)
            flat_idx = idx.unsqueeze(1).expand(idx.size(0), len(chunk_pairs), idx.size(1)).reshape(
                idx.size(0) * len(chunk_pairs),
                idx.size(1),
            )
            metrics = evaluate_block_order_quality(
                model,
                flat_idx,
                flat_orders,
                prefix_k=max(2, min(int(pair_score_k), num_blocks)),
                block_len=model.block_order_block_len,
                autocast_context=autocast_context,
            )
            local = compute_window_area_plus_tv(metrics["block_losses"], window_k=pair_score_k, tv_weight=tv_weight)
            local = local.view(idx.size(0), len(chunk_pairs)).mean(dim=0).cpu().double()
            score_sums[start:start + len(chunk_pairs)] += local

    score_means = score_sums / max(1, int(num_batches))
    score_map = {
        (int(first), int(second)): float(score_means[pair_idx].item())
        for pair_idx, (first, second) in enumerate(candidate_pairs)
    }
    validated = []
    for first, second in candidate_pairs:
        reverse = score_map.get((int(second), int(first)), float("nan"))
        margin = float(score_map[(int(first), int(second))] - reverse) if not math.isnan(reverse) else float("nan")
        validated.append(
            {
                "first": int(first),
                "second": int(second),
                "score": float(score_map[(int(first), int(second))]),
                "reverse_score": None if math.isnan(reverse) else float(reverse),
                "reverse_margin": margin,
            }
        )
    validated.sort(key=lambda row: (row["score"], row["reverse_margin"]), reverse=True)
    return validated


def aggregate_top_pairs_to_segments(top_pairs, num_blocks, top_k, max_segment_len):
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

    for row in top_pairs[:top_k]:
        first = int(row["first"])
        second = int(row["second"])
        if first == second:
            continue
        if first in next_map or second in prev_map:
            continue
        if trace_start(first) == trace_start(second):
            continue
        seg_first = build_segment(trace_start(first))
        seg_second = build_segment(trace_start(second))
        if len(seg_first) + len(seg_second) > max_segment_len:
            continue
        next_map[first] = second
        prev_map[second] = first

    starts = [node for node in range(num_blocks) if node not in prev_map and node in next_map]
    segments = []
    covered = set()
    for start in starts:
        segment = build_segment(start)
        if len(segment) >= 2:
            segments.append(segment)
            covered.update(segment)
    segments.sort(key=lambda values: (-len(values), values[0], values[-1]))
    payload = [
        {
            "segment": [int(v) for v in values],
            "length": int(len(values)),
            "first": int(values[0]),
            "last": int(values[-1]),
        }
        for values in segments
    ]
    coverage_ratio = float(len(covered) / max(1, num_blocks))
    stats = {
        "num_segments": int(len(payload)),
        "mean_segment_length": float(np.mean([row["length"] for row in payload])) if payload else 0.0,
        "max_segment_length": int(max([row["length"] for row in payload], default=0)),
        "coverage_ratio": coverage_ratio,
    }
    return payload, stats


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman_correlation(xs, ys):
    if len(xs) < 2:
        return None
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    rx = rankdata(xs)
    ry = rankdata(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    if denom == 0.0:
        return None
    return float((rx * ry).sum() / denom)


def select_target_units(num_blocks, num_targets, seed):
    if num_targets is None or int(num_targets) <= 0 or int(num_targets) >= num_blocks:
        return list(range(num_blocks))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(num_blocks, size=int(num_targets), replace=False).tolist())


def group_pairs_by_target(candidate_pairs):
    grouped = {}
    for first, second in candidate_pairs:
        grouped.setdefault(int(second), []).append(int(first))
    return grouped


def sample_target_shared_context_templates(target, num_blocks, batch_size, template_len, device, generator):
    """
    Sample shared block templates for a fixed target.

    These templates exclude the target itself and are reused across different
    candidate predecessors so the proposal stage acts as candidate filtering
    instead of implicitly re-scoring different random completions.
    """
    candidates = [idx for idx in range(num_blocks) if idx != target]
    candidate_tensor = torch.tensor(candidates, device=device, dtype=torch.long)
    templates = []
    for _ in range(batch_size):
        perm = torch.randperm(candidate_tensor.numel(), generator=generator, device=device)
        templates.append(candidate_tensor[perm][: int(template_len)])
    return torch.stack(templates, dim=0)


def build_target_probe_context_orders(
    target,
    candidates,
    shared_context_templates,
    probe_prefix_len,
    target_probe_index,
    device,
):
    """
    Build target-centered early/mid probes.

    The target is fixed at a configurable position inside an early prefix, rather
    than being hard-coded at reveal position 2. Each candidate block is inserted
    into one context slot of a shared template; the rest of the prefix and suffix
    are held as consistent as possible across candidates.

    This makes proposal scores mean:
    "for this fixed target and this early/mid probe context, which candidate
    blocks seem helpful / related?"

    It does NOT mean:
    "this candidate must be the immediate predecessor of the target".
    """
    batch_size, template_len = shared_context_templates.shape
    num_candidates = len(candidates)
    prefix_context_slots = int(probe_prefix_len) - 1
    if template_len < prefix_context_slots:
        raise ValueError("shared_context_templates is too short for the requested probe_prefix_len.")

    base_orders = []
    ctrl_orders = []
    ctrl_valid_mask = []
    candidate_slots = []

    for batch_idx in range(batch_size):
        template = shared_context_templates[batch_idx].tolist()
        for candidate in candidates:
            base_context = template[:prefix_context_slots]
            if candidate not in base_context:
                replace_slot = 0
                base_context[replace_slot] = candidate
            else:
                replace_slot = base_context.index(candidate)
            prefix = list(base_context)
            prefix.insert(int(target_probe_index), int(target))
            candidate_slot = replace_slot if replace_slot < int(target_probe_index) else replace_slot + 1
            candidate_slots.append(int(candidate_slot))

            used_base = set(prefix)
            # Build the suffix from the full shared template, not only the tail slice.
            # Otherwise, when a candidate is injected into the prefix by replacing an
            # existing context block, the displaced block can be dropped entirely,
            # producing inconsistent order lengths (e.g. 31 instead of 32 for block32).
            rest = [idx for idx in template if idx not in used_base]
            base_orders.append(torch.tensor(prefix + rest, device=device, dtype=torch.long))

            ctrl_fill = next(
                (idx for idx in template[prefix_context_slots:] if idx != target and idx != candidate and idx not in base_context),
                None,
            )
            if ctrl_fill is None:
                ctrl_prefix_context = [idx for idx in base_context if idx != candidate]
                ctrl_valid_mask.append(False)
            else:
                ctrl_prefix_context = [idx for idx in base_context if idx != candidate]
                ctrl_prefix_context.append(ctrl_fill)
                ctrl_valid_mask.append(True)
            ctrl_prefix_context = ctrl_prefix_context[:prefix_context_slots]
            ctrl_prefix = list(ctrl_prefix_context)
            ctrl_prefix.insert(int(target_probe_index), int(target))
            used_ctrl = set(ctrl_prefix)
            ctrl_rest = [idx for idx in template if idx not in used_ctrl]
            ctrl_orders.append(torch.tensor(ctrl_prefix + ctrl_rest, device=device, dtype=torch.long))

    seq_len = shared_context_templates.size(1) + 1
    base_orders = torch.stack(base_orders, dim=0).view(batch_size, num_candidates, seq_len)
    ctrl_orders = torch.stack(ctrl_orders, dim=0).view(batch_size, num_candidates, seq_len)
    ctrl_valid_mask = torch.tensor(ctrl_valid_mask, device=device, dtype=torch.bool).view(batch_size, num_candidates)
    candidate_slots = torch.tensor(candidate_slots, device=device, dtype=torch.long).view(batch_size, num_candidates)
    return base_orders, ctrl_orders, ctrl_valid_mask, candidate_slots


def compute_attention_mass_for_target_candidate(attentions, block_len, target_slot, candidate_slot):
    per_layer = []
    target_start = 1 + int(target_slot) * block_len
    target_end = 1 + (int(target_slot) + 1) * block_len
    cand_start = 1 + int(candidate_slot) * block_len
    cand_end = 1 + (int(candidate_slot) + 1) * block_len
    for att in attentions:
        local = att[:, :, target_start:target_end, cand_start:cand_end].mean(dim=(-1, -2, -3))
        per_layer.append(local)
    return torch.stack(per_layer, dim=0).mean(dim=0)


def proposal_scores_attention_early(
    model,
    tokens,
    candidate_pairs,
    proposal_batch_size,
    proposal_num_batches,
    device,
    autocast_context,
    seed,
    probe_prefix_len,
    target_probe_index,
    token_perm=None,
):
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cuda" if "cuda" in device else "cpu")
    generator.manual_seed(seed)
    batch_size = max(1, int(proposal_batch_size))
    pair_scores = {(int(first), int(second)): 0.0 for first, second in candidate_pairs}
    grouped = group_pairs_by_target(candidate_pairs)

    for target, predecessors in grouped.items():
        predecessors = list(predecessors)
        local_sums = torch.zeros(len(predecessors), dtype=torch.float64)
        for _ in range(int(proposal_num_batches)):
            idx = sample_batch(tokens, batch_size, model.config.block_size, rng, device, token_perm=token_perm)
            shared_templates = sample_target_shared_context_templates(
                target,
                model.num_blocks,
                idx.size(0),
                template_len=model.num_blocks - 1,
                device=device,
                generator=generator,
            )
            base_orders, _, _, candidate_slots = build_target_probe_context_orders(
                target,
                predecessors,
                shared_templates,
                probe_prefix_len=probe_prefix_len,
                target_probe_index=target_probe_index,
                device=device,
            )

            flat_orders = base_orders.reshape(idx.size(0) * len(predecessors), model.num_blocks)
            flat_idx = idx.unsqueeze(1).expand(idx.size(0), len(predecessors), idx.size(1)).reshape(
                idx.size(0) * len(predecessors),
                idx.size(1),
            )
            with autocast_context:
                _, _, attention_outputs = model(
                    flat_idx,
                    mode=None,
                    orders=model._expand_block_orders_to_token_orders(flat_orders),
                    return_token_loss=False,
                    return_hidden=False,
                    return_attentions=True,
                )
            masses = []
            flat_candidate_slots = candidate_slots.reshape(-1)
            for sample_idx in range(flat_orders.size(0)):
                sample_attentions = [layer_att[sample_idx : sample_idx + 1] for layer_att in attention_outputs]
                masses.append(
                    compute_attention_mass_for_target_candidate(
                        sample_attentions,
                        model.block_order_block_len,
                        target_slot=target_probe_index,
                        candidate_slot=int(flat_candidate_slots[sample_idx].item()),
                    )
                )
            masses = torch.stack(masses, dim=0).view(idx.size(0), len(predecessors))
            local_sums += masses.mean(dim=0).cpu().double()

        local_means = local_sums / max(1, int(proposal_num_batches))
        for pred_idx, pred in enumerate(predecessors):
            pair_scores[(int(pred), int(target))] = float(local_means[pred_idx].item())

    return [pair_scores[(int(first), int(second))] for first, second in candidate_pairs]


def proposal_scores_loo_early(
    model,
    tokens,
    candidate_pairs,
    proposal_batch_size,
    proposal_num_batches,
    device,
    autocast_context,
    seed,
    probe_prefix_len,
    target_probe_index,
    token_perm=None,
):
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cuda" if "cuda" in device else "cpu")
    generator.manual_seed(seed + 17)
    batch_size = max(1, int(proposal_batch_size))
    probe_window_k = max(2, int(probe_prefix_len), int(target_probe_index) + 1)
    pair_scores = {(int(first), int(second)): 0.0 for first, second in candidate_pairs}
    grouped = group_pairs_by_target(candidate_pairs)

    for target, predecessors in grouped.items():
        predecessors = list(predecessors)
        local_sums = torch.zeros(len(predecessors), dtype=torch.float64)
        local_counts = torch.zeros(len(predecessors), dtype=torch.float64)
        for _ in range(int(proposal_num_batches)):
            idx = sample_batch(tokens, batch_size, model.config.block_size, rng, device, token_perm=token_perm)
            shared_templates = sample_target_shared_context_templates(
                target,
                model.num_blocks,
                idx.size(0),
                template_len=model.num_blocks - 1,
                device=device,
                generator=generator,
            )
            base_orders, ctrl_orders, ctrl_valid_mask, _ = build_target_probe_context_orders(
                target,
                predecessors,
                shared_templates,
                probe_prefix_len=probe_prefix_len,
                target_probe_index=target_probe_index,
                device=device,
            )

            flat_base = base_orders.reshape(idx.size(0) * len(predecessors), model.num_blocks)
            flat_ctrl = ctrl_orders.reshape(idx.size(0) * len(predecessors), model.num_blocks)
            all_orders = torch.cat([flat_base, flat_ctrl], dim=0)
            flat_idx = idx.unsqueeze(1).expand(idx.size(0), len(predecessors), idx.size(1)).reshape(
                idx.size(0) * len(predecessors),
                idx.size(1),
            )
            all_idx = torch.cat([flat_idx, flat_idx], dim=0)
            metrics = evaluate_block_order_quality(
                model,
                all_idx,
                all_orders,
                # Keep the probe evaluation window consistent with the fixed
                # target probe slot. We are still using block_losses at
                # target_probe_index below, but this avoids a misleading
                # "prefix_k=2" setting when the target is intentionally placed
                # later in the early/mid probe prefix.
                prefix_k=probe_window_k,
                block_len=model.block_order_block_len,
                autocast_context=autocast_context,
            )
            block_losses = metrics["block_losses"]
            # This is a fixed-target-position removal-style probe:
            # the candidate is removed from the probe prefix context, and a
            # shared-template filler may move forward only to keep the prefix
            # length valid. We then measure the target block loss change at
            # target_probe_index.
            base_losses = block_losses[: flat_idx.size(0), int(target_probe_index)].view(idx.size(0), len(predecessors))
            ctrl_losses = block_losses[flat_idx.size(0) :, int(target_probe_index)].view(idx.size(0), len(predecessors))
            deltas = (ctrl_losses - base_losses)
            valid = ctrl_valid_mask.float()
            local_sums += (deltas * valid).sum(dim=0).cpu().double()
            local_counts += valid.sum(dim=0).cpu().double()

        local_means = torch.where(local_counts > 0, local_sums / local_counts.clamp_min(1.0), torch.zeros_like(local_sums))
        for pred_idx, pred in enumerate(predecessors):
            pair_scores[(int(pred), int(target))] = float(local_means[pred_idx].item())

    return [pair_scores[(int(first), int(second))] for first, second in candidate_pairs]


def build_full_candidate_pairs(num_blocks, targets):
    return [(int(j), int(i)) for i in targets for j in range(num_blocks) if j != i]


def build_random_proposals(num_blocks, targets, k, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        candidates = [idx for idx in range(num_blocks) if idx != target]
        chosen = rng.choice(candidates, size=min(int(k), len(candidates)), replace=False).tolist()
        for rank, pred in enumerate(chosen, start=1):
            rows.append(
                {
                    "first": int(pred),
                    "second": int(target),
                    "proposal_score": float(len(chosen) - rank + 1),
                    "proposal_rank_within_target": int(rank),
                }
            )
    return rows


def build_scored_topk_proposals(num_blocks, targets, k, scores):
    by_target = {int(target): [] for target in targets}
    for first, second, score in scores:
        by_target[int(second)].append((float(score), int(first)))
    rows = []
    for target in targets:
        local = sorted(by_target[int(target)], reverse=True)[: min(int(k), len(by_target[int(target)]))]
        for rank, (score, pred) in enumerate(local, start=1):
            rows.append(
                {
                    "first": int(pred),
                    "second": int(target),
                    "proposal_score": float(score),
                    "proposal_rank_within_target": int(rank),
                }
            )
    return rows


def attach_original_indices(rows, block_perm):
    if block_perm is None:
        return rows
    out = []
    for row in rows:
        payload = dict(row)
        payload["first_original"] = int(block_perm[int(row["first"])].item())
        payload["second_original"] = int(block_perm[int(row["second"])].item())
        out.append(payload)
    return out


def merge_validation_into_proposals(proposals, validated_rows):
    validation_map = {(int(row["first"]), int(row["second"])): row for row in validated_rows}
    merged = []
    for row in proposals:
        key = (int(row["first"]), int(row["second"]))
        merged.append({**row, **validation_map[key]})
    return merged


def summarize_run(
    ckpt_path,
    proposal_name,
    proposals,
    validated_rows,
    segment_stats,
    full_enum_validated_rows,
):
    proposal_scores = [float(row["proposal_score"]) for row in proposals]
    validation_scores = [float(row["score"]) for row in validated_rows]
    reverse_margins = [float(row["reverse_margin"]) for row in validated_rows if row["reverse_margin"] == row["reverse_margin"]]
    positive_margin = sum(1 for row in validated_rows if row["reverse_margin"] == row["reverse_margin"] and row["reverse_margin"] > 0.0)
    spearman = spearman_correlation(proposal_scores, validation_scores)

    proposal_count = len(proposals)
    validated_count = len(validated_rows)
    gold_k = min(len(full_enum_validated_rows), validated_count)
    gold_topk = {
        (int(row["first"]), int(row["second"]))
        for row in full_enum_validated_rows[:gold_k]
    }
    ours = {(int(row["first"]), int(row["second"])) for row in validated_rows[:gold_k]}
    overlap = len(gold_topk & ours)
    recall_at_k = float(overlap / max(1, len(gold_topk)))

    return {
        "ckpt_path": str(ckpt_path),
        "proposal": proposal_name,
        "proposal_count": int(proposal_count),
        "validated_count": int(validated_count),
        "validated_hit_rate": float(positive_margin / max(1, validated_count)),
        "proposal_validation_spearman": spearman,
        "validated_edge_mean_score": float(np.mean(validation_scores)) if validation_scores else None,
        "validated_edge_mean_reverse_margin": float(np.mean(reverse_margins)) if reverse_margins else None,
        "segment_count": int(segment_stats["num_segments"]),
        "segment_mean_length": float(segment_stats["mean_segment_length"]),
        "segment_max_length": int(segment_stats["max_segment_length"]),
        "segment_coverage_ratio": float(segment_stats["coverage_ratio"]),
        "recall_at_k_vs_full_enum": recall_at_k,
        "overlap_at_k_vs_full_enum": int(overlap),
        "segment_reproducibility": None,
    }


def main():
    args = parse_args()
    proposals_requested = parse_csv_list(args.proposal, str)
    proposals_requested = [item.strip() for item in proposals_requested]
    compare_rows = []
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for ckpt_raw in args.ckpt:
        ckpt_path = Path(ckpt_raw)
        checkpoint = load_checkpoint(ckpt_path, args.device)
        model = build_model(checkpoint, args.device)
        if int(args.probe_prefix_len) > int(model.num_blocks):
            raise ValueError(
                f"probe_prefix_len={args.probe_prefix_len} exceeds num_blocks={model.num_blocks} "
                f"for checkpoint {ckpt_path}."
            )
        autocast_context = get_autocast_context(args.device, args.dtype)
        data_dir = resolve_data_dir(args.dataset, args.data_dir, checkpoint)
        tokens = load_tokens(data_dir, args.split)
        perm_state = load_block_permutation_from_checkpoint(
            checkpoint,
            num_blocks=model.num_blocks,
            block_len=model.block_order_block_len,
        )
        token_perm = None if perm_state is None else perm_state["token_perm"]
        block_perm = None if perm_state is None else perm_state["block_perm"]

        targets = select_target_units(model.num_blocks, args.num_targets, args.seed)
        full_candidate_pairs = build_full_candidate_pairs(model.num_blocks, targets)

        ckpt_out_dir = out_root / ckpt_path.stem
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[proposal-benchmark] checkpoint={ckpt_path} num_targets={len(targets)} num_pairs={len(full_candidate_pairs)}")

        full_enum_validated_rows = validate_candidate_pairs(
            model,
            tokens,
            full_candidate_pairs,
            batch_size=int(args.proposal_batch_size),
            num_batches=int(args.pair_mining_batches),
            pair_score_k=int(args.pair_score_k),
            tv_weight=float(args.tv_weight),
            device=args.device,
            autocast_context=autocast_context,
            seed=int(args.seed),
            eval_batch_size=int(args.pair_eval_batch_size),
            token_perm=token_perm,
        )
        full_enum_validated_rows = attach_original_indices(full_enum_validated_rows, block_perm)

        requested_plus_gold = list(dict.fromkeys(["full_enum"] + proposals_requested))
        for proposal_name in requested_plus_gold:
            proposal_dir = ckpt_out_dir / proposal_name
            proposal_dir.mkdir(parents=True, exist_ok=True)

            if proposal_name == "full_enum":
                proposals = [
                    {
                        "first": int(first),
                        "second": int(second),
                        "proposal_score": 1.0,
                        "proposal_rank_within_target": None,
                    }
                    for first, second in full_candidate_pairs
                ]
                validated_rows = merge_validation_into_proposals(proposals, full_enum_validated_rows)
            elif proposal_name == "random":
                proposals = build_random_proposals(
                    model.num_blocks,
                    targets,
                    k=int(args.num_candidates_per_target),
                    seed=int(args.seed),
                )
                candidate_pairs = [(int(row["first"]), int(row["second"])) for row in proposals]
                validated_rows = validate_candidate_pairs(
                    model,
                    tokens,
                    candidate_pairs,
                    batch_size=int(args.proposal_batch_size),
                    num_batches=int(args.pair_mining_batches),
                    pair_score_k=int(args.pair_score_k),
                    tv_weight=float(args.tv_weight),
                    device=args.device,
                    autocast_context=autocast_context,
                    seed=int(args.seed) + 11,
                    eval_batch_size=int(args.pair_eval_batch_size),
                    token_perm=token_perm,
                )
                validated_rows = merge_validation_into_proposals(proposals, attach_original_indices(validated_rows, block_perm))
            elif proposal_name == "attn_early":
                attn_scores = proposal_scores_attention_early(
                    model,
                    tokens,
                    full_candidate_pairs,
                    proposal_batch_size=int(args.proposal_batch_size),
                    proposal_num_batches=int(args.proposal_num_batches),
                    device=args.device,
                    autocast_context=autocast_context,
                    seed=int(args.seed) + 101,
                    probe_prefix_len=int(args.probe_prefix_len),
                    target_probe_index=int(args.target_probe_index),
                    token_perm=token_perm,
                )
                score_rows = [(first, second, score) for (first, second), score in zip(full_candidate_pairs, attn_scores)]
                proposals = build_scored_topk_proposals(
                    model.num_blocks,
                    targets,
                    k=int(args.num_candidates_per_target),
                    scores=score_rows,
                )
                candidate_pairs = [(int(row["first"]), int(row["second"])) for row in proposals]
                validated_rows = validate_candidate_pairs(
                    model,
                    tokens,
                    candidate_pairs,
                    batch_size=int(args.proposal_batch_size),
                    num_batches=int(args.pair_mining_batches),
                    pair_score_k=int(args.pair_score_k),
                    tv_weight=float(args.tv_weight),
                    device=args.device,
                    autocast_context=autocast_context,
                    seed=int(args.seed) + 19,
                    eval_batch_size=int(args.pair_eval_batch_size),
                    token_perm=token_perm,
                )
                validated_rows = merge_validation_into_proposals(proposals, attach_original_indices(validated_rows, block_perm))
            elif proposal_name == "loo_early":
                loo_scores = proposal_scores_loo_early(
                    model,
                    tokens,
                    full_candidate_pairs,
                    proposal_batch_size=int(args.proposal_batch_size),
                    proposal_num_batches=int(args.proposal_num_batches),
                    device=args.device,
                    autocast_context=autocast_context,
                    seed=int(args.seed) + 203,
                    probe_prefix_len=int(args.probe_prefix_len),
                    target_probe_index=int(args.target_probe_index),
                    token_perm=token_perm,
                )
                score_rows = [(first, second, score) for (first, second), score in zip(full_candidate_pairs, loo_scores)]
                proposals = build_scored_topk_proposals(
                    model.num_blocks,
                    targets,
                    k=int(args.num_candidates_per_target),
                    scores=score_rows,
                )
                candidate_pairs = [(int(row["first"]), int(row["second"])) for row in proposals]
                validated_rows = validate_candidate_pairs(
                    model,
                    tokens,
                    candidate_pairs,
                    batch_size=int(args.proposal_batch_size),
                    num_batches=int(args.pair_mining_batches),
                    pair_score_k=int(args.pair_score_k),
                    tv_weight=float(args.tv_weight),
                    device=args.device,
                    autocast_context=autocast_context,
                    seed=int(args.seed) + 29,
                    eval_batch_size=int(args.pair_eval_batch_size),
                    token_perm=token_perm,
                )
                validated_rows = merge_validation_into_proposals(proposals, attach_original_indices(validated_rows, block_perm))
            else:
                raise ValueError(f"Unknown proposal type: {proposal_name}")

            proposals = attach_original_indices(proposals, block_perm)
            if proposal_name == "full_enum":
                validated_rows = attach_original_indices(validated_rows, block_perm)
            validated_rows.sort(key=lambda row: (row["score"], row["reverse_margin"]), reverse=True)

            segments, segment_stats = aggregate_top_pairs_to_segments(
                validated_rows,
                num_blocks=model.num_blocks,
                top_k=min(len(validated_rows), max(1, len(targets) * int(args.num_candidates_per_target))),
                max_segment_len=int(args.segment_len),
            )
            if block_perm is not None:
                for row in segments:
                    row["segment_original"] = [int(block_perm[int(v)].item()) for v in row["segment"]]

            summary = summarize_run(
                ckpt_path=ckpt_path,
                proposal_name=proposal_name,
                proposals=proposals,
                validated_rows=validated_rows,
                segment_stats=segment_stats,
                full_enum_validated_rows=full_enum_validated_rows,
            )
            summary["num_targets"] = int(len(targets))
            summary["num_candidates_per_target"] = int(args.num_candidates_per_target)
            summary["pair_mining_batches"] = int(args.pair_mining_batches)
            summary["proposal_num_batches"] = int(args.proposal_num_batches)
            summary["seed"] = int(args.seed)

            save_jsonl(proposal_dir / "proposals.jsonl", proposals)
            save_jsonl(proposal_dir / "validated_edges.jsonl", validated_rows)
            save_json(
                proposal_dir / "segments.json",
                {
                    "segments": segments,
                    "stats": segment_stats,
                },
            )
            save_json(proposal_dir / "summary.json", summary)
            compare_rows.append(summary)

    compare_path = out_root / "compare.csv"
    fieldnames = sorted({key for row in compare_rows for key in row.keys()})
    with compare_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compare_rows)
    print(f"saved proposal benchmark compare table to {compare_path}")


if __name__ == "__main__":
    main()
