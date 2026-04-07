import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    build_ascending_block_orders,
    evaluate_block_order_quality,
    kendall_tau_to_l2r_per_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical structured benchmark: mine pair2 structure at the current unit level, "
            "aggregate directed segments, then recurse with segments as new units."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, default=Path("out/out-wikitext103-random-b32/ckpt.pt"))
    parser.add_argument("--out_dir", type=Path, default=Path("Report/hierarchical_structured_benchmark_b32"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--pair_mining_batches", type=int, default=24)
    parser.add_argument("--pair_eval_batch_size", type=int, default=8)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--random_pool_size", type=int, default=64)
    parser.add_argument("--structured_pool_size", type=int, default=64)
    parser.add_argument("--top_pair_pool_size", type=int, default=128)
    parser.add_argument("--aggregate_top_k_pairs", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--segment_len", type=int, default=4)
    parser.add_argument("--pair_score_k", type=int, default=2)
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--log_every_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
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
    return parser.parse_args()


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in device or dtype == "float32":
        return nullcontext()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def resolve_data_dir(args, checkpoint):
    if args.data_dir is not None:
        return args.data_dir
    dataset = args.dataset or checkpoint.get("config", {}).get("dataset")
    if dataset is None:
        raise ValueError("Could not infer dataset. Pass --dataset or --data_dir.")
    return Path(__file__).resolve().parent / "data" / dataset


def load_tokens(data_dir: Path, split: str):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str):
    max_start = len(tokens) - block_size
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than block_size.")
    starts = rng.integers(0, max_start, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    )
    return batch.to(device)


def build_initial_units(num_blocks):
    return [{"unit_id": idx, "blocks": [idx]} for idx in range(num_blocks)]


def flatten_unit_order(unit_order, units):
    blocks = []
    for unit_idx in unit_order:
        blocks.extend(units[int(unit_idx)]["blocks"])
    return blocks


def unit_order_to_block_order_tensor(unit_order, units, device):
    return torch.tensor(flatten_unit_order(unit_order, units), device=device, dtype=torch.long)


def build_random_suffix_orders_for_unit_pairs(unit_pairs, units, batch_size, device, generator):
    num_units = len(units)
    orders = []
    all_unit_indices = list(range(num_units))
    for first_unit, second_unit in unit_pairs:
        pair_orders = []
        remaining_units = [idx for idx in all_unit_indices if idx not in (first_unit, second_unit)]
        remaining_tensor = torch.tensor(remaining_units, device=device, dtype=torch.long)
        for _ in range(batch_size):
            perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
            ordered_units = [first_unit, second_unit] + remaining_tensor[perm].tolist()
            pair_orders.append(unit_order_to_block_order_tensor(ordered_units, units, device))
        orders.append(torch.stack(pair_orders, dim=0))
    return torch.stack(orders, dim=1)


def compute_signal_from_block_losses(block_losses, early_k, tv_weight):
    block_losses = block_losses.float()
    early = block_losses[:, :early_k]
    full = block_losses

    def build_metrics(window):
        area = -window.mean(dim=-1)
        if window.size(1) < 2:
            total_variation = torch.zeros_like(area)
        else:
            total_variation = -(window[:, 1:] - window[:, :-1]).abs().sum(dim=-1)
        return area, total_variation

    early_area, early_tv = build_metrics(early)
    full_area, full_tv = build_metrics(full)
    return {
        "early_area_plus_tv": early_area + float(tv_weight) * early_tv,
        "full_area_plus_tv": full_area + float(tv_weight) * full_tv,
    }


def compute_pair2_unit_scores(block_losses, first_two_unit_lengths, tv_weight):
    scores = []
    cursor = 0
    for unit_len in first_two_unit_lengths:
        next_cursor = cursor + int(unit_len)
        scores.append(block_losses[:, cursor:next_cursor].mean(dim=1))
        cursor = next_cursor
    unit_losses = torch.stack(scores, dim=1)
    area = -unit_losses.mean(dim=1)
    if unit_losses.size(1) < 2:
        total_variation = torch.zeros_like(area)
    else:
        total_variation = -(unit_losses[:, 1:] - unit_losses[:, :-1]).abs().sum(dim=1)
    return area + float(tv_weight) * total_variation


def evaluate_orders_with_metrics(model, idx, block_orders, early_k, tv_weight, autocast_context, eval_batch_size):
    total = block_orders.size(0)
    chunk_size = max(1, int(eval_batch_size))
    block_loss_parts = []
    kendall_parts = []
    early_parts = []
    full_parts = []
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        metrics = evaluate_block_order_quality(
            model,
            idx[start:end],
            block_orders[start:end],
            prefix_k=max(2, min(early_k, model.num_blocks)),
            block_len=model.block_order_block_len,
            autocast_context=autocast_context,
        )
        signal_metrics = compute_signal_from_block_losses(metrics["block_losses"], early_k=early_k, tv_weight=tv_weight)
        block_loss_parts.append(metrics["block_losses"])
        kendall_parts.append(metrics["kendall_per_sample"])
        early_parts.append(signal_metrics["early_area_plus_tv"])
        full_parts.append(signal_metrics["full_area_plus_tv"])
    kendall = torch.cat(kendall_parts, dim=0)
    return {
        "block_losses": torch.cat(block_loss_parts, dim=0),
        "kendall": kendall,
        "kendall_distance": (1.0 - kendall) / 2.0,
        "early_area_plus_tv": torch.cat(early_parts, dim=0),
        "full_area_plus_tv": torch.cat(full_parts, dim=0),
    }


def mine_unit_pair_scores(
    model,
    units,
    tokens,
    batch_size,
    num_batches,
    pair_score_k,
    tv_weight,
    device,
    autocast_context,
    seed,
    pair_eval_batch_size,
):
    num_units = len(units)
    unit_pairs = [(i, j) for i in range(num_units) for j in range(num_units) if i != j]
    pair_score_sums = torch.zeros(len(unit_pairs), dtype=torch.float64)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cuda" if "cuda" in device else "cpu")
    generator.manual_seed(seed)
    chunk_size = max(1, int(pair_eval_batch_size))
    eval_chunk_size = max(1, int(pair_eval_batch_size))

    for _ in range(int(num_batches)):
        idx = sample_batch(tokens, batch_size, model.config.block_size, rng, device)
        for start in range(0, len(unit_pairs), chunk_size):
            chunk_pairs = unit_pairs[start:start + chunk_size]
            pair_orders = build_random_suffix_orders_for_unit_pairs(
                chunk_pairs,
                units=units,
                batch_size=idx.size(0),
                device=device,
                generator=generator,
            )
            flat_orders = pair_orders.reshape(idx.size(0) * len(chunk_pairs), model.num_blocks)
            flat_idx = idx.unsqueeze(1).expand(idx.size(0), len(chunk_pairs), idx.size(1)).reshape(
                idx.size(0) * len(chunk_pairs),
                idx.size(1),
            )
            score_parts = []
            for eval_start in range(0, flat_orders.size(0), eval_chunk_size):
                eval_end = min(flat_orders.size(0), eval_start + eval_chunk_size)
                eval_out = evaluate_orders_with_metrics(
                    model,
                    flat_idx[eval_start:eval_end],
                    flat_orders[eval_start:eval_end],
                    early_k=max(2, int(pair_score_k)),
                    tv_weight=tv_weight,
                    autocast_context=autocast_context,
                    eval_batch_size=eval_chunk_size,
                )
                local_scores = []
                for pair_idx, pair in enumerate(chunk_pairs):
                    first_unit, second_unit = pair
                    unit_lens = [
                        len(units[first_unit]["blocks"]),
                        len(units[second_unit]["blocks"]),
                    ]
                    begin = pair_idx * idx.size(0)
                    end = (pair_idx + 1) * idx.size(0)
                    local_scores.append(
                        compute_pair2_unit_scores(
                            eval_out["block_losses"][begin:end],
                            first_two_unit_lengths=unit_lens[:pair_score_k],
                            tv_weight=tv_weight,
                        ).cpu()
                    )
                score_parts.append(torch.stack(local_scores, dim=1))
            scores = torch.cat(score_parts, dim=0).mean(dim=0).double()
            pair_score_sums[start:start + len(chunk_pairs)] += scores

    pair_score_means = pair_score_sums / max(1, int(num_batches))
    pair_score_matrix = torch.full((num_units, num_units), float("-inf"), dtype=torch.float64)
    for pair_idx, (i, j) in enumerate(unit_pairs):
        pair_score_matrix[i, j] = pair_score_means[pair_idx]

    top_pairs = sorted(
        [
            {
                "first_unit": int(i),
                "second_unit": int(j),
                "first_blocks": units[i]["blocks"],
                "second_blocks": units[j]["blocks"],
                "score": float(pair_score_matrix[i, j].item()),
                "margin_vs_reverse": float((pair_score_matrix[i, j] - pair_score_matrix[j, i]).item()),
            }
            for i in range(num_units)
            for j in range(num_units)
            if i != j
        ],
        key=lambda row: (row["score"], row["margin_vs_reverse"]),
        reverse=True,
    )
    return pair_score_matrix, top_pairs


def aggregate_top_unit_pairs_to_segments(top_pairs, num_units, top_k, max_segment_len):
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
        first = int(pair["first_unit"])
        second = int(pair["second_unit"])
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

    starts = [node for node in range(num_units) if node not in prev_map and node in next_map]
    segments = []
    for start in starts:
        unit_indices = build_segment(start)
        if len(unit_indices) >= 2:
            segments.append(unit_indices)
    segments.sort(key=lambda values: (-len(values), values[0], values[-1]))
    return segments


def build_next_level_units(units, aggregated_segments):
    used = set()
    next_units = []
    for unit_indices in aggregated_segments:
        blocks = []
        for unit_idx in unit_indices:
            used.add(unit_idx)
            blocks.extend(units[unit_idx]["blocks"])
        next_units.append({"unit_id": len(next_units), "blocks": blocks, "source_units": unit_indices})
    for unit_idx, unit in enumerate(units):
        if unit_idx in used:
            continue
        next_units.append({"unit_id": len(next_units), "blocks": list(unit["blocks"]), "source_units": [unit_idx]})
    next_units.sort(key=lambda row: (row["blocks"][0], len(row["blocks"])))
    for unit_id, unit in enumerate(next_units):
        unit["unit_id"] = unit_id
    return next_units


def longest_contiguous_run(order):
    values = order.tolist()
    best = 1
    current = 1
    for idx in range(1, len(values)):
        if values[idx] == values[idx - 1] + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def adjacent_pair_count(orders):
    shifted = orders[:, 1:] - orders[:, :-1]
    return (shifted == 1).sum(dim=1).float()


def prefix_mean_index(orders, prefix_len):
    prefix_len = max(1, min(prefix_len, orders.size(1)))
    return orders[:, :prefix_len].float().mean(dim=1)


def build_segment_guided_unit_order(units, unit_segments, prefix_len, generator, device):
    num_units = len(units)
    if not unit_segments:
        return torch.randperm(num_units, generator=generator, device=device)
    used = set()
    chosen_segments = []
    prefix_budget = max(2, min(prefix_len, num_units))
    shuffled_segment_indices = torch.randperm(len(unit_segments), generator=generator, device=device).tolist()
    for seg_idx in shuffled_segment_indices:
        segment = unit_segments[seg_idx]
        if any(value in used for value in segment):
            continue
        if sum(len(seg) for seg in chosen_segments) + len(segment) > prefix_budget:
            continue
        chosen_segments.append(segment)
        used.update(segment)
        if sum(len(seg) for seg in chosen_segments) >= prefix_budget:
            break

    prefix_units = []
    for segment in chosen_segments:
        prefix_units.extend(segment)
    remaining = [idx for idx in range(num_units) if idx not in used]
    remaining_tensor = torch.tensor(remaining, device=device, dtype=torch.long)
    perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
    prefix_tensor = torch.tensor(prefix_units, device=device, dtype=torch.long)
    return torch.cat([prefix_tensor, remaining_tensor[perm]], dim=0)


def build_chain_biased_unit_order(units, pair_score_matrix, top_pairs, segment_len, generator, device):
    num_units = len(units)
    seed = top_pairs[torch.randint(0, len(top_pairs), (1,), generator=generator, device=device).item()]
    chain = [int(seed["first_unit"]), int(seed["second_unit"])]
    used = set(chain)
    while len(chain) < min(segment_len, num_units):
        last = chain[-1]
        candidates = []
        for nxt in range(num_units):
            if nxt in used or nxt == last:
                continue
            candidates.append((pair_score_matrix[last, nxt].item(), nxt))
        if not candidates:
            break
        candidates.sort(reverse=True)
        top_choices = [nxt for _, nxt in candidates[: min(4, len(candidates))]]
        nxt = top_choices[torch.randint(0, len(top_choices), (1,), generator=generator, device=device).item()]
        chain.append(nxt)
        used.add(nxt)
    remaining = [idx for idx in range(num_units) if idx not in used]
    remaining_tensor = torch.tensor(remaining, device=device, dtype=torch.long)
    perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
    return torch.cat([torch.tensor(chain, device=device), remaining_tensor[perm]], dim=0)


def build_pair_biased_unit_order(units, top_pairs, generator, device):
    num_units = len(units)
    pair = top_pairs[torch.randint(0, len(top_pairs), (1,), generator=generator, device=device).item()]
    chosen = [int(pair["first_unit"]), int(pair["second_unit"])]
    remaining = [idx for idx in range(num_units) if idx not in chosen]
    remaining_tensor = torch.tensor(remaining, device=device, dtype=torch.long)
    perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
    return torch.cat([torch.tensor(chosen, device=device), remaining_tensor[perm]], dim=0)


def build_random_block_orders_from_unit_pool(pool_unit_orders, units, device):
    return torch.stack(
        [unit_order_to_block_order_tensor(unit_order.tolist(), units, device) for unit_order in pool_unit_orders],
        dim=0,
    )


def build_candidate_pool(units, pair_score_matrix, top_pairs, aggregated_segments, structured_pool_size, random_pool_size, prefix_len, segment_len, generator, device):
    num_units = len(units)
    structured_unit_orders = []
    per_group = max(1, structured_pool_size // 4)
    for _ in range(per_group):
        structured_unit_orders.append(torch.randperm(num_units, generator=generator, device=device))
    for _ in range(per_group):
        structured_unit_orders.append(build_segment_guided_unit_order(units, aggregated_segments, prefix_len, generator, device))
    for _ in range(per_group):
        structured_unit_orders.append(build_chain_biased_unit_order(units, pair_score_matrix, top_pairs, segment_len, generator, device))
    while len(structured_unit_orders) < structured_pool_size:
        structured_unit_orders.append(build_pair_biased_unit_order(units, top_pairs, generator, device))

    random_unit_orders = [torch.randperm(num_units, generator=generator, device=device) for _ in range(random_pool_size)]
    return (
        build_random_block_orders_from_unit_pool(structured_unit_orders, units, device),
        build_random_block_orders_from_unit_pool(random_unit_orders, units, device),
    )


def summarize_selected_orders(selected_orders, selected_signals, selected_full_signals, prefix_len):
    kendall = kendall_tau_to_l2r_per_sample(selected_orders)
    return {
        "num_samples": int(selected_orders.size(0)),
        "mean_prefix_mean_index": float(prefix_mean_index(selected_orders, prefix_len).mean().item()),
        "mean_adjacent_pairs": float(adjacent_pair_count(selected_orders).mean().item()),
        "mean_longest_run": float(np.mean([longest_contiguous_run(order.cpu()) for order in selected_orders])),
        "mean_kendall_tau": float(kendall.mean().item()),
        "mean_kendall_distance": float(((1.0 - kendall) / 2.0).mean().item()),
        "mean_early_area_plus_tv": float(selected_signals.mean().item()),
        "mean_full_area_plus_tv": float(selected_full_signals.mean().item()),
    }


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    model = build_model(checkpoint, args.device)
    autocast_context = get_autocast_context(args.device, args.dtype)
    early_k = max(2, model.num_blocks // 4)

    units = build_initial_units(model.num_blocks)
    rng = np.random.default_rng(args.seed + 101)
    generator = torch.Generator(device="cuda" if "cuda" in args.device else "cpu")
    generator.manual_seed(args.seed + 101)

    level_results = []
    for level_idx in range(1, int(args.num_levels) + 1):
        pair_score_matrix, top_pairs = mine_unit_pair_scores(
            model,
            units=units,
            tokens=tokens,
            batch_size=int(args.batch_size),
            num_batches=int(args.pair_mining_batches),
            pair_score_k=int(args.pair_score_k),
            tv_weight=float(args.tv_weight),
            device=args.device,
            autocast_context=autocast_context,
            seed=int(args.seed) + 1000 * level_idx,
            pair_eval_batch_size=int(args.pair_eval_batch_size),
        )
        aggregated_segments = aggregate_top_unit_pairs_to_segments(
            top_pairs,
            num_units=len(units),
            top_k=int(args.aggregate_top_k_pairs),
            max_segment_len=int(args.segment_len),
        )

        best_structured_orders = []
        best_random_orders = []
        l2r_orders = []
        best_structured_early = []
        best_random_early = []
        l2r_early = []
        best_structured_full = []
        best_random_full = []
        l2r_full = []
        top_pairs_for_generation = top_pairs[: int(args.top_pair_pool_size)]

        for batch_idx in range(1, int(args.num_batches) + 1):
            idx = sample_batch(tokens, int(args.batch_size), model.config.block_size, rng, args.device)
            structured_pool, random_pool = build_candidate_pool(
                units=units,
                pair_score_matrix=pair_score_matrix,
                top_pairs=top_pairs_for_generation,
                aggregated_segments=aggregated_segments,
                structured_pool_size=int(args.structured_pool_size),
                random_pool_size=int(args.random_pool_size),
                prefix_len=int(args.prefix_len),
                segment_len=int(args.segment_len),
                generator=generator,
                device=args.device,
            )
            l2r_pool = build_ascending_block_orders(1, model.num_blocks, args.device)

            def evaluate_pool(pool):
                tiled_orders = pool.unsqueeze(0).expand(idx.size(0), -1, -1)
                flat_orders = tiled_orders.reshape(idx.size(0) * pool.size(0), model.num_blocks)
                flat_idx = idx.unsqueeze(1).expand(idx.size(0), pool.size(0), idx.size(1)).reshape(
                    idx.size(0) * pool.size(0),
                    idx.size(1),
                )
                out = evaluate_orders_with_metrics(
                    model,
                    flat_idx,
                    flat_orders,
                    early_k=early_k,
                    tv_weight=float(args.tv_weight),
                    autocast_context=autocast_context,
                    eval_batch_size=int(args.candidate_eval_batch_size),
                )
                return {
                    "orders": tiled_orders,
                    "early_signal": out["early_area_plus_tv"].view(idx.size(0), pool.size(0)),
                    "full_signal": out["full_area_plus_tv"].view(idx.size(0), pool.size(0)),
                }

            structured_eval = evaluate_pool(structured_pool)
            random_eval = evaluate_pool(random_pool)
            l2r_eval = evaluate_pool(l2r_pool)
            structured_combined = structured_eval["early_signal"] + 1e-4 * structured_eval["full_signal"]
            random_combined = random_eval["early_signal"] + 1e-4 * random_eval["full_signal"]
            batch_indices = torch.arange(idx.size(0), device=idx.device)
            best_struct_idx = structured_combined.argmax(dim=1)
            best_rand_idx = random_combined.argmax(dim=1)

            best_structured_orders.append(structured_eval["orders"][batch_indices, best_struct_idx].cpu())
            best_random_orders.append(random_eval["orders"][batch_indices, best_rand_idx].cpu())
            l2r_orders.append(l2r_eval["orders"][:, 0].cpu())
            best_structured_early.append(structured_eval["early_signal"][batch_indices, best_struct_idx].cpu())
            best_random_early.append(random_eval["early_signal"][batch_indices, best_rand_idx].cpu())
            l2r_early.append(l2r_eval["early_signal"][:, 0].cpu())
            best_structured_full.append(structured_eval["full_signal"][batch_indices, best_struct_idx].cpu())
            best_random_full.append(random_eval["full_signal"][batch_indices, best_rand_idx].cpu())
            l2r_full.append(l2r_eval["full_signal"][:, 0].cpu())

            if batch_idx == 1 or batch_idx % max(1, int(args.log_every_batches)) == 0:
                print(
                    f"[level {level_idx}] batch {batch_idx}/{int(args.num_batches)} "
                    f"struct_best_early={torch.cat(best_structured_early).mean().item():.4f} "
                    f"rand_best_early={torch.cat(best_random_early).mean().item():.4f}"
                )

        best_structured_orders = torch.cat(best_structured_orders, dim=0)
        best_random_orders = torch.cat(best_random_orders, dim=0)
        l2r_orders = torch.cat(l2r_orders, dim=0)
        best_structured_early = torch.cat(best_structured_early, dim=0)
        best_random_early = torch.cat(best_random_early, dim=0)
        l2r_early = torch.cat(l2r_early, dim=0)
        best_structured_full = torch.cat(best_structured_full, dim=0)
        best_random_full = torch.cat(best_random_full, dim=0)
        l2r_full = torch.cat(l2r_full, dim=0)

        level_results.append(
            {
                "level": level_idx,
                "num_units": len(units),
                "units": units,
                "top_pairs": top_pairs[: min(50, len(top_pairs))],
                "aggregated_segments": [
                    {
                        "unit_indices": segment,
                        "blocks": flatten_unit_order(segment, units),
                        "length_units": len(segment),
                        "length_blocks": len(flatten_unit_order(segment, units)),
                    }
                    for segment in aggregated_segments
                ],
                "summaries": {
                    "best_structured": summarize_selected_orders(best_structured_orders, best_structured_early, best_structured_full, prefix_len=int(args.prefix_len)),
                    "best_random_pool": summarize_selected_orders(best_random_orders, best_random_early, best_random_full, prefix_len=int(args.prefix_len)),
                    "l2r_reference": summarize_selected_orders(l2r_orders, l2r_early, l2r_full, prefix_len=int(args.prefix_len)),
                },
            }
        )

        if level_idx < int(args.num_levels):
            units = build_next_level_units(units, aggregated_segments)

    payload = {
        "run_meta": {
            "ckpt_path": str(args.ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": args.split,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "pair_mining_batches": args.pair_mining_batches,
            "pair_eval_batch_size": args.pair_eval_batch_size,
            "candidate_eval_batch_size": args.candidate_eval_batch_size,
            "random_pool_size": args.random_pool_size,
            "structured_pool_size": args.structured_pool_size,
            "top_pair_pool_size": args.top_pair_pool_size,
            "aggregate_top_k_pairs": args.aggregate_top_k_pairs,
            "prefix_len": args.prefix_len,
            "segment_len": args.segment_len,
            "pair_score_k": args.pair_score_k,
            "num_levels": args.num_levels,
            "tv_weight": args.tv_weight,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "note": (
                "At each level, pair2 mining is done over the current unit vocabulary. "
                "Top directed unit pairs are aggregated into longer segments. "
                "Those segments become the units of the next level."
            ),
        },
        "levels": level_results,
    }
    save_json(args.out_dir / "results.json", payload)
    print(f"saved hierarchical structured benchmark to {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
