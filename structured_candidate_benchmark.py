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
    expand_block_orders_to_token_orders,
    kendall_tau_to_l2r_per_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine local pair structure from a checkpoint and build a structured candidate pool."
    )
    parser.add_argument("--ckpt_path", type=Path, default=Path("out/out-wikitext103-random-b32/ckpt.pt"))
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("Report/structured_candidate_benchmark_b32"),
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--pair_mining_batches", type=int, default=24)
    parser.add_argument("--pair_eval_batch_size", type=int, default=32)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--random_pool_size", type=int, default=64)
    parser.add_argument("--structured_pool_size", type=int, default=64)
    parser.add_argument("--top_pair_pool_size", type=int, default=128)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--segment_len", type=int, default=4)
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


def evaluate_orders_with_metrics(model, idx, block_orders, early_k, tv_weight, autocast_context):
    metrics = evaluate_block_order_quality(
        model,
        idx,
        block_orders,
        prefix_k=max(2, min(early_k, model.num_blocks)),
        block_len=model.block_order_block_len,
        autocast_context=autocast_context,
    )
    signal_metrics = compute_signal_from_block_losses(metrics["block_losses"], early_k=early_k, tv_weight=tv_weight)
    return {
        "block_losses": metrics["block_losses"],
        "kendall": metrics["kendall_per_sample"],
        "kendall_distance": (1.0 - metrics["kendall_per_sample"]) / 2.0,
        "early_area_plus_tv": signal_metrics["early_area_plus_tv"],
        "full_area_plus_tv": signal_metrics["full_area_plus_tv"],
    }


def build_random_suffix_orders(pairs, batch_size, num_blocks, device, generator):
    orders = []
    blocks = torch.arange(num_blocks, device=device)
    for pair in pairs:
        first, second = pair
        pair_orders = []
        mask = (blocks != first) & (blocks != second)
        remaining = blocks[mask]
        for _ in range(batch_size):
            perm = torch.randperm(remaining.numel(), generator=generator, device=device)
            suffix = remaining[perm]
            pair_orders.append(torch.cat([torch.tensor([first, second], device=device), suffix], dim=0))
        orders.append(torch.stack(pair_orders, dim=0))
    return torch.stack(orders, dim=1)  # (B, num_pairs, num_blocks)


def mine_pair_scores(model, tokens, batch_size, num_batches, early_k, tv_weight, device, autocast_context, seed, pair_eval_batch_size):
    num_blocks = model.num_blocks
    pair_list = [(i, j) for i in range(num_blocks) for j in range(num_blocks) if i != j]
    pair_score_sums = torch.zeros(len(pair_list), dtype=torch.float64)
    pair_score_counts = 0

    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cuda" if "cuda" in device else "cpu")
    generator.manual_seed(seed)

    chunk_size = max(1, int(pair_eval_batch_size))
    eval_chunk_size = max(1, int(pair_eval_batch_size))
    for _ in range(int(num_batches)):
        idx = sample_batch(tokens, batch_size, model.config.block_size, rng, device)
        for start in range(0, len(pair_list), chunk_size):
            chunk_pairs = pair_list[start : start + chunk_size]
            pair_orders = build_random_suffix_orders(
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
            score_parts = []
            for eval_start in range(0, flat_orders.size(0), eval_chunk_size):
                eval_end = min(flat_orders.size(0), eval_start + eval_chunk_size)
                eval_out = evaluate_orders_with_metrics(
                    model,
                    flat_idx[eval_start:eval_end],
                    flat_orders[eval_start:eval_end],
                    early_k=early_k,
                    tv_weight=tv_weight,
                    autocast_context=autocast_context,
                )
                score_parts.append(eval_out["early_area_plus_tv"].cpu())
            scores = torch.cat(score_parts, dim=0).view(idx.size(0), len(chunk_pairs)).mean(dim=0).double()
            pair_score_sums[start : start + len(chunk_pairs)] += scores
        pair_score_counts += 1

    pair_score_means = pair_score_sums / max(1, pair_score_counts)
    pair_score_matrix = torch.full((num_blocks, num_blocks), float("-inf"), dtype=torch.float64)
    for pair_idx, (i, j) in enumerate(pair_list):
        pair_score_matrix[i, j] = pair_score_means[pair_idx]

    top_pairs = sorted(
        [
            {
                "first": i,
                "second": j,
                "score": float(pair_score_matrix[i, j].item()),
                "margin_vs_reverse": float((pair_score_matrix[i, j] - pair_score_matrix[j, i]).item()),
            }
            for i in range(num_blocks)
            for j in range(num_blocks)
            if i != j
        ],
        key=lambda row: (row["score"], row["margin_vs_reverse"]),
        reverse=True,
    )
    return pair_score_matrix, top_pairs


def longest_contiguous_run(order):
    best = 1
    current = 1
    values = order.tolist()
    for i in range(1, len(values)):
        if values[i] == values[i - 1] + 1:
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


def build_pair_prefix_order(num_blocks, top_pairs, generator, device):
    pair = top_pairs[torch.randint(0, len(top_pairs), (1,), generator=generator, device=device).item()]
    used = {pair["first"], pair["second"]}
    remaining = [idx for idx in range(num_blocks) if idx not in used]
    remaining_tensor = torch.tensor(remaining, device=device)
    perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
    return torch.cat(
        [
            torch.tensor([pair["first"], pair["second"]], device=device),
            remaining_tensor[perm],
        ],
        dim=0,
    )


def build_chain_prefix_order(num_blocks, pair_score_matrix, top_pairs, segment_len, generator, device):
    seed_pair = top_pairs[torch.randint(0, len(top_pairs), (1,), generator=generator, device=device).item()]
    chain = [seed_pair["first"], seed_pair["second"]]
    used = set(chain)
    while len(chain) < min(segment_len, num_blocks):
        last = chain[-1]
        candidates = []
        for nxt in range(num_blocks):
            if nxt in used or nxt == last:
                continue
            score = pair_score_matrix[last, nxt].item()
            candidates.append((score, nxt))
        if not candidates:
            break
        candidates.sort(reverse=True)
        top_choices = [nxt for _, nxt in candidates[: min(4, len(candidates))]]
        nxt = top_choices[torch.randint(0, len(top_choices), (1,), generator=generator, device=device).item()]
        chain.append(nxt)
        used.add(nxt)
    remaining = [idx for idx in range(num_blocks) if idx not in used]
    remaining_tensor = torch.tensor(remaining, device=device)
    perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
    return torch.cat([torch.tensor(chain, device=device), remaining_tensor[perm]], dim=0)


def build_prefix_biased_order(num_blocks, pair_score_matrix, prefix_len, generator, device):
    prefix = []
    used = set()
    current = min(torch.randint(0, max(1, num_blocks // 4), (1,), generator=generator, device=device).item(), num_blocks - 1)
    prefix.append(current)
    used.add(current)
    while len(prefix) < min(prefix_len, num_blocks):
        last = prefix[-1]
        candidates = []
        for nxt in range(num_blocks):
            if nxt in used:
                continue
            front_bias = -0.05 * float(nxt)
            score = float(pair_score_matrix[last, nxt].item()) + front_bias
            candidates.append((score, nxt))
        candidates.sort(reverse=True)
        top_choices = [nxt for _, nxt in candidates[: min(6, len(candidates))]]
        nxt = top_choices[torch.randint(0, len(top_choices), (1,), generator=generator, device=device).item()]
        prefix.append(nxt)
        used.add(nxt)
    remaining = [idx for idx in range(num_blocks) if idx not in used]
    remaining_tensor = torch.tensor(remaining, device=device)
    perm = torch.randperm(remaining_tensor.numel(), generator=generator, device=device)
    return torch.cat([torch.tensor(prefix, device=device), remaining_tensor[perm]], dim=0)


def build_candidate_pool(num_blocks, pair_score_matrix, top_pairs, structured_pool_size, random_pool_size, prefix_len, segment_len, generator, device):
    top_pairs = top_pairs[: max(1, min(len(top_pairs), 128))]
    structured_orders = []
    per_group = max(1, structured_pool_size // 4)
    for _ in range(per_group):
        structured_orders.append(torch.randperm(num_blocks, generator=generator, device=device))
    for _ in range(per_group):
        structured_orders.append(build_pair_prefix_order(num_blocks, top_pairs, generator, device))
    for _ in range(per_group):
        structured_orders.append(build_chain_prefix_order(num_blocks, pair_score_matrix, top_pairs, segment_len, generator, device))
    while len(structured_orders) < structured_pool_size:
        structured_orders.append(build_prefix_biased_order(num_blocks, pair_score_matrix, prefix_len, generator, device))
    random_orders = [
        torch.randperm(num_blocks, generator=generator, device=device)
        for _ in range(random_pool_size)
    ]
    return torch.stack(structured_orders, dim=0), torch.stack(random_orders, dim=0)


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
    pair_score_matrix, top_pairs = mine_pair_scores(
        model,
        tokens=tokens,
        batch_size=int(args.batch_size),
        num_batches=int(args.pair_mining_batches),
        early_k=early_k,
        tv_weight=float(args.tv_weight),
        device=args.device,
        autocast_context=autocast_context,
        seed=int(args.seed),
        pair_eval_batch_size=int(args.pair_eval_batch_size),
    )

    rng = np.random.default_rng(args.seed + 101)
    generator = torch.Generator(device="cuda" if "cuda" in args.device else "cpu")
    generator.manual_seed(args.seed + 101)

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
            num_blocks=model.num_blocks,
            pair_score_matrix=pair_score_matrix,
            top_pairs=top_pairs_for_generation,
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
                f"[progress] batch {batch_idx}/{int(args.num_batches)} "
                f"structured_best_early={torch.cat(best_structured_early).mean().item():.4f} "
                f"random_best_early={torch.cat(best_random_early).mean().item():.4f}"
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

    summaries = {
        "best_structured": summarize_selected_orders(
            best_structured_orders,
            best_structured_early,
            best_structured_full,
            prefix_len=int(args.prefix_len),
        ),
        "best_random_pool": summarize_selected_orders(
            best_random_orders,
            best_random_early,
            best_random_full,
            prefix_len=int(args.prefix_len),
        ),
        "l2r_reference": summarize_selected_orders(
            l2r_orders,
            l2r_early,
            l2r_full,
            prefix_len=int(args.prefix_len),
        ),
    }

    top_pair_rows = top_pairs[: min(50, len(top_pairs))]
    run_meta = {
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
        "prefix_len": args.prefix_len,
        "segment_len": args.segment_len,
        "early_k": early_k,
        "tv_weight": args.tv_weight,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "note": (
            "Local structure is mined from ordered pairs under the frozen checkpoint. "
            "A structured candidate pool is then built from random, pair-biased, chain-biased, "
            "and prefix-biased proposals, and compared against a pure-random candidate pool."
        ),
    }
    payload = {
        "run_meta": run_meta,
        "summaries": summaries,
        "top_pairs": top_pair_rows,
    }
    save_json(args.out_dir / "results.json", payload)
    print(f"saved structured candidate benchmark to {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
