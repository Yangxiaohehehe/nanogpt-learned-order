import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from AOGPT import AOGPT, AOGPTConfig
from order_utils import evaluate_block_order_quality, kendall_tau_to_l2r_per_sample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark staged order construction: prefix-only, suffix-only, or two-stage."
    )
    parser.add_argument("--experiment", type=str, default="prefix_only", choices=["prefix_only", "suffix_only", "two_stage"])
    parser.add_argument("--ckpt_path", type=Path, default=Path("out/out-wikitext103-random-b32/ckpt.pt"))
    parser.add_argument("--out_dir", type=Path, default=Path("Report/staged_order_benchmark"))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--expand_candidates", type=int, default=8)
    parser.add_argument("--rollout_repeats", type=int, default=4)
    parser.add_argument("--prefix_eval_window", type=int, default=8)
    parser.add_argument("--suffix_eval_window", type=int, default=8)
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--suffix_prefix_source", type=str, default="random", choices=["random", "l2r"])
    parser.add_argument("--log_every_batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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


def prefix_mean_index(order, prefix_len):
    return float(np.mean(order[:prefix_len]))


def adjacent_correct_pairs(order):
    return int(sum(1 for a, b in zip(order[:-1], order[1:]) if b == a + 1))


def longest_contiguous_run(order):
    best = 1
    cur = 1
    for a, b in zip(order[:-1], order[1:]):
        if b == a + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return int(best)


def area_plus_tv_from_block_losses(block_losses, start, window, tv_weight):
    num_steps = block_losses.size(1)
    start = max(0, min(int(start), num_steps - 1))
    end = max(start + 1, min(num_steps, start + int(window)))
    segment = block_losses[:, start:end].float()
    area_term = -segment.mean(dim=-1)
    if segment.size(1) < 2:
        tv_term = torch.zeros_like(area_term)
    else:
        tv_term = (segment[:, 1:] - segment[:, :-1]).abs().sum(dim=-1)
    return area_term - float(tv_weight) * tv_term


def random_completion(prefix, num_blocks, rng):
    remaining = [b for b in range(num_blocks) if b not in prefix]
    perm = rng.permutation(remaining).tolist()
    return prefix + perm


def evaluate_order_list(model, idx_single, orders, block_len, candidate_eval_batch_size, autocast_context):
    device = idx_single.device
    order_tensor = torch.tensor(orders, device=device, dtype=torch.long)
    idx_batch = idx_single.unsqueeze(0).expand(order_tensor.size(0), -1)
    chunk = max(1, int(candidate_eval_batch_size))
    block_losses_all = []
    kendall_all = []
    for start in range(0, order_tensor.size(0), chunk):
        end = min(order_tensor.size(0), start + chunk)
        metrics = evaluate_block_order_quality(
            model,
            idx_batch[start:end],
            order_tensor[start:end],
            prefix_k=2,
            block_len=block_len,
            autocast_context=autocast_context,
        )
        block_losses_all.append(metrics["block_losses"].detach().cpu())
        kendall_all.append(metrics["kendall_per_sample"].detach().cpu())
    block_losses = torch.cat(block_losses_all, dim=0)
    kendall = torch.cat(kendall_all, dim=0)
    return block_losses, kendall


def choose_candidates(remaining, expand_candidates, rng):
    if len(remaining) <= expand_candidates:
        return list(remaining)
    choice = rng.choice(np.asarray(remaining), size=int(expand_candidates), replace=False)
    return [int(v) for v in choice.tolist()]


def beam_construct(
    model,
    idx_single,
    num_blocks,
    block_len,
    rng,
    beam_width,
    target_len,
    expand_candidates,
    rollout_repeats,
    score_start,
    score_window,
    tv_weight,
    candidate_eval_batch_size,
    autocast_context,
    fixed_prefix=None,
):
    initial_prefix = [] if fixed_prefix is None else list(fixed_prefix)
    beams = [{"prefix": initial_prefix, "score": 0.0}]
    while len(beams[0]["prefix"]) < int(target_len):
        next_beams = []
        for beam in beams:
            prefix = beam["prefix"]
            remaining = [b for b in range(num_blocks) if b not in prefix]
            if not remaining:
                next_beams.append({"prefix": prefix, "score": beam["score"]})
                continue
            for candidate in choose_candidates(remaining, expand_candidates, rng):
                new_prefix = prefix + [candidate]
                completed_orders = [random_completion(new_prefix, num_blocks, rng) for _ in range(int(rollout_repeats))]
                block_losses, _ = evaluate_order_list(
                    model,
                    idx_single,
                    completed_orders,
                    block_len,
                    candidate_eval_batch_size,
                    autocast_context,
                )
                scores = area_plus_tv_from_block_losses(
                    block_losses,
                    start=score_start,
                    window=score_window,
                    tv_weight=tv_weight,
                )
                next_beams.append(
                    {
                        "prefix": new_prefix,
                        "score": float(scores.mean().item()),
                    }
                )
        next_beams.sort(key=lambda row: row["score"], reverse=True)
        beams = next_beams[: max(1, int(beam_width))]
    return beams[0]["prefix"], beams


def build_suffix_from_prefix(
    model,
    idx_single,
    prefix,
    num_blocks,
    block_len,
    rng,
    beam_width,
    expand_candidates,
    rollout_repeats,
    score_window,
    tv_weight,
    candidate_eval_batch_size,
    autocast_context,
):
    best_full_prefix, beams = beam_construct(
        model=model,
        idx_single=idx_single,
        num_blocks=num_blocks,
        block_len=block_len,
        rng=rng,
        beam_width=beam_width,
        target_len=num_blocks,
        expand_candidates=expand_candidates,
        rollout_repeats=rollout_repeats,
        score_start=len(prefix),
        score_window=score_window,
        tv_weight=tv_weight,
        candidate_eval_batch_size=candidate_eval_batch_size,
        autocast_context=autocast_context,
        fixed_prefix=prefix,
    )
    return best_full_prefix, beams


def summarize_orders(name, rows):
    mean_tau = float(np.mean([row["kendall_tau"] for row in rows]))
    return {
        "name": name,
        "num_samples": len(rows),
        "mean_prefix_mean_index": float(np.mean([row["prefix_mean_index"] for row in rows])),
        "mean_adjacent_pairs": float(np.mean([row["adjacent_pairs"] for row in rows])),
        "mean_longest_run": float(np.mean([row["longest_run"] for row in rows])),
        "mean_kendall_tau": mean_tau,
        "mean_kendall_distance": float((1.0 - mean_tau) / 2.0),
        "mean_area_plus_tv": float(np.mean([row["area_plus_tv"] for row in rows])),
    }


def maybe_print_progress(batch_idx, num_batches, log_every_batches, full_rows, random_rows, l2r_rows):
    if int(log_every_batches) <= 0:
        return
    current_batch = int(batch_idx) + 1
    if current_batch % int(log_every_batches) != 0 and current_batch != int(num_batches):
        return
    print(f"[progress] batch {current_batch}/{int(num_batches)}")
    if full_rows:
        best_summary = summarize_orders("best", full_rows)
        print(
            "  best:"
            f" kendall_tau={best_summary['mean_kendall_tau']:.4f}"
            f" kendall_distance={best_summary['mean_kendall_distance']:.4f}"
            f" prefix_mean_index={best_summary['mean_prefix_mean_index']:.4f}"
            f" adjacent_pairs={best_summary['mean_adjacent_pairs']:.4f}"
            f" longest_run={best_summary['mean_longest_run']:.4f}"
            f" area_plus_tv={best_summary['mean_area_plus_tv']:.4f}"
        )
    if random_rows:
        random_summary = summarize_orders("random", random_rows)
        print(
            "  random:"
            f" kendall_tau={random_summary['mean_kendall_tau']:.4f}"
            f" kendall_distance={random_summary['mean_kendall_distance']:.4f}"
            f" prefix_mean_index={random_summary['mean_prefix_mean_index']:.4f}"
            f" adjacent_pairs={random_summary['mean_adjacent_pairs']:.4f}"
            f" longest_run={random_summary['mean_longest_run']:.4f}"
            f" area_plus_tv={random_summary['mean_area_plus_tv']:.4f}"
        )
    if l2r_rows:
        l2r_summary = summarize_orders("l2r", l2r_rows)
        print(
            "  l2r:"
            f" kendall_tau={l2r_summary['mean_kendall_tau']:.4f}"
            f" kendall_distance={l2r_summary['mean_kendall_distance']:.4f}"
            f" prefix_mean_index={l2r_summary['mean_prefix_mean_index']:.4f}"
            f" adjacent_pairs={l2r_summary['mean_adjacent_pairs']:.4f}"
            f" longest_run={l2r_summary['mean_longest_run']:.4f}"
            f" area_plus_tv={l2r_summary['mean_area_plus_tv']:.4f}"
        )


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    model = build_model(checkpoint, args.device)
    rng = np.random.default_rng(args.seed)
    autocast_context = get_autocast_context(args.device, args.dtype)

    prefix_rows = []
    suffix_rows = []
    full_rows = []
    random_rows = []
    l2r_rows = []

    for batch_idx in range(int(args.num_batches)):
        X = sample_batch(tokens, args.batch_size, model.config.block_size, rng, args.device)
        for sample_idx in range(X.size(0)):
            idx_single = X[sample_idx]
            num_blocks = model.num_blocks
            random_order = rng.permutation(num_blocks).tolist()
            l2r_order = list(range(num_blocks))

            if args.experiment in ("prefix_only", "two_stage"):
                best_prefix, _ = beam_construct(
                    model=model,
                    idx_single=idx_single,
                    num_blocks=num_blocks,
                    block_len=model.block_order_block_len,
                    rng=rng,
                    beam_width=args.beam_width,
                    target_len=args.prefix_len,
                    expand_candidates=args.expand_candidates,
                    rollout_repeats=args.rollout_repeats,
                    score_start=0,
                    score_window=args.prefix_eval_window,
                    tv_weight=args.tv_weight,
                    candidate_eval_batch_size=args.candidate_eval_batch_size,
                    autocast_context=autocast_context,
                )
            else:
                if args.suffix_prefix_source == "l2r":
                    best_prefix = l2r_order[: args.prefix_len]
                else:
                    best_prefix = random_order[: args.prefix_len]

            if args.experiment == "prefix_only":
                best_order = best_prefix + [b for b in random_order if b not in best_prefix]
            elif args.experiment == "suffix_only":
                best_order, _ = build_suffix_from_prefix(
                    model=model,
                    idx_single=idx_single,
                    prefix=best_prefix,
                    num_blocks=num_blocks,
                    block_len=model.block_order_block_len,
                    rng=rng,
                    beam_width=args.beam_width,
                    expand_candidates=args.expand_candidates,
                    rollout_repeats=args.rollout_repeats,
                    score_window=args.suffix_eval_window,
                    tv_weight=args.tv_weight,
                    candidate_eval_batch_size=args.candidate_eval_batch_size,
                    autocast_context=autocast_context,
                )
            else:
                best_order, _ = build_suffix_from_prefix(
                    model=model,
                    idx_single=idx_single,
                    prefix=best_prefix,
                    num_blocks=num_blocks,
                    block_len=model.block_order_block_len,
                    rng=rng,
                    beam_width=args.beam_width,
                    expand_candidates=args.expand_candidates,
                    rollout_repeats=args.rollout_repeats,
                    score_window=args.suffix_eval_window,
                    tv_weight=args.tv_weight,
                    candidate_eval_batch_size=args.candidate_eval_batch_size,
                    autocast_context=autocast_context,
                )

            order_list = {
                "random": random_order,
                "l2r": l2r_order,
                "best": best_order,
            }
            for key, order in order_list.items():
                block_losses, kendall = evaluate_order_list(
                    model,
                    idx_single,
                    [order],
                    model.block_order_block_len,
                    args.candidate_eval_batch_size,
                    autocast_context,
                )
                row = {
                    "order_prefix": order[:8],
                    "prefix_mean_index": prefix_mean_index(order, args.prefix_len),
                    "adjacent_pairs": adjacent_correct_pairs(order),
                    "longest_run": longest_contiguous_run(order),
                    "kendall_tau": float(kendall[0].item()),
                    "area_plus_tv": float(
                        area_plus_tv_from_block_losses(
                            block_losses,
                            start=0,
                            window=args.prefix_eval_window,
                            tv_weight=args.tv_weight,
                        )[0].item()
                    ),
                }
                if key == "random":
                    random_rows.append(row)
                elif key == "l2r":
                    l2r_rows.append(row)
                else:
                    full_rows.append(row)
                    prefix_rows.append(
                        {
                            "order_prefix": order[: args.prefix_len],
                            "prefix_mean_index": prefix_mean_index(order, args.prefix_len),
                            "adjacent_pairs": adjacent_correct_pairs(order[: args.prefix_len]),
                            "longest_run": longest_contiguous_run(order[: args.prefix_len]),
                            "kendall_tau": float(
                                kendall_tau_to_l2r_per_sample(
                                    torch.tensor(order[: args.prefix_len], device=args.device).view(1, -1)
                                )[0].item()
                            ),
                            "area_plus_tv": row["area_plus_tv"],
                        }
                    )
                    suffix_rows.append(
                        {
                            "order_prefix": order[args.prefix_len : args.prefix_len + 8],
                            "prefix_mean_index": float(np.mean(order[args.prefix_len :])),
                            "adjacent_pairs": adjacent_correct_pairs(order[args.prefix_len :]),
                            "longest_run": longest_contiguous_run(order[args.prefix_len :]),
                            "kendall_tau": row["kendall_tau"],
                            "area_plus_tv": float(
                                area_plus_tv_from_block_losses(
                                    block_losses,
                                    start=args.prefix_len,
                                    window=args.suffix_eval_window,
                                    tv_weight=args.tv_weight,
                                )[0].item()
                            ),
                        }
                    )
        maybe_print_progress(
            batch_idx=batch_idx,
            num_batches=args.num_batches,
            log_every_batches=args.log_every_batches,
            full_rows=full_rows,
            random_rows=random_rows,
            l2r_rows=l2r_rows,
        )

    results = {
        "experiment": args.experiment,
        "run_meta": {
            "ckpt_path": str(args.ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": args.split,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "prefix_len": args.prefix_len,
            "beam_width": args.beam_width,
            "expand_candidates": args.expand_candidates,
            "rollout_repeats": args.rollout_repeats,
            "prefix_eval_window": args.prefix_eval_window,
            "suffix_eval_window": args.suffix_eval_window,
            "tv_weight": args.tv_weight,
            "suffix_prefix_source": args.suffix_prefix_source,
        },
        "summaries": {
            "best_prefix_or_order": summarize_orders("best", full_rows),
            "random_baseline": summarize_orders("random", random_rows),
            "l2r_reference": summarize_orders("l2r", l2r_rows),
            "prefix_view": summarize_orders("prefix_view", prefix_rows),
            "suffix_view": summarize_orders("suffix_view", suffix_rows),
        },
    }
    save_json(args.out_dir / "results.json", results)
    print(f"saved staged order benchmark to {args.out_dir}")
    for key, value in results["summaries"].items():
        print(key, value)


if __name__ == "__main__":
    main()
