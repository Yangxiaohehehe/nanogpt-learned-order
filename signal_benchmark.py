import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    evaluate_block_order_quality,
    generate_adjacent_swap_candidates,
    prefix_position_stats,
    token_losses_to_block_losses,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark candidate order signals on a random-order AO-GPT checkpoint."
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("out-wikitext103-random/ckpt.pt"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("Report/signal_benchmark_generated"),
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument(
        "--candidate_eval_batch_size",
        type=int,
        default=0,
        help="Optional chunk size for candidate evaluation. 0 means evaluate all candidates at once.",
    )
    parser.add_argument(
        "--move_mode",
        type=str,
        default="adjacent",
        choices=["adjacent", "insert_front", "insert_anywhere"],
    )
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


def sample_random_block_orders(batch_size, num_blocks, device, generator):
    return torch.stack(
        [torch.randperm(num_blocks, generator=generator, device=device) for _ in range(batch_size)]
    )


def count_adjacent_pairs(order_row):
    values = order_row.tolist()
    count = 0
    for left, right in zip(values[:-1], values[1:]):
        if right == left + 1:
            count += 1
    return count


def tau_to_distance(tau):
    return (1.0 - tau) / 2.0


def generate_insert_front_candidates(block_orders):
    batch_size, num_blocks = block_orders.shape
    candidates = []
    for src_idx in range(1, num_blocks):
        remaining = torch.cat([block_orders[:, :src_idx], block_orders[:, src_idx + 1 :]], dim=1)
        moved = torch.cat([block_orders[:, src_idx : src_idx + 1], remaining], dim=1)
        candidates.append(moved)
    if not candidates:
        return block_orders.unsqueeze(1)
    return torch.stack(candidates, dim=1)


def generate_insert_anywhere_candidates(block_orders):
    batch_size, num_blocks = block_orders.shape
    candidates = []
    for src_idx in range(num_blocks):
        without_src = torch.cat([block_orders[:, :src_idx], block_orders[:, src_idx + 1 :]], dim=1)
        moved_token = block_orders[:, src_idx : src_idx + 1]
        for dst_idx in range(num_blocks):
            if dst_idx == src_idx:
                continue
            left = without_src[:, :dst_idx]
            right = without_src[:, dst_idx:]
            moved = torch.cat([left, moved_token, right], dim=1)
            candidates.append(moved)
    if not candidates:
        return block_orders.unsqueeze(1)
    return torch.stack(candidates, dim=1)


def generate_candidates(block_orders, move_mode):
    if move_mode == "adjacent":
        return generate_adjacent_swap_candidates(block_orders)
    if move_mode == "insert_front":
        return generate_insert_front_candidates(block_orders)
    if move_mode == "insert_anywhere":
        return generate_insert_anywhere_candidates(block_orders)
    raise ValueError(f"Unsupported move_mode: {move_mode}")


def compute_block_losses_for_orders(model, idx, block_orders, block_len, autocast_context):
    metrics = evaluate_block_order_quality(
        model,
        idx,
        block_orders,
        prefix_k=2,
        block_len=block_len,
        autocast_context=autocast_context,
    )
    return metrics["block_losses"], metrics


def build_signal_specs(num_blocks):
    early_k = min(4, num_blocks)
    late_start = early_k
    early_weights = torch.tensor([1.0, 0.7, 0.4, 0.2], dtype=torch.float32)[:early_k]

    def prefix_auc_4(block_losses):
        return -block_losses[:, :early_k].mean(dim=-1)

    def early_weighted_auc(block_losses):
        weighted = block_losses[:, :early_k] * early_weights.view(1, -1).to(block_losses.device)
        return -weighted.sum(dim=-1) / early_weights.sum().to(block_losses.device)

    def early_weighted_var(block_losses):
        early = block_losses[:, :early_k]
        weighted = early * early_weights.view(1, -1).to(block_losses.device)
        auc_term = -weighted.sum(dim=-1) / early_weights.sum().to(block_losses.device)
        var_term = early.var(dim=-1, unbiased=False)
        return auc_term - 0.3 * var_term

    def early_weighted_tv(block_losses):
        early = block_losses[:, :early_k]
        weighted = early * early_weights.view(1, -1).to(block_losses.device)
        auc_term = -weighted.sum(dim=-1) / early_weights.sum().to(block_losses.device)
        if early.size(1) < 2:
            tv_term = torch.zeros_like(auc_term)
        else:
            tv_term = (early[:, 1:] - early[:, :-1]).abs().sum(dim=-1)
        return auc_term - 0.3 * tv_term

    def early_drop_plus_weighted(block_losses):
        early = block_losses[:, :early_k]
        auc_term = early_weighted_auc(block_losses)
        if early.size(1) < 2:
            drop_term = torch.zeros_like(auc_term)
        else:
            drop_term = early[:, 0] - early[:, 1:].mean(dim=-1)
        return auc_term + 0.3 * drop_term

    def early_weighted_drop_late_penalty(block_losses):
        early = block_losses[:, :early_k]
        auc_term = early_weighted_auc(block_losses)
        if early.size(1) < 2:
            drop_term = torch.zeros_like(auc_term)
        else:
            drop_term = early[:, 0] - early[:, 1:].mean(dim=-1)
        if late_start >= block_losses.size(1):
            late_penalty = torch.zeros_like(auc_term)
        else:
            late_penalty = block_losses[:, late_start:].mean(dim=-1) - early.mean(dim=-1)
        return auc_term + 0.3 * drop_term + 0.3 * late_penalty

    return [
        ("prefix_auc_4", prefix_auc_4),
        ("early_weighted_auc", early_weighted_auc),
        ("early_weighted_auc_plus_variance", early_weighted_var),
        ("early_weighted_auc_plus_total_variation", early_weighted_tv),
        ("early_drop_plus_early_weighted_auc", early_drop_plus_weighted),
        ("early_weighted_auc_plus_early_drop_plus_late_penalty", early_weighted_drop_late_penalty),
    ]


@torch.no_grad()
def greedy_signal_search(
    model,
    idx,
    init_block_orders,
    signal_name,
    signal_fn,
    num_steps,
    block_len,
    move_mode,
    candidate_eval_batch_size,
    autocast_context,
):
    history = []
    current_orders = init_block_orders.clone()
    current_block_losses, current_metrics = compute_block_losses_for_orders(
        model,
        idx,
        current_orders,
        block_len,
        autocast_context,
    )
    current_quality = signal_fn(current_block_losses)
    history.append(
        {
            "step": 0,
            "orders": current_orders.clone(),
            "quality": current_quality.clone(),
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
        candidates = generate_candidates(current_orders, move_mode)
        batch_size, num_candidates, num_blocks = candidates.shape
        flat_candidates = candidates.reshape(batch_size * num_candidates, num_blocks)
        flat_idx = idx.unsqueeze(1).expand(batch_size, num_candidates, idx.size(1)).reshape(
            batch_size * num_candidates,
            idx.size(1),
        )
        chunk_size = int(candidate_eval_batch_size)
        if chunk_size <= 0:
            chunk_size = flat_candidates.size(0)
        quality_chunks = []
        for start in range(0, flat_candidates.size(0), chunk_size):
            end = min(flat_candidates.size(0), start + chunk_size)
            chunk_block_losses, _ = compute_block_losses_for_orders(
                model,
                flat_idx[start:end],
                flat_candidates[start:end],
                block_len,
                autocast_context,
            )
            quality_chunks.append(signal_fn(chunk_block_losses))
        candidate_quality = torch.cat(quality_chunks, dim=0).view(batch_size, num_candidates)
        best_candidate_idx = candidate_quality.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=current_orders.device)
        best_quality = candidate_quality[batch_indices, best_candidate_idx]
        improvement = best_quality - current_quality
        improved_mask = improvement > 0
        best_orders = candidates[batch_indices, best_candidate_idx]
        current_orders = torch.where(improved_mask.unsqueeze(-1), best_orders, current_orders)

        current_block_losses, current_metrics = compute_block_losses_for_orders(
            model,
            idx,
            current_orders,
            block_len,
            autocast_context,
        )
        current_quality = signal_fn(current_block_losses)
        history.append(
            {
                "step": step_idx,
                "orders": current_orders.clone(),
                "quality": current_quality.clone(),
                "prefix_auc_per_sample": current_metrics["prefix_auc_per_sample"].clone(),
                "full_loss_per_sample": current_metrics["full_loss_per_sample"].clone(),
                "kendall_per_sample": current_metrics["kendall_per_sample"].clone(),
                "swap_idx": torch.where(
                    improved_mask,
                    best_candidate_idx,
                    torch.full_like(best_candidate_idx, -1),
                ),
                "improvement": torch.where(
                    improved_mask,
                    improvement,
                    torch.zeros_like(improvement),
                ),
                "improved_mask": improved_mask.clone(),
            }
        )

    return {
        "signal": signal_name,
        "history": history,
        "final_orders": current_orders,
    }


def summarize_signal(signal_name, rows):
    initial_prefix2 = [row["initial_prefix_auc_2"] for row in rows]
    final_prefix2 = [row["final_prefix_auc_2"] for row in rows]
    initial_full = [row["initial_full_loss"] for row in rows]
    final_full = [row["final_full_loss"] for row in rows]
    initial_kendall = [row["initial_kendall_tau"] for row in rows]
    final_kendall = [row["final_kendall_tau"] for row in rows]
    initial_kendall_distance = [row["initial_kendall_distance"] for row in rows]
    final_kendall_distance = [row["final_kendall_distance"] for row in rows]
    initial_adjacent = [row["initial_adjacent_pairs"] for row in rows]
    final_adjacent = [row["final_adjacent_pairs"] for row in rows]
    initial_prefix_index = [row["initial_prefix_mean_index"] for row in rows]
    final_prefix_index = [row["final_prefix_mean_index"] for row in rows]
    improved = [1 if row["num_improving_steps"] > 0 else 0 for row in rows]

    return {
        "signal": signal_name,
        "move_mode": rows[0]["move_mode"] if rows else "",
        "num_samples": len(rows),
        "evaluation_note": (
            "kendall_distance and kendall_tau are evaluation-only diagnostics for l2r proximity; "
            "they must not be used as training targets or propagated as supervision signals."
        ),
        "mean_prefix_auc_2_initial": float(np.mean(initial_prefix2)),
        "mean_prefix_auc_2_final": float(np.mean(final_prefix2)),
        "mean_full_loss_initial": float(np.mean(initial_full)),
        "mean_full_loss_final": float(np.mean(final_full)),
        "mean_kendall_initial": float(np.mean(initial_kendall)),
        "mean_kendall_final": float(np.mean(final_kendall)),
        "mean_kendall_distance_initial": float(np.mean(initial_kendall_distance)),
        "mean_kendall_distance_final": float(np.mean(final_kendall_distance)),
        "mean_adjacent_pairs_initial": float(np.mean(initial_adjacent)),
        "mean_adjacent_pairs_final": float(np.mean(final_adjacent)),
        "mean_prefix_index_initial": float(np.mean(initial_prefix_index)),
        "mean_prefix_index_final": float(np.mean(final_prefix_index)),
        "mean_prefix_auc_2_gain": float(np.mean(np.asarray(initial_prefix2) - np.asarray(final_prefix2))),
        "mean_full_loss_gain": float(np.mean(np.asarray(initial_full) - np.asarray(final_full))),
        "mean_kendall_gain": float(np.mean(np.asarray(final_kendall) - np.asarray(initial_kendall))),
        "mean_kendall_distance_gain": float(
            np.mean(np.asarray(initial_kendall_distance) - np.asarray(final_kendall_distance))
        ),
        "mean_adjacent_pairs_gain": float(np.mean(np.asarray(final_adjacent) - np.asarray(initial_adjacent))),
        "mean_prefix_index_gain": float(np.mean(np.asarray(initial_prefix_index) - np.asarray(final_prefix_index))),
        "improved_sample_ratio": float(np.mean(improved)),
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    model = build_model(checkpoint, args.device)
    autocast_context = get_autocast_context(args.device, args.dtype)
    rng = np.random.default_rng(args.seed)
    order_generator = torch.Generator(device="cuda" if "cuda" in args.device else "cpu")
    order_generator.manual_seed(args.seed)

    signal_specs = build_signal_specs(model.num_blocks)
    all_rows = {signal_name: [] for signal_name, _ in signal_specs}

    for batch_idx in range(int(args.num_batches)):
        idx = sample_batch(tokens, args.batch_size, model.config.block_size, rng, args.device)
        init_block_orders = sample_random_block_orders(
            idx.size(0),
            model.num_blocks,
            idx.device,
            order_generator,
        )
        init_metrics = evaluate_block_order_quality(
            model,
            idx,
            init_block_orders,
            prefix_k=2,
            block_len=model.block_order_block_len,
            autocast_context=autocast_context,
        )
        init_prefix_stats = prefix_position_stats(init_block_orders, 4)

        for signal_name, signal_fn in signal_specs:
            result = greedy_signal_search(
                model=model,
                idx=idx,
                init_block_orders=init_block_orders,
                signal_name=signal_name,
                signal_fn=signal_fn,
                num_steps=args.num_steps,
                block_len=model.block_order_block_len,
                move_mode=args.move_mode,
                candidate_eval_batch_size=args.candidate_eval_batch_size,
                autocast_context=autocast_context,
            )
            final_orders = result["final_orders"]
            final_metrics = evaluate_block_order_quality(
                model,
                idx,
                final_orders,
                prefix_k=2,
                block_len=model.block_order_block_len,
                autocast_context=autocast_context,
            )
            final_prefix_stats = prefix_position_stats(final_orders, 4)

            for sample_idx in range(idx.size(0)):
                history = result["history"]
                all_rows[signal_name].append(
                    {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "signal": signal_name,
                        "move_mode": args.move_mode,
                        "initial_order": [int(v) for v in init_block_orders[sample_idx].detach().cpu().tolist()],
                        "final_order": [int(v) for v in final_orders[sample_idx].detach().cpu().tolist()],
                        "initial_prefix_auc_2": float(init_metrics["prefix_auc_per_sample"][sample_idx].item()),
                        "final_prefix_auc_2": float(final_metrics["prefix_auc_per_sample"][sample_idx].item()),
                        "initial_full_loss": float(init_metrics["full_loss_per_sample"][sample_idx].item()),
                        "final_full_loss": float(final_metrics["full_loss_per_sample"][sample_idx].item()),
                        "initial_kendall_tau": float(init_metrics["kendall_per_sample"][sample_idx].item()),
                        "final_kendall_tau": float(final_metrics["kendall_per_sample"][sample_idx].item()),
                        "initial_kendall_distance": float(
                            tau_to_distance(init_metrics["kendall_per_sample"][sample_idx].item())
                        ),
                        "final_kendall_distance": float(
                            tau_to_distance(final_metrics["kendall_per_sample"][sample_idx].item())
                        ),
                        "initial_adjacent_pairs": int(count_adjacent_pairs(init_block_orders[sample_idx].detach().cpu())),
                        "final_adjacent_pairs": int(count_adjacent_pairs(final_orders[sample_idx].detach().cpu())),
                        "initial_prefix_mean_index": float(init_block_orders[sample_idx, :4].float().mean().item()),
                        "final_prefix_mean_index": float(final_orders[sample_idx, :4].float().mean().item()),
                        "num_improving_steps": int(sum(int(step["improved_mask"][sample_idx].item()) for step in history[1:])),
                        "final_signal_quality": float(history[-1]["quality"][sample_idx].item()),
                    }
                )

    summaries = []
    for signal_name, rows in all_rows.items():
        rows_sorted = sorted(rows, key=lambda row: row["sample_idx"])
        save_json(args.out_dir / f"{signal_name}_rows.json", rows_sorted)
        summary = summarize_signal(signal_name, rows_sorted)
        summaries.append(summary)
        save_json(args.out_dir / f"{signal_name}_summary.json", summary)

    summaries = sorted(
        summaries,
        key=lambda row: (
            -row["mean_kendall_distance_gain"],
            -row["mean_kendall_gain"],
            row["mean_prefix_auc_2_final"],
        ),
    )
    save_json(args.out_dir / "ranking.json", summaries)

    run_meta = {
        "ckpt_path": str(args.ckpt_path),
        "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_steps": args.num_steps,
        "candidate_eval_batch_size": args.candidate_eval_batch_size,
        "move_mode": args.move_mode,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "signals": [name for name, _ in signal_specs],
        "evaluation_policy": {
            "primary_metric": "kendall_distance",
            "secondary_metric": "kendall_tau",
            "note": (
                "These metrics are evaluation-only for measuring proximity to l2r. "
                "They are not training targets and must not be propagated as supervision."
            ),
        },
    }
    save_json(args.out_dir / "run_meta.json", run_meta)

    print(f"saved signal benchmark outputs to {args.out_dir}")
    for row in summaries:
        print(
            f"{row['signal']}: "
            f"kendall_distance_gain={row['mean_kendall_distance_gain']:.6f}, "
            f"kendall_gain={row['mean_kendall_gain']:.6f}, "
            f"prefix_auc_2_gain={row['mean_prefix_auc_2_gain']:.6f}, "
            f"improved_ratio={row['improved_sample_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
