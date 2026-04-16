"""
Purpose:
Build order-unit signatures from reveal-trajectory statistics collected over
many random trajectories, then benchmark the resulting top-k pair proposals.

Typical usage:
python scripts/benchmark/trajectory_statistical_signature_benchmark.py \
  --ckpt_path out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt \
  --out_path Report/trajectory/trajectory_statistical_signature_benchmark/example.json \
  --raw_block_size 32 \
  --num_samples 16
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    get_order_unit_name,
    invert_permutation,
    token_losses_to_block_losses,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline benchmark for trajectory-statistical raw-block signatures "
            "and top-k candidate proposals."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument(
        "--raw_block_size",
        type=int,
        default=None,
        help="Evaluation unit size in tokens. Defaults to the checkpoint order unit size; token-level checkpoints therefore default to 1.",
    )
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=0)
    parser.add_argument("--sample_batch_size", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--sample_offsets", type=str, default="")
    parser.add_argument("--num_trajectories", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--local_response_horizon", type=int, default=4)
    parser.add_argument("--tv_weight", type=float, default=0.3)
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
    return REPO_ROOT / "data" / dataset


def load_tokens(data_dir: Path, split: str):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def parse_csv_int_list(raw_value: str):
    if not raw_value:
        return []
    return [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def resolve_seq_len(seq_len, model):
    if seq_len is None:
        return int(model.config.block_size)
    seq_len = int(seq_len)
    if seq_len <= 0 or seq_len > int(model.config.block_size):
        raise ValueError(
            f"seq_len must be in [1, {int(model.config.block_size)}], got {seq_len}."
        )
    return seq_len


def validate_raw_block_layout(seq_len, raw_block_size):
    if raw_block_size <= 0:
        raise ValueError(f"raw_block_size must be positive, got {raw_block_size}.")
    if seq_len % raw_block_size != 0:
        raise ValueError(
            f"seq_len={seq_len} must be divisible by raw_block_size={raw_block_size}."
        )
    return seq_len // raw_block_size


def resolve_raw_block_size(raw_block_size, model):
    if raw_block_size is not None:
        return int(raw_block_size)
    return int(model.block_order_block_len)


def load_block_permutation_from_checkpoint(checkpoint, seq_len, raw_block_size):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    if seq_len % raw_block_size != 0:
        raise ValueError(
            "When permute_data=True, seq_len must be divisible by raw_block_size so the "
            "fixed dataset permutation can be applied consistently."
        )
    num_blocks = seq_len // raw_block_size
    perm_state = checkpoint.get("data_permutation")
    if perm_state is not None and perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
        if block_perm.numel() != num_blocks:
            block_perm = block_perm[:num_blocks]
    else:
        permute_mode = str(config.get("permute_mode", "block"))
        if permute_mode != "block":
            raise ValueError(f"Unsupported permute_mode={permute_mode!r} in checkpoint config.")
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(block_perm, block_len=raw_block_size)
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def sample_offsets(tokens, seq_len, num_samples, seed, explicit_offsets, num_batches=0, sample_batch_size=0):
    if explicit_offsets:
        return explicit_offsets, None
    max_start = len(tokens) - seq_len
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than seq_len.")
    rng = np.random.default_rng(seed)
    if int(num_batches) > 0 and int(sample_batch_size) > 0:
        sampled_offset_batches = []
        for _ in range(int(num_batches)):
            batch_offsets = rng.integers(0, max_start, size=int(sample_batch_size)).tolist()
            sampled_offset_batches.append(batch_offsets)
        flat_offsets = [int(offset) for batch in sampled_offset_batches for offset in batch]
        return flat_offsets, sampled_offset_batches
    return rng.integers(0, max_start, size=int(num_samples)).tolist(), None


def load_sequences(tokens, offsets, seq_len, device, token_perm=None):
    sequences = []
    for offset in offsets:
        seq = torch.from_numpy(tokens[offset : offset + seq_len].astype(np.int64))
        if token_perm is not None:
            seq = seq[token_perm]
        sequences.append(seq)
    return torch.stack(sequences, dim=0).to(device)


def sample_random_raw_block_orders(batch_size, num_blocks, device, generator):
    return torch.stack(
        [torch.randperm(num_blocks, generator=generator, device=device) for _ in range(batch_size)],
        dim=0,
    )


def compute_phase_ids(num_blocks):
    phase_ids = []
    for step in range(num_blocks):
        ratio = float(step) / max(1, num_blocks)
        if ratio < 1.0 / 3.0:
            phase_ids.append(0)
        elif ratio < 2.0 / 3.0:
            phase_ids.append(1)
        else:
            phase_ids.append(2)
    return phase_ids


def compute_local_response_stats(block_losses, horizon, tv_weight):
    batch_size, num_blocks = block_losses.shape
    local_area = torch.zeros_like(block_losses, dtype=torch.float32)
    local_tv = torch.zeros_like(block_losses, dtype=torch.float32)
    local_combo = torch.zeros_like(block_losses, dtype=torch.float32)
    for step in range(num_blocks):
        window_end = min(num_blocks, step + max(1, int(horizon)))
        window = block_losses[:, step:window_end].float()
        area = -window.mean(dim=-1)
        if window.size(1) < 2:
            tv = torch.zeros_like(area)
        else:
            tv = -(window[:, 1:] - window[:, :-1]).abs().sum(dim=-1)
        combo = area + float(tv_weight) * tv
        local_area[:, step] = area
        local_tv[:, step] = tv
        local_combo[:, step] = combo
    centered_gain = local_combo - local_combo.mean(dim=1, keepdim=True)
    return {
        "local_area": local_area,
        "local_tv": local_tv,
        "local_combo": local_combo,
        "centered_gain": centered_gain,
    }


def build_trajectory_stat_accumulators(num_samples, num_blocks, device):
    return {
        "phase_gain_sum": torch.zeros(num_samples, num_blocks, 3, device=device, dtype=torch.float32),
        "phase_gain_count": torch.zeros(num_samples, num_blocks, 3, device=device, dtype=torch.float32),
        "area_sum": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "tv_sum": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "combo_sum": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "gain_sum": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "gain_sq_sum": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "positive_count": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "rank_sum": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "rank_sq_sum": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
        "event_count": torch.zeros(num_samples, num_blocks, device=device, dtype=torch.float32),
    }


def accumulate_trajectory_events(accumulators, block_orders, local_stats):
    num_samples, num_blocks = block_orders.shape
    phase_ids = compute_phase_ids(num_blocks)
    normalized_steps = torch.linspace(
        0.0,
        1.0,
        steps=num_blocks,
        device=block_orders.device,
        dtype=torch.float32,
    )

    for step in range(num_blocks):
        block_ids = block_orders[:, step : step + 1]
        phase = phase_ids[step]
        accumulators["phase_gain_sum"][:, :, phase].scatter_add_(
            1,
            block_ids,
            local_stats["centered_gain"][:, step : step + 1],
        )
        accumulators["phase_gain_count"][:, :, phase].scatter_add_(
            1,
            block_ids,
            torch.ones_like(local_stats["centered_gain"][:, step : step + 1]),
        )
        accumulators["area_sum"].scatter_add_(1, block_ids, local_stats["local_area"][:, step : step + 1])
        accumulators["tv_sum"].scatter_add_(1, block_ids, local_stats["local_tv"][:, step : step + 1])
        accumulators["combo_sum"].scatter_add_(1, block_ids, local_stats["local_combo"][:, step : step + 1])
        accumulators["gain_sum"].scatter_add_(1, block_ids, local_stats["centered_gain"][:, step : step + 1])
        accumulators["gain_sq_sum"].scatter_add_(
            1,
            block_ids,
            local_stats["centered_gain"][:, step : step + 1].pow(2),
        )
        accumulators["positive_count"].scatter_add_(
            1,
            block_ids,
            (local_stats["centered_gain"][:, step : step + 1] > 0).float(),
        )
        accumulators["rank_sum"].scatter_add_(
            1,
            block_ids,
            torch.full_like(local_stats["centered_gain"][:, step : step + 1], normalized_steps[step].item()),
        )
        accumulators["rank_sq_sum"].scatter_add_(
            1,
            block_ids,
            torch.full_like(local_stats["centered_gain"][:, step : step + 1], normalized_steps[step].item() ** 2),
        )
        accumulators["event_count"].scatter_add_(
            1,
            block_ids,
            torch.ones_like(local_stats["centered_gain"][:, step : step + 1]),
        )


def finalize_trajectory_stat_signatures(accumulators, eps=1e-6):
    event_count = accumulators["event_count"].clamp_min(1.0)
    gain_mean = accumulators["gain_sum"] / event_count
    gain_var = (accumulators["gain_sq_sum"] / event_count) - gain_mean.pow(2)
    gain_std = gain_var.clamp_min(0.0).sqrt()
    positive_gain_ratio = accumulators["positive_count"] / event_count
    rank_mean = accumulators["rank_sum"] / event_count
    rank_var = (accumulators["rank_sq_sum"] / event_count) - rank_mean.pow(2)
    rank_std = rank_var.clamp_min(0.0).sqrt()

    phase_means = []
    for phase_idx in range(3):
        phase_count = accumulators["phase_gain_count"][:, :, phase_idx].clamp_min(1.0)
        phase_means.append(accumulators["phase_gain_sum"][:, :, phase_idx] / phase_count)
    gain_early, gain_mid, gain_late = phase_means

    delay_regret = gain_early - gain_late
    local_area_mean = accumulators["area_sum"] / event_count
    local_tv_mean = accumulators["tv_sum"] / event_count
    local_combo_mean = accumulators["combo_sum"] / event_count

    signature = torch.stack(
        [
            gain_early,
            gain_mid,
            gain_late,
            delay_regret,
            local_area_mean,
            local_tv_mean,
            local_combo_mean,
            gain_std,
            positive_gain_ratio,
            rank_mean,
            rank_std,
        ],
        dim=-1,
    )
    return {
        "signature": signature,
        "fields": [
            "gain_early",
            "gain_mid",
            "gain_late",
            "delay_regret",
            "local_area_mean",
            "local_tv_mean",
            "local_area_plus_tv_mean",
            "gain_std",
            "positive_gain_ratio",
            "reveal_rank_mean",
            "reveal_rank_std",
        ],
        "phase_event_count": accumulators["phase_gain_count"],
        "event_count": accumulators["event_count"],
    }


def compute_similarity_matrix(signatures):
    normalized = F.normalize(signatures.float(), p=2, dim=-1)
    return normalized @ normalized.transpose(-1, -2)


def build_topk_from_similarity(similarity, topk):
    sim = similarity.clone()
    sim.fill_diagonal_(-float("inf"))
    k = min(max(1, int(topk)), max(1, sim.size(0) - 1))
    values, indices = torch.topk(sim, k=k, dim=-1)
    neighbors = []
    candidate_pairs = []
    for source in range(sim.size(0)):
        rows = []
        for rank in range(k):
            target = int(indices[source, rank].item())
            score = float(values[source, rank].item())
            rows.append(
                {
                    "target_block": target,
                    "similarity": score,
                    "rank": int(rank + 1),
                }
            )
            candidate_pairs.append(
                {
                    "source_block": int(source),
                    "target_block": target,
                    "similarity": score,
                }
            )
        neighbors.append({"source_block": int(source), "neighbors": rows})
    candidate_pairs.sort(key=lambda row: (row["source_block"], -row["similarity"], row["target_block"]))
    return neighbors, candidate_pairs


def map_block_index_to_original(block_idx, permutation_state):
    if permutation_state is None:
        return int(block_idx)
    return int(permutation_state["block_perm"][int(block_idx)].item())


def annotate_neighbors_with_original(neighbors, permutation_state):
    annotated = []
    for row in neighbors:
        source_block = int(row["source_block"])
        payload = {
            "source_block": source_block,
            "source_block_original": map_block_index_to_original(source_block, permutation_state),
            "neighbors": [],
        }
        for neighbor in row["neighbors"]:
            payload["neighbors"].append(
                {
                    **neighbor,
                    "target_block_original": map_block_index_to_original(neighbor["target_block"], permutation_state),
                }
            )
        annotated.append(payload)
    return annotated


def annotate_candidate_pairs_with_original(candidate_pairs, permutation_state):
    annotated = []
    for row in candidate_pairs:
        annotated.append(
            {
                **row,
                "source_block_original": map_block_index_to_original(row["source_block"], permutation_state),
                "target_block_original": map_block_index_to_original(row["target_block"], permutation_state),
            }
        )
    return annotated


def summarize_candidate_pairs(candidate_pairs, num_blocks):
    pairset_current = {(row["source_block"], row["target_block"]) for row in candidate_pairs}
    pairset_original = {
        (row["source_block_original"], row["target_block_original"]) for row in candidate_pairs
    }
    num_pairs = len(candidate_pairs)

    def summarize_pairset(pairset, use_original):
        if use_original:
            src_key = "source_block_original"
            tgt_key = "target_block_original"
        else:
            src_key = "source_block"
            tgt_key = "target_block"
        distances = [abs(row[tgt_key] - row[src_key]) for row in candidate_pairs]
        adjacent = sum(1 for row in candidate_pairs if abs(row[tgt_key] - row[src_key]) == 1)
        within2 = sum(1 for row in candidate_pairs if abs(row[tgt_key] - row[src_key]) <= 2)
        within4 = sum(1 for row in candidate_pairs if abs(row[tgt_key] - row[src_key]) <= 4)
        direct = [(i, i + 1) for i in range(num_blocks - 1) if (i, i + 1) in pairset]
        reverse = [(i + 1, i) for i in range(num_blocks - 1) if (i + 1, i) in pairset]
        mutual = sum(1 for a, b in pairset if a < b and (b, a) in pairset)
        forward = sum(1 for row in candidate_pairs if row[tgt_key] > row[src_key])
        return {
            "mean_neighbor_distance": float(np.mean(np.asarray(distances))) if distances else 0.0,
            "adjacent_ratio": float(adjacent / max(1, num_pairs)),
            "within2_ratio": float(within2 / max(1, num_pairs)),
            "within4_ratio": float(within4 / max(1, num_pairs)),
            "direct_i_to_i_plus_1_count": int(len(direct)),
            "direct_i_to_i_plus_1_pairs": direct,
            "reverse_i_plus_1_to_i_count": int(len(reverse)),
            "reverse_i_plus_1_to_i_pairs": reverse,
            "mutual_pair_count": int(mutual),
            "forward_pair_count": int(forward),
            "backward_pair_count": int(num_pairs - forward),
        }

    return {
        "num_pairs": int(num_pairs),
        "current_index_space": summarize_pairset(pairset_current, use_original=False),
        "original_index_space": summarize_pairset(pairset_original, use_original=True),
    }


def mean_summary(metric_rows):
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    summary = {}
    for key in keys:
        values = [row[key] for row in metric_rows]
        if isinstance(values[0], (int, float)):
            summary[key] = float(np.mean(np.asarray(values)))
    return summary


def chunk_list(values, chunk_size):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def run_benchmark(args):
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device)
    args.raw_block_size = resolve_raw_block_size(args.raw_block_size, model)
    unit_name = get_order_unit_name(args.raw_block_size)
    seq_len = resolve_seq_len(args.seq_len, model)
    num_blocks = validate_raw_block_layout(seq_len, int(args.raw_block_size))
    autocast_context = get_autocast_context(args.device, args.dtype)

    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    offsets, sampled_offset_batches = sample_offsets(
        tokens,
        seq_len=seq_len,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        explicit_offsets=parse_csv_int_list(args.sample_offsets),
        num_batches=int(args.num_batches),
        sample_batch_size=int(args.sample_batch_size),
    )

    permutation_state = load_block_permutation_from_checkpoint(
        checkpoint,
        seq_len=seq_len,
        raw_block_size=int(args.raw_block_size),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    generator = torch.Generator(device="cuda" if "cuda" in args.device else "cpu")
    generator.manual_seed(int(args.seed) + 17)

    stat_aggregate_sum = torch.zeros(num_blocks, num_blocks, dtype=torch.float64)
    stat_signature_sum = None
    stat_phase_count_sum = None
    stat_event_count_sum = None
    sample_metrics = []
    total_sample_count = 0
    batch_offsets_list = list(chunk_list(offsets, int(args.eval_batch_size)))
    for batch_idx, offset_chunk in enumerate(batch_offsets_list):
        idx = load_sequences(
            tokens,
            offset_chunk,
            seq_len=seq_len,
            device=args.device,
            token_perm=token_perm,
        )
        accumulators = build_trajectory_stat_accumulators(
            num_samples=idx.size(0),
            num_blocks=num_blocks,
            device=idx.device,
        )

        for trajectory_idx in range(int(args.num_trajectories)):
            block_orders = sample_random_raw_block_orders(
                batch_size=idx.size(0),
                num_blocks=num_blocks,
                device=idx.device,
                generator=generator,
            )
            token_orders = expand_block_orders_to_token_orders(
                block_orders,
                block_len=int(args.raw_block_size),
            )
            with torch.no_grad():
                with autocast_context:
                    outputs = model(
                        idx,
                        mode=None,
                        orders=token_orders,
                        return_token_loss=True,
                    )
            token_losses = outputs[2].float()
            block_losses = token_losses_to_block_losses(token_losses, block_len=int(args.raw_block_size))
            local_stats = compute_local_response_stats(
                block_losses,
                horizon=int(args.local_response_horizon),
                tv_weight=float(args.tv_weight),
            )
            accumulate_trajectory_events(accumulators, block_orders, local_stats)

        stat_payload = finalize_trajectory_stat_signatures(accumulators)
        stat_signature = stat_payload["signature"].cpu()
        if stat_signature_sum is None:
            stat_signature_sum = stat_signature.sum(dim=0)
        else:
            stat_signature_sum += stat_signature.sum(dim=0)
        batch_phase_count = stat_payload["phase_event_count"].sum(dim=0).cpu()
        if stat_phase_count_sum is None:
            stat_phase_count_sum = batch_phase_count
        else:
            stat_phase_count_sum += batch_phase_count
        batch_event_count = stat_payload["event_count"].sum(dim=0).cpu()
        if stat_event_count_sum is None:
            stat_event_count_sum = batch_event_count
        else:
            stat_event_count_sum += batch_event_count
        total_sample_count += idx.size(0)

        for local_sample_idx in range(idx.size(0)):
            stat_sim = compute_similarity_matrix(stat_signature[local_sample_idx]).cpu()
            stat_aggregate_sum += stat_sim.double()
            neighbors, candidate_pairs = build_topk_from_similarity(stat_sim, topk=int(args.topk))
            annotated_pairs = annotate_candidate_pairs_with_original(candidate_pairs, permutation_state)
            summary = summarize_candidate_pairs(annotated_pairs, num_blocks=num_blocks)
            sample_metrics.append(
                {
                    "adjacent_ratio_current": summary["current_index_space"]["adjacent_ratio"],
                    "adjacent_ratio_original": summary["original_index_space"]["adjacent_ratio"],
                    "within2_ratio_current": summary["current_index_space"]["within2_ratio"],
                    "within2_ratio_original": summary["original_index_space"]["within2_ratio"],
                    "mean_neighbor_distance_current": summary["current_index_space"]["mean_neighbor_distance"],
                    "mean_neighbor_distance_original": summary["original_index_space"]["mean_neighbor_distance"],
                    "direct_l2r_count_current": summary["current_index_space"]["direct_i_to_i_plus_1_count"],
                    "direct_l2r_count_original": summary["original_index_space"]["direct_i_to_i_plus_1_count"],
                }
            )

        print(
            f"[sample-batch {batch_idx + 1}/{len(batch_offsets_list)}] "
            f"processed_samples={total_sample_count}"
        )

    aggregate_similarity = stat_aggregate_sum / max(1, total_sample_count)
    neighbors, candidate_pairs = build_topk_from_similarity(aggregate_similarity.float(), topk=int(args.topk))
    annotated_pairs = annotate_candidate_pairs_with_original(candidate_pairs, permutation_state)
    annotated_neighbors = annotate_neighbors_with_original(neighbors, permutation_state)

    result = {
        "run_meta": {
            "ckpt_path": str(args.ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "data_dir": str(data_dir),
            "split": args.split,
            "seq_len": int(seq_len),
            "raw_block_size": int(args.raw_block_size),
            "order_unit": unit_name,
            "num_blocks": int(num_blocks),
            "num_samples": int(total_sample_count),
            "num_batches": int(args.num_batches),
            "sample_batch_size": int(args.sample_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "sample_offsets": [int(v) for v in offsets],
            "num_trajectories": int(args.num_trajectories),
            "topk": int(args.topk),
            "local_response_horizon": int(args.local_response_horizon),
            "tv_weight": float(args.tv_weight),
            "seed": int(args.seed),
            "device": args.device,
            "dtype": args.dtype,
            "signature_definition_note": (
                "Main method uses per-block event-local trajectory statistics aggregated "
                "across many reveal events: phase-specific centered gain, delay regret, "
                "local area/tv/area_plus_tv, gain stability, and reveal-rank statistics."
            ),
        },
        "trajectory_statistical": {
            "aggregate": {
                "summary": summarize_candidate_pairs(annotated_pairs, num_blocks=num_blocks),
                "candidate_pairs": annotated_pairs,
                "topk_neighbors": annotated_neighbors,
            },
            "sample_metric_mean": mean_summary(sample_metrics),
            "signature_fields": stat_payload["fields"],
            "signature_mean": (stat_signature_sum / max(1, total_sample_count)).tolist(),
            "phase_event_count_mean": (stat_phase_count_sum / max(1, total_sample_count)).tolist(),
            "event_count_mean": (stat_event_count_sum / max(1, total_sample_count)).tolist(),
        },
    }
    if permutation_state is not None:
        result["run_meta"]["data_permutation"] = {
            "block_perm": permutation_state["block_perm"].tolist(),
            "inverse_block_perm": permutation_state["inverse_block_perm"].tolist(),
        }
    if sampled_offset_batches is not None:
        result["run_meta"]["sampled_offsets_by_batch"] = [
            [int(offset) for offset in batch] for batch in sampled_offset_batches
        ]
    return result


def main():
    args = parse_args()
    payload = run_benchmark(args)
    save_json(args.out_path, payload)
    print(f"saved benchmark json to {args.out_path}")


if __name__ == "__main__":
    main()
