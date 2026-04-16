"""
Purpose:
Extract hidden-state-derived signatures for order units and use nearest-neighbor
similarity in signature space to propose sparse top-k local pair candidates.

Typical usage:
python scripts/benchmark/trajectory_signature_proposal.py \
  --ckpt_path out-wikitext103-random-b32-curriculum-permute-block/ckpt.pt \
  --out_path Report/trajectory/trajectory_signature_proposal/example.json \
  --raw_block_size 32 \
  --num_batches 8 \
  --batch_size 4
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
    get_order_unit_name,
    invert_permutation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract hidden-derived raw-block signatures and build sparse top-k "
            "pair proposals for each sample."
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
    parser.add_argument("--mode", type=str, default="AR", choices=["AR", "Random"])
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num_random_trials", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sample_offsets", type=str, default="")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save_signatures", action="store_true")
    parser.add_argument(
        "--save_samples",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to save per-sample proposal details in the output JSON.",
    )
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


def sample_offsets(tokens, seq_len, num_samples, seed, explicit_offsets):
    if explicit_offsets:
        return explicit_offsets
    max_start = len(tokens) - seq_len
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than seq_len.")
    rng = np.random.default_rng(seed)
    return rng.integers(0, max_start, size=int(num_samples)).tolist()


def sample_batched_offsets(tokens, seq_len, num_batches, batch_size, seed):
    max_start = len(tokens) - seq_len
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than seq_len.")
    rng = np.random.default_rng(seed)
    total = int(num_batches) * int(batch_size)
    flat = rng.integers(0, max_start, size=total).tolist()
    batched = []
    cursor = 0
    for _ in range(int(num_batches)):
        batched.append(flat[cursor : cursor + int(batch_size)])
        cursor += int(batch_size)
    return batched


def load_sequences(tokens, offsets, seq_len, device, token_perm=None):
    sequences = []
    for offset in offsets:
        seq = torch.from_numpy(tokens[offset : offset + seq_len].astype(np.int64))
        if token_perm is not None:
            seq = seq[token_perm]
        sequences.append(seq)
    return torch.stack(sequences, dim=0).to(device)


def build_ar_orders(batch_size, seq_len, device):
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def build_random_raw_block_orders(batch_size, seq_len, raw_block_size, device, generator):
    num_blocks = validate_raw_block_layout(seq_len, raw_block_size)
    block_offsets = torch.arange(raw_block_size, device=device, dtype=torch.long).view(1, 1, raw_block_size)
    orders = []
    for _ in range(batch_size):
        block_perm = torch.randperm(num_blocks, generator=generator, device=device)
        token_order = block_perm.view(num_blocks, 1) * raw_block_size + block_offsets[0]
        orders.append(token_order.reshape(seq_len))
    return torch.stack(orders, dim=0)


def run_hidden_trials(model, idx, mode, raw_block_size, num_random_trials, autocast_context, seed):
    batch_size, seq_len = idx.shape
    if mode == "AR":
        orders = build_ar_orders(batch_size, seq_len, idx.device)
        with torch.no_grad():
            with autocast_context:
                outputs = model(
                    idx,
                    mode=None,
                    orders=orders,
                    return_hidden=True,
                    hidden_return_mode="predictor",
                )
        return outputs[2].float().detach()

    generator = torch.Generator(device="cuda" if "cuda" in idx.device.type else "cpu")
    generator.manual_seed(int(seed))
    hidden_trials = []
    for _ in range(max(1, int(num_random_trials))):
        orders = build_random_raw_block_orders(
            batch_size=batch_size,
            seq_len=seq_len,
            raw_block_size=raw_block_size,
            device=idx.device,
            generator=generator,
        )
        with torch.no_grad():
            with autocast_context:
                outputs = model(
                    idx,
                    mode=None,
                    orders=orders,
                    return_hidden=True,
                    hidden_return_mode="predictor",
                )
        hidden_trials.append(outputs[2].float().detach())
    return torch.stack(hidden_trials, dim=0).mean(dim=0)


def build_block_signatures(predictor_hidden_states, raw_block_size):
    batch_size, seq_len, hidden_dim = predictor_hidden_states.shape
    num_blocks = validate_raw_block_layout(seq_len, raw_block_size)
    blocks = predictor_hidden_states.view(batch_size, num_blocks, raw_block_size, hidden_dim)
    block_mean = blocks.mean(dim=2)
    block_first = blocks[:, :, 0, :]
    block_last = blocks[:, :, -1, :]
    signatures = torch.cat([block_mean, block_first, block_last], dim=-1)
    return F.normalize(signatures, p=2, dim=-1)


def compute_topk_neighbors(signatures, topk):
    num_blocks = signatures.size(0)
    sim = signatures @ signatures.transpose(0, 1)
    sim.fill_diagonal_(-float("inf"))
    k = min(max(1, int(topk)), max(1, num_blocks - 1))
    values, indices = torch.topk(sim, k=k, dim=-1)

    neighbors = []
    candidate_pairs = []
    for source in range(num_blocks):
        rows = []
        for rank in range(k):
            target = int(indices[source, rank].item())
            score = float(values[source, rank].item())
            row = {
                "target_block": target,
                "similarity": score,
                "rank": int(rank + 1),
            }
            rows.append(row)
            candidate_pairs.append(
                {
                    "source_block": int(source),
                    "target_block": target,
                    "similarity": score,
                }
            )
        neighbors.append(
            {
                "source_block": int(source),
                "neighbors": rows,
            }
        )
    candidate_pairs.sort(
        key=lambda row: (row["source_block"], -row["similarity"], row["target_block"])
    )
    return sim, neighbors, candidate_pairs


def compute_topk_neighbors_from_similarity(sim, topk):
    sim = sim.clone()
    num_blocks = sim.size(0)
    sim.fill_diagonal_(-float("inf"))
    k = min(max(1, int(topk)), max(1, num_blocks - 1))
    values, indices = torch.topk(sim, k=k, dim=-1)

    neighbors = []
    candidate_pairs = []
    for source in range(num_blocks):
        rows = []
        for rank in range(k):
            target = int(indices[source, rank].item())
            score = float(values[source, rank].item())
            row = {
                "target_block": target,
                "similarity": score,
                "rank": int(rank + 1),
            }
            rows.append(row)
            candidate_pairs.append(
                {
                    "source_block": int(source),
                    "target_block": target,
                    "similarity": score,
                }
            )
        neighbors.append(
            {
                "source_block": int(source),
                "neighbors": rows,
            }
        )
    candidate_pairs.sort(
        key=lambda row: (row["source_block"], -row["similarity"], row["target_block"])
    )
    return neighbors, candidate_pairs


def tensor_to_rounded_nested_list(tensor, decimals=6):
    values = tensor.detach().cpu().tolist()
    return [[round(float(v), decimals) for v in row] for row in values]


def map_block_index_to_original(block_idx, permutation_state):
    if permutation_state is None:
        return int(block_idx)
    return int(permutation_state["block_perm"][int(block_idx)].item())


def annotate_neighbors_with_original(neighbors, permutation_state):
    if permutation_state is None:
        return neighbors
    annotated = []
    for row in neighbors:
        source_block = int(row["source_block"])
        annotated.append(
            {
                "source_block": source_block,
                "source_block_original": map_block_index_to_original(source_block, permutation_state),
                "neighbors": [
                    {
                        **neighbor,
                        "target_block_original": map_block_index_to_original(neighbor["target_block"], permutation_state),
                    }
                    for neighbor in row["neighbors"]
                ],
            }
        )
    return annotated


def annotate_candidate_pairs_with_original(candidate_pairs, permutation_state):
    if permutation_state is None:
        return candidate_pairs
    annotated = []
    for row in candidate_pairs:
        source_block = int(row["source_block"])
        target_block = int(row["target_block"])
        annotated.append(
            {
                **row,
                "source_block_original": map_block_index_to_original(source_block, permutation_state),
                "target_block_original": map_block_index_to_original(target_block, permutation_state),
            }
        )
    return annotated


def main():
    args = parse_args()
    save_samples = args.save_samples.lower() == "true"
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device)
    args.raw_block_size = resolve_raw_block_size(args.raw_block_size, model)
    unit_name = get_order_unit_name(args.raw_block_size)
    seq_len = resolve_seq_len(args.seq_len, model)
    num_blocks = validate_raw_block_layout(seq_len, int(args.raw_block_size))
    autocast_context = get_autocast_context(args.device, args.dtype)

    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    permutation_state = load_block_permutation_from_checkpoint(
        checkpoint,
        seq_len=seq_len,
        raw_block_size=int(args.raw_block_size),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    explicit_offsets = parse_csv_int_list(args.sample_offsets)
    use_batched_aggregate = not explicit_offsets and (
        int(args.num_batches) > 1 or int(args.batch_size) > 1
    )

    sample_payloads = []
    aggregate_similarity_sum = torch.zeros(num_blocks, num_blocks, dtype=torch.float64)
    aggregate_sample_count = 0
    batched_offsets = []

    if use_batched_aggregate:
        batched_offsets = sample_batched_offsets(
            tokens,
            seq_len=seq_len,
            num_batches=int(args.num_batches),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        for batch_idx, offsets in enumerate(batched_offsets):
            idx = load_sequences(tokens, offsets, seq_len=seq_len, device=args.device, token_perm=token_perm)
            hidden_states = run_hidden_trials(
                model,
                idx,
                mode=args.mode,
                raw_block_size=int(args.raw_block_size),
                num_random_trials=int(args.num_random_trials),
                autocast_context=autocast_context,
                seed=int(args.seed) + 17 + batch_idx,
            )
            block_signatures = build_block_signatures(hidden_states, raw_block_size=int(args.raw_block_size))
            for sample_idx in range(idx.size(0)):
                sim_matrix, neighbors, candidate_pairs = compute_topk_neighbors(
                    block_signatures[sample_idx],
                    topk=int(args.topk),
                )
                aggregate_similarity_sum += sim_matrix.double().cpu()
                aggregate_sample_count += 1
                payload = {
                    "sample_id": int(len(sample_payloads)),
                    "batch_id": int(batch_idx),
                    "in_batch_index": int(sample_idx),
                    "split": args.split,
                    "offset": int(offsets[sample_idx]),
                    "seq_len": int(seq_len),
                    "raw_block_size": int(args.raw_block_size),
                    "num_blocks": int(num_blocks),
                    "mode": args.mode,
                    "topk": int(args.topk),
                    "num_random_trials": int(args.num_random_trials if args.mode == "Random" else 1),
                    "candidate_pairs": annotate_candidate_pairs_with_original(candidate_pairs, permutation_state),
                    "topk_neighbors": annotate_neighbors_with_original(neighbors, permutation_state),
                }
                if args.save_signatures:
                    payload["signatures"] = tensor_to_rounded_nested_list(block_signatures[sample_idx])
                    payload["similarity_matrix"] = tensor_to_rounded_nested_list(sim_matrix)
                if save_samples:
                    sample_payloads.append(payload)
            print(
                f"[batch {batch_idx}] batch_size={len(offsets)} "
                f"aggregate_samples={aggregate_sample_count}"
            )
    else:
        offsets = sample_offsets(
            tokens,
            seq_len=seq_len,
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            explicit_offsets=explicit_offsets,
        )
        idx = load_sequences(tokens, offsets, seq_len=seq_len, device=args.device, token_perm=token_perm)
        hidden_states = run_hidden_trials(
            model,
            idx,
            mode=args.mode,
            raw_block_size=int(args.raw_block_size),
            num_random_trials=int(args.num_random_trials),
            autocast_context=autocast_context,
            seed=int(args.seed) + 17,
        )
        block_signatures = build_block_signatures(hidden_states, raw_block_size=int(args.raw_block_size))
        for sample_idx in range(idx.size(0)):
            sim_matrix, neighbors, candidate_pairs = compute_topk_neighbors(
                block_signatures[sample_idx],
                topk=int(args.topk),
            )
            aggregate_similarity_sum += sim_matrix.double().cpu()
            aggregate_sample_count += 1
            payload = {
                "sample_id": int(sample_idx),
                "split": args.split,
                "offset": int(offsets[sample_idx]),
                "seq_len": int(seq_len),
                "raw_block_size": int(args.raw_block_size),
                "num_blocks": int(num_blocks),
                "mode": args.mode,
                "topk": int(args.topk),
                "num_random_trials": int(args.num_random_trials if args.mode == "Random" else 1),
                "candidate_pairs": annotate_candidate_pairs_with_original(candidate_pairs, permutation_state),
                "topk_neighbors": annotate_neighbors_with_original(neighbors, permutation_state),
            }
            if args.save_signatures:
                payload["signatures"] = tensor_to_rounded_nested_list(block_signatures[sample_idx])
                payload["similarity_matrix"] = tensor_to_rounded_nested_list(sim_matrix)
            if save_samples:
                sample_payloads.append(payload)
            print(
                f"[sample {sample_idx}] offset={int(offsets[sample_idx])} "
                f"num_blocks={int(num_blocks)} candidate_pairs={len(candidate_pairs)}"
            )

    aggregate_similarity = aggregate_similarity_sum / max(1, aggregate_sample_count)
    aggregate_neighbors, aggregate_candidate_pairs = compute_topk_neighbors_from_similarity(
        aggregate_similarity.float(),
        topk=int(args.topk),
    )

    run_meta = {
        "ckpt_path": str(args.ckpt_path),
        "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
        "data_dir": str(data_dir),
        "split": args.split,
        "seq_len": int(seq_len),
        "raw_block_size": int(args.raw_block_size),
        "order_unit": unit_name,
        "num_blocks": int(num_blocks),
        "mode": args.mode,
        "topk": int(args.topk),
        "num_random_trials": int(args.num_random_trials if args.mode == "Random" else 1),
        "num_samples": int(aggregate_sample_count),
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "use_batched_aggregate": bool(use_batched_aggregate),
        "seed": int(args.seed),
        "device": args.device,
        "dtype": args.dtype,
        "save_signatures": bool(args.save_signatures),
        "save_samples": bool(save_samples),
    }
    if permutation_state is not None:
        run_meta["data_permutation"] = {
            "block_perm": permutation_state["block_perm"].tolist(),
            "inverse_block_perm": permutation_state["inverse_block_perm"].tolist(),
        }

    payload = {
        "run_meta": run_meta,
        "aggregate": {
            "num_sequences_averaged": int(aggregate_sample_count),
            "sampled_offsets_by_batch": batched_offsets if batched_offsets else None,
            "candidate_pairs": annotate_candidate_pairs_with_original(aggregate_candidate_pairs, permutation_state),
            "topk_neighbors": annotate_neighbors_with_original(aggregate_neighbors, permutation_state),
        },
    }
    if save_samples:
        payload["samples"] = sample_payloads
    if args.save_signatures:
        payload["aggregate"]["similarity_matrix"] = tensor_to_rounded_nested_list(aggregate_similarity.float())
    save_json(args.out_path, payload)
    print(f"saved proposal json to {args.out_path}")


if __name__ == "__main__":
    main()
