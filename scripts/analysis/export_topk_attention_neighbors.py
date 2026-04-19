"""
Export top-k attention neighbors for each token/block under a frozen AO-GPT checkpoint.

This script mirrors the attention-pruning stage of structured_candidate_benchmark.py,
but stops after aggregating the attention graph and extracting the top-k neighbors.
It does not run pair scoring or candidate-order benchmarking.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig
from order_utils import (
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    invert_permutation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate AO-GPT attention over multiple samples and export the top-k "
            "neighbors for every token/block."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=24)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--attn_mode", type=str, default="Random", choices=["AR", "Random"])
    parser.add_argument(
        "--attn_export_type",
        type=str,
        default="with_none",
        choices=["with_none", "without_none"],
        help="Which AO-GPT attention view to aggregate before taking top-k neighbors.",
    )
    parser.add_argument(
        "--attn_symmetrize",
        type=str,
        default="mean",
        choices=["mean", "max", "none"],
        help="How to form the neighbor graph from the aggregated original-frame attention matrix.",
    )
    parser.add_argument(
        "--save_attention_matrices",
        action="store_true",
        help="Also save aggregated attention matrices into results.json.",
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


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str, token_perm=None):
    max_start = len(tokens) - block_size
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than block_size.")
    starts = rng.integers(0, max_start, size=batch_size)
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    )
    batch = batch.to(device)
    if token_perm is not None:
        batch = batch[:, token_perm.to(device)]
    return batch, [int(v) for v in starts.tolist()]


def load_block_permutation_from_checkpoint(checkpoint, num_blocks, block_len):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    perm_state = checkpoint.get("data_permutation")
    if perm_state is not None and perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
    else:
        permute_mode = str(config.get("permute_mode", "block"))
        if permute_mode != "block":
            raise ValueError(f"Unsupported permute_mode={permute_mode!r} in checkpoint config.")
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(block_perm, block_len=block_len)
    return {
        "permute_mode": "block",
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def build_orders_for_mode(model, idx, mode):
    if mode == "AR":
        token_orders = model.set_ascending_orders(idx)
    elif mode == "Random":
        token_orders = model.sample_random_orders(idx)
    else:
        raise ValueError(f"Unsupported attention mining mode: {mode}")
    block_len = model.block_order_block_len
    block_orders = token_orders.view(token_orders.size(0), model.num_blocks, block_len)[:, :, 0] // block_len
    return token_orders, block_orders


def extract_attentions_from_outputs(outputs):
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Expected model(..., return_attentions=True) to return a tuple/list.")
    for candidate in reversed(outputs[2:]):
        if isinstance(candidate, (list, tuple)) and candidate:
            if all(torch.is_tensor(item) and item.ndim == 4 for item in candidate):
                return candidate
    raise RuntimeError("Could not find attention tensors in model outputs.")


def reduce_attention_batch(attn_outputs):
    layers = [layer_att.detach().float() for layer_att in attn_outputs]
    stacked = torch.stack(layers, dim=0).mean(dim=0)
    return stacked.mean(dim=1)


def aggregate_real_token_attention_to_block(attn_2d, block_len):
    real_token_attn = attn_2d[1:, 1:]
    num_real_positions = real_token_attn.size(0)
    if num_real_positions % block_len != 0:
        raise ValueError(
            f"Real-token attention length {num_real_positions} is not divisible by block_len={block_len}."
        )
    num_blocks = num_real_positions // block_len
    return real_token_attn.view(num_blocks, block_len, num_blocks, block_len).mean(dim=(1, 3))


def aggregate_predictor_aligned_attention_to_block(attn_2d, block_len):
    shifted_attn = attn_2d[:-1, :-1]
    num_predictor_positions = shifted_attn.size(0)
    if num_predictor_positions % block_len != 0:
        raise ValueError(
            f"Predictor-aligned attention length {num_predictor_positions} is not divisible by block_len={block_len}."
        )
    num_blocks = num_predictor_positions // block_len
    return shifted_attn.view(num_blocks, block_len, num_blocks, block_len).mean(dim=(1, 3))


def reorder_block_attention_to_original(block_matrix, block_order):
    if block_order.device != block_matrix.device:
        block_order = block_order.to(device=block_matrix.device)
    inverse = invert_permutation(block_order)
    return block_matrix[inverse][:, inverse]


def map_neighbor_map_to_original(neighbor_map, block_perm):
    mapped = {}
    for block_idx_str, neighbors in neighbor_map.items():
        block_idx = int(block_idx_str)
        original_key = str(int(block_perm[block_idx].item()))
        mapped[original_key] = [
            {
                **row,
                "neighbor_original": int(block_perm[int(row["neighbor"])].item()),
            }
            for row in neighbors
        ]
    return mapped


def annotate_neighbor_map_with_original(neighbor_map, block_perm):
    annotated = {}
    for block_idx_str, neighbors in neighbor_map.items():
        block_idx = int(block_idx_str)
        annotated[block_idx_str] = {
            "block_current": block_idx,
            "block_original": int(block_perm[block_idx].item()),
            "neighbors": [
                {
                    **row,
                    "neighbor_original": int(block_perm[int(row["neighbor"])].item()),
                }
                for row in neighbors
            ],
        }
    return annotated


def map_edge_rows_to_original(rows, block_perm):
    return [
        {
            **row,
            "first_original": int(block_perm[int(row["first"])].item()),
            "second_original": int(block_perm[int(row["second"])].item()),
        }
        for row in rows
    ]


def build_topk_neighbors(attention_graph, top_k):
    num_blocks = attention_graph.size(0)
    masked = attention_graph.clone()
    masked.fill_diagonal_(float("-inf"))
    top_k = max(1, min(int(top_k), num_blocks - 1))

    neighbor_map = {}
    undirected_edges = set()
    directed_edges = []
    for block_idx in range(num_blocks):
        values, indices = torch.topk(masked[block_idx], k=top_k)
        neighbors = []
        for weight, neighbor_idx in zip(values.tolist(), indices.tolist()):
            if not np.isfinite(weight):
                continue
            neighbor_idx = int(neighbor_idx)
            weight = float(weight)
            neighbors.append({"neighbor": neighbor_idx, "weight": weight})
            edge = tuple(sorted((int(block_idx), neighbor_idx)))
            undirected_edges.add(edge)
            directed_edges.append(
                {"first": int(block_idx), "second": neighbor_idx, "weight": weight}
            )
        neighbor_map[str(int(block_idx))] = neighbors
    return neighbor_map, sorted(undirected_edges), directed_edges


def main():
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    autocast_context = get_autocast_context(args.device, args.dtype)
    permutation_state = load_block_permutation_from_checkpoint(
        checkpoint,
        num_blocks=model.num_blocks,
        block_len=model.block_order_block_len,
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    block_perm = None if permutation_state is None else permutation_state["block_perm"]

    rng = np.random.default_rng(args.seed)
    matrix_sum = torch.zeros((model.num_blocks, model.num_blocks), dtype=torch.float64)
    total_samples = 0
    batch_start_offsets = []

    for batch_idx in range(int(args.num_batches)):
        idx, starts = sample_batch(
            tokens,
            batch_size=args.batch_size,
            block_size=model.config.block_size,
            rng=rng,
            device=args.device,
            token_perm=token_perm,
        )
        batch_start_offsets.extend(starts)
        token_orders, block_orders = build_orders_for_mode(model, idx, args.attn_mode)
        with torch.no_grad():
            with autocast_context:
                outputs = model(idx, mode=None, orders=token_orders, return_attentions=True)
        attn_outputs = extract_attentions_from_outputs(outputs)
        attn_batch = reduce_attention_batch(attn_outputs)
        for sample_idx in range(attn_batch.size(0)):
            if args.attn_export_type == "with_none":
                block_matrix = aggregate_predictor_aligned_attention_to_block(
                    attn_batch[sample_idx],
                    block_len=model.block_order_block_len,
                )
            else:
                block_matrix = aggregate_real_token_attention_to_block(
                    attn_batch[sample_idx],
                    block_len=model.block_order_block_len,
                )
            matrix_sum += reorder_block_attention_to_original(
                block_matrix,
                block_orders[sample_idx].detach(),
            ).double().cpu()
            total_samples += 1

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx + 1 == int(args.num_batches):
            print(f"processed {batch_idx + 1}/{args.num_batches} batches")

    if total_samples <= 0:
        raise RuntimeError("No attention samples were aggregated.")

    attention_matrix = matrix_sum / float(total_samples)
    if args.attn_symmetrize == "max":
        attention_graph = torch.maximum(attention_matrix, attention_matrix.t())
    elif args.attn_symmetrize == "mean":
        attention_graph = 0.5 * (attention_matrix + attention_matrix.t())
    else:
        attention_graph = attention_matrix

    neighbor_map, undirected_edges, directed_edges = build_topk_neighbors(
        attention_graph,
        top_k=args.top_k,
    )

    payload = {
        "run_meta": {
            "ckpt_path": str(args.ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": args.split,
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "device": args.device,
            "dtype": args.dtype,
            "attn_mode": str(args.attn_mode),
            "attn_export_type": str(args.attn_export_type),
            "attn_symmetrize": str(args.attn_symmetrize),
            "num_blocks": int(model.num_blocks),
            "block_len": int(model.block_order_block_len),
            "num_attention_samples": int(total_samples),
            "batch_start_offsets_preview": batch_start_offsets[: min(32, len(batch_start_offsets))],
            "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
            "permute_mode": checkpoint.get("config", {}).get("permute_mode", ""),
            "permute_seed": checkpoint.get("config", {}).get("permute_seed", None),
            "output_coordinate_frame": "current_l2r",
        },
        "attention_topk": {
            "top_neighbors_per_block": neighbor_map,
            "num_undirected_edges": int(len(undirected_edges)),
            "undirected_edges": [
                {"first": int(first), "second": int(second)}
                for first, second in undirected_edges
            ],
            "directed_topk_pairs": directed_edges,
        },
    }

    if block_perm is not None:
        payload["permute_map"] = {
            "permute_mode": "block",
            "block_perm": [int(v) for v in block_perm.tolist()],
            "inverse_block_perm": [int(v) for v in permutation_state["inverse_block_perm"].tolist()],
        }
        payload["attention_topk"]["top_neighbors_per_block_with_original"] = (
            annotate_neighbor_map_with_original(neighbor_map, block_perm)
        )
        payload["attention_topk"]["top_neighbors_per_block_original"] = map_neighbor_map_to_original(
            neighbor_map,
            block_perm,
        )
        payload["attention_topk"]["undirected_edges_with_original"] = map_edge_rows_to_original(
            payload["attention_topk"]["undirected_edges"],
            block_perm,
        )
        payload["attention_topk"]["directed_topk_pairs_with_original"] = map_edge_rows_to_original(
            directed_edges,
            block_perm,
        )
        payload["attention_topk"]["undirected_edges_original"] = payload["attention_topk"][
            "undirected_edges_with_original"
        ]
        payload["attention_topk"]["directed_topk_pairs_original"] = payload["attention_topk"][
            "directed_topk_pairs_with_original"
        ]

    if bool(args.save_attention_matrices):
        payload["attention_topk"]["attention_matrix"] = attention_matrix.tolist()
        payload["attention_topk"]["attention_graph"] = attention_graph.tolist()

    save_json(args.out_dir / "results.json", payload)
    print(f"saved top-k attention neighbors to {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
