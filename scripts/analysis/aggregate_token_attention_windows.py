"""
Aggregate a token-level attention matrix into coarse window-level attention.

Typical usage:
python scripts/analysis/aggregate_token_attention_windows.py \
  --input_npy /home/devbox/project/AOGPT-test-order/nanogpt_learned_order/Report/analysis/attn_map_seq256_b1_permute_random-6-8-256/without_none/true_original/block_attention_without_none.npy \
  --out_dir /home/devbox/project/AOGPT-test-order/nanogpt_learned_order/Report/analysis/attn_map_seq256_b1_permute_random-6-8-256_window8/without_none/true_original \
  --window_size 8 \
  --label b1_token_true_original_without_none
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a token-level attention matrix into fixed-size windows to check whether "
            "block-like locality emerges after pooling."
        )
    )
    parser.add_argument("--input_npy", type=Path, required=True, help="Path to a square token-level attention .npy file.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to save pooled outputs.")
    parser.add_argument("--window_size", type=int, default=8, help="Number of tokens per pooled window.")
    parser.add_argument("--label", type=str, default="token_attention_window_pool", help="Label used in titles and metadata.")
    return parser.parse_args()


def save_heatmap_png(matrix, out_path: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("key window")
    plt.ylabel("query window")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def aggregate_to_windows(matrix: np.ndarray, window_size: int) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape={matrix.shape}.")
    if matrix.shape[0] % window_size != 0:
        raise ValueError(
            f"Matrix size {matrix.shape[0]} is not divisible by window_size={window_size}."
        )

    num_windows = matrix.shape[0] // window_size
    pooled = matrix.reshape(num_windows, window_size, num_windows, window_size).mean(axis=(1, 3))
    return pooled


def summarize_matrix(matrix: np.ndarray):
    n = matrix.shape[0]
    diag = np.diag(matrix)
    off = matrix[~np.eye(n, dtype=bool)]

    row_top5 = {}
    for row_idx in sorted(set([0, 1, 2, 3, max(0, n // 4), max(0, n // 2), max(0, n - 1)])):
        idx = np.argsort(matrix[row_idx])[::-1][:5]
        row_top5[str(row_idx)] = {
            "idx": idx.tolist(),
            "vals": np.round(matrix[row_idx, idx], 6).tolist(),
            "dist": [abs(int(col) - row_idx) for col in idx],
        }

    band_means = []
    for distance in range(min(9, n)):
        vals = []
        for i in range(n - distance):
            vals.append(matrix[i, i + distance])
            if distance > 0:
                vals.append(matrix[i + distance, i])
        band_means.append(float(np.mean(vals)))

    excl_self_distances = []
    for row_idx in range(n):
        row = matrix[row_idx].copy()
        row[row_idx] = -np.inf
        idx = np.argsort(row)[::-1][:5]
        excl_self_distances.extend(abs(int(col) - row_idx) for col in idx)
    excl_self_distances = np.asarray(excl_self_distances)

    return {
        "shape": list(matrix.shape),
        "diag_mean": float(diag.mean()),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
        "off_mean": float(off.mean()),
        "off_std": float(off.std()),
        "diag_off_ratio": float(diag.mean() / off.mean()) if off.mean() != 0 else None,
        "top_left_8x8": np.round(matrix[:8, :8], 6).tolist(),
        "row_top5": row_top5,
        "band_means_d0_to_d8": band_means,
        "top5_excl_self_within1": float((excl_self_distances <= 1).mean()),
        "top5_excl_self_within2": float((excl_self_distances <= 2).mean()),
        "top5_excl_self_within4": float((excl_self_distances <= 4).mean()),
        "top5_excl_self_within8": float((excl_self_distances <= 8).mean()),
        "top5_excl_self_mean_dist": float(excl_self_distances.mean()),
    }


def main():
    args = parse_args()

    matrix = np.load(args.input_npy)
    pooled = aggregate_to_windows(matrix, args.window_size)
    summary = summarize_matrix(pooled)
    summary.update(
        {
            "label": args.label,
            "input_npy": str(args.input_npy),
            "window_size": int(args.window_size),
            "input_shape": list(matrix.shape),
            "pooling": "reshape(num_windows, window_size, num_windows, window_size).mean(axis=(1, 3))",
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "window_attention.npy", pooled)
    (args.out_dir / "window_attention_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    save_heatmap_png(
        pooled,
        args.out_dir / "window_attention.png",
        title=f"{args.label} | pooled window attention",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved pooled outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
