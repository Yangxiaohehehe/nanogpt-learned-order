import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def parse_csv_list(raw_value, cast_fn):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [cast_fn(item) for item in values]


def run_command(cmd, cwd):
    print("[scan] exec:")
    print("  " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_step_from_path(path: Path):
    match = re.search(r"iter(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def discover_checkpoints(checkpoint_dir: Path, step_filter=None, limit=None):
    ckpts = sorted(checkpoint_dir.glob("ckpt_iter*.pt"))
    if step_filter is not None:
        step_filter = set(int(v) for v in step_filter)
        ckpts = [path for path in ckpts if infer_step_from_path(path) in step_filter]
    if limit is not None and int(limit) > 0:
        ckpts = ckpts[: int(limit)]
    return ckpts


def build_benchmark_cmd(args, ckpt_path: Path, out_dir: Path):
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "benchmark" / "structured_candidate_benchmark.py"),
        f"--ckpt_path={ckpt_path}",
        f"--out_dir={out_dir}",
        f"--batch_size={int(args.batch_size)}",
        f"--num_batches={int(args.num_batches)}",
        f"--pair_mining_batches={int(args.pair_mining_batches)}",
        f"--pair_eval_batch_size={int(args.pair_eval_batch_size)}",
        f"--candidate_eval_batch_size={int(args.candidate_eval_batch_size)}",
        f"--random_pool_size={int(args.random_pool_size)}",
        f"--structured_pool_size={int(args.structured_pool_size)}",
        f"--top_pair_pool_size={int(args.top_pair_pool_size)}",
        f"--aggregate_top_k_pairs={int(args.aggregate_top_k_pairs)}",
        f"--prefix_len={int(args.prefix_len)}",
        f"--segment_len={int(args.segment_len)}",
        f"--pair_score_k={int(args.pair_score_k)}",
        f"--tv_weight={float(args.tv_weight)}",
        f"--log_every_batches={int(args.log_every_batches)}",
        f"--seed={int(args.seed)}",
    ]
    if args.device:
        cmd.append(f"--device={args.device}")
    if args.dtype:
        cmd.append(f"--dtype={args.dtype}")
    if args.dataset:
        cmd.append(f"--dataset={args.dataset}")
    if args.data_dir:
        cmd.append(f"--data_dir={args.data_dir}")
    if args.split:
        cmd.append(f"--split={args.split}")
    if bool(args.save_all_pair_scores):
        cmd.append("--save_all_pair_scores")
    return cmd


def build_summary_row(step, ckpt_path: Path, result_payload):
    summaries = result_payload.get("summaries", {})
    best_structured = summaries.get("best_structured", {})
    best_random = summaries.get("best_random_pool", {})
    l2r_ref = summaries.get("l2r_reference", {})
    top_pairs = result_payload.get("top_pairs", [])
    aggregated_segments = result_payload.get("aggregated_segments", [])
    return {
        "step": int(step) if step is not None else None,
        "ckpt_path": str(ckpt_path),
        "structured_mean_kendall_tau": best_structured.get("mean_kendall_tau"),
        "structured_mean_kendall_distance": best_structured.get("mean_kendall_distance"),
        "structured_mean_prefix_mean_index": best_structured.get("mean_prefix_mean_index"),
        "structured_mean_adjacent_pairs": best_structured.get("mean_adjacent_pairs"),
        "structured_mean_longest_run": best_structured.get("mean_longest_run"),
        "structured_mean_early_area_plus_tv": best_structured.get("mean_early_area_plus_tv"),
        "random_mean_kendall_tau": best_random.get("mean_kendall_tau"),
        "random_mean_prefix_mean_index": best_random.get("mean_prefix_mean_index"),
        "random_mean_adjacent_pairs": best_random.get("mean_adjacent_pairs"),
        "random_mean_early_area_plus_tv": best_random.get("mean_early_area_plus_tv"),
        "l2r_mean_early_area_plus_tv": l2r_ref.get("mean_early_area_plus_tv"),
        "num_top_pairs": len(top_pairs),
        "num_aggregated_segments": len(aggregated_segments),
        "top_pair_first": top_pairs[0]["first"] if top_pairs else None,
        "top_pair_second": top_pairs[0]["second"] if top_pairs else None,
        "top_pair_score": top_pairs[0]["score"] if top_pairs else None,
        "top_pair_margin_vs_reverse": top_pairs[0]["margin_vs_reverse"] if top_pairs else None,
        "aggregated_segments": aggregated_segments,
        "top_pairs": top_pairs,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-scan early checkpoints, run structured pair/segment mining, "
            "and summarize how local order signal emerges over training."
        )
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("out-wikitext103-random-b32/checkpoints"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("Report/analysis/early_signal_scan_b32"),
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated iteration steps to scan. Empty means scan every discovered checkpoint.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--pair_mining_batches", type=int, default=24)
    parser.add_argument("--pair_eval_batch_size", type=int, default=32)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--random_pool_size", type=int, default=64)
    parser.add_argument("--structured_pool_size", type=int, default=64)
    parser.add_argument("--top_pair_pool_size", type=int, default=128)
    parser.add_argument("--aggregate_top_k_pairs", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--segment_len", type=int, default=4)
    parser.add_argument("--pair_score_k", type=int, default=2)
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--log_every_batches", type=int, default=10)
    parser.add_argument("--save_all_pair_scores", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoints whose results.json already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else REPO_ROOT / args.checkpoint_dir
    out_root = args.out_root if args.out_root.is_absolute() else REPO_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    step_filter = parse_csv_list(args.steps, int) if str(args.steps).strip() else None
    limit = int(args.limit) if int(args.limit) > 0 else None
    checkpoints = discover_checkpoints(checkpoint_dir, step_filter=step_filter, limit=limit)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    summary_rows = []
    for ckpt_path in checkpoints:
        step = infer_step_from_path(ckpt_path)
        label = f"iter_{int(step):07d}" if step is not None else ckpt_path.stem
        run_out_dir = out_root / label
        result_path = run_out_dir / "results.json"
        if not (args.skip_existing and result_path.exists()):
            run_command(build_benchmark_cmd(args, ckpt_path=ckpt_path, out_dir=run_out_dir), cwd=REPO_ROOT)
        result_payload = load_json(result_path)
        summary_rows.append(build_summary_row(step, ckpt_path, result_payload))

    summary_payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "out_root": str(out_root),
        "num_checkpoints": len(checkpoints),
        "rows": summary_rows,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scan] saved summary to {summary_path}")


if __name__ == "__main__":
    main()
