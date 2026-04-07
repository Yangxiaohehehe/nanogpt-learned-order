import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_csv_list(raw_value, cast_fn):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [cast_fn(item) for item in values]


def run_command(cmd, cwd):
    print("[runner] exec:")
    print("  " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def build_train_cmd(
    repo_dir,
    config_path,
    out_dir,
    init_from,
    max_iters,
    use_order_head,
    segment_guided_ratio,
    segment_source_json,
    segment_top_k_pairs,
    segment_max_len,
    segment_max_units_per_order,
):
    cmd = [
        sys.executable,
        "train.py",
        str(config_path),
        f"--out_dir={out_dir}",
        f"--init_from={init_from}",
        f"--max_iters={int(max_iters)}",
        f"--use_order_head={bool(use_order_head)}",
        f"--segment_guided_ratio={float(segment_guided_ratio)}",
        f"--segment_top_k_pairs={int(segment_top_k_pairs)}",
        f"--segment_max_len={int(segment_max_len)}",
        f"--segment_max_units_per_order={int(segment_max_units_per_order)}",
    ]
    if segment_source_json:
        cmd.append(f"--segment_source_json={segment_source_json}")
    return cmd


def build_benchmark_cmd(
    benchmark_out_dir,
    ckpt_path,
    batch_size,
    num_batches,
    pair_mining_batches,
    pair_eval_batch_size,
    candidate_eval_batch_size,
    random_pool_size,
    structured_pool_size,
    top_pair_pool_size,
    aggregate_top_k_pairs,
    prefix_len,
    segment_len,
    pair_score_k,
    tv_weight,
    log_every_batches,
):
    return [
        sys.executable,
        "structured_candidate_benchmark.py",
        f"--ckpt_path={ckpt_path}",
        f"--out_dir={benchmark_out_dir}",
        f"--batch_size={int(batch_size)}",
        f"--num_batches={int(num_batches)}",
        f"--pair_mining_batches={int(pair_mining_batches)}",
        f"--pair_eval_batch_size={int(pair_eval_batch_size)}",
        f"--candidate_eval_batch_size={int(candidate_eval_batch_size)}",
        f"--random_pool_size={int(random_pool_size)}",
        f"--structured_pool_size={int(structured_pool_size)}",
        f"--top_pair_pool_size={int(top_pair_pool_size)}",
        f"--aggregate_top_k_pairs={int(aggregate_top_k_pairs)}",
        f"--prefix_len={int(prefix_len)}",
        f"--segment_len={int(segment_len)}",
        f"--pair_score_k={int(pair_score_k)}",
        f"--tv_weight={float(tv_weight)}",
        f"--log_every_batches={int(log_every_batches)}",
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-wise curriculum runner: train random backbone, mine structured segments, "
            "then continue mixed random + segment-guided training across multiple stages."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/WikiText103/block32/standard/random.py"),
    )
    parser.add_argument(
        "--train_out_dir",
        type=str,
        default="out-wikitext103-random-b32-curriculum",
    )
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("Report/segment_curriculum_b32"),
    )
    parser.add_argument("--warmup_iters", type=int, default=4000)
    parser.add_argument("--stage_iters", type=int, default=4000)
    parser.add_argument("--num_curriculum_stages", type=int, default=2)
    parser.add_argument(
        "--segment_guided_ratios",
        type=str,
        default="0.3,0.5",
        help="Comma-separated ratios, one per curriculum stage.",
    )
    parser.add_argument(
        "--segment_max_lens",
        type=str,
        default="4,6",
        help="Comma-separated max segment lengths, one per curriculum stage.",
    )
    parser.add_argument("--segment_max_units_per_order", type=int, default=2)
    parser.add_argument("--segment_top_k_pairs", type=int, default=64)
    parser.add_argument("--disable_order_head", action="store_true")
    parser.add_argument("--benchmark_batch_size", type=int, default=64)
    parser.add_argument("--benchmark_num_batches", type=int, default=200)
    parser.add_argument("--pair_mining_batches", type=int, default=24)
    parser.add_argument("--pair_eval_batch_size", type=int, default=8)
    parser.add_argument("--candidate_eval_batch_size", type=int, default=64)
    parser.add_argument("--random_pool_size", type=int, default=64)
    parser.add_argument("--structured_pool_size", type=int, default=64)
    parser.add_argument("--top_pair_pool_size", type=int, default=128)
    parser.add_argument("--aggregate_top_k_pairs", type=int, default=64)
    parser.add_argument("--prefix_len", type=int, default=8)
    parser.add_argument("--pair_score_k", type=int, default=2)
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--benchmark_log_every_batches", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    config_path = args.config

    ratios = parse_csv_list(args.segment_guided_ratios, float)
    segment_lens = parse_csv_list(args.segment_max_lens, int)
    if len(ratios) != int(args.num_curriculum_stages):
        raise ValueError("segment_guided_ratios length must equal num_curriculum_stages")
    if len(segment_lens) != int(args.num_curriculum_stages):
        raise ValueError("segment_max_lens length must equal num_curriculum_stages")

    train_out_dir = args.train_out_dir
    benchmark_root = args.benchmark_root
    benchmark_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = repo_dir / train_out_dir / "ckpt.pt"

    # Stage 0: pure-random warmup from scratch.
    run_command(
        build_train_cmd(
            repo_dir=repo_dir,
            config_path=config_path,
            out_dir=train_out_dir,
            init_from="scratch",
            max_iters=args.warmup_iters,
            use_order_head=not args.disable_order_head,
            segment_guided_ratio=0.0,
            segment_source_json="",
            segment_top_k_pairs=args.segment_top_k_pairs,
            segment_max_len=segment_lens[0],
            segment_max_units_per_order=args.segment_max_units_per_order,
        ),
        cwd=repo_dir,
    )

    cumulative_max_iters = int(args.warmup_iters)
    for stage_idx in range(1, int(args.num_curriculum_stages) + 1):
        benchmark_dir = benchmark_root / f"stage_{stage_idx:02d}"
        run_command(
            build_benchmark_cmd(
                benchmark_out_dir=benchmark_dir,
                ckpt_path=ckpt_path,
                batch_size=args.benchmark_batch_size,
                num_batches=args.benchmark_num_batches,
                pair_mining_batches=args.pair_mining_batches,
                pair_eval_batch_size=args.pair_eval_batch_size,
                candidate_eval_batch_size=args.candidate_eval_batch_size,
                random_pool_size=args.random_pool_size,
                structured_pool_size=args.structured_pool_size,
                top_pair_pool_size=args.top_pair_pool_size,
                aggregate_top_k_pairs=args.aggregate_top_k_pairs,
                prefix_len=args.prefix_len,
                segment_len=segment_lens[stage_idx - 1],
                pair_score_k=args.pair_score_k,
                tv_weight=args.tv_weight,
                log_every_batches=args.benchmark_log_every_batches,
            ),
            cwd=repo_dir,
        )

        cumulative_max_iters += int(args.stage_iters)
        run_command(
            build_train_cmd(
                repo_dir=repo_dir,
                config_path=config_path,
                out_dir=train_out_dir,
                init_from="resume",
                max_iters=cumulative_max_iters,
                use_order_head=not args.disable_order_head,
                segment_guided_ratio=ratios[stage_idx - 1],
                segment_source_json=str(benchmark_dir / "results.json"),
                segment_top_k_pairs=args.segment_top_k_pairs,
                segment_max_len=segment_lens[stage_idx - 1],
                segment_max_units_per_order=args.segment_max_units_per_order,
            ),
            cwd=repo_dir,
        )

    final_payload = {
        "config": str(config_path),
        "train_out_dir": train_out_dir,
        "benchmark_root": str(benchmark_root),
        "warmup_iters": int(args.warmup_iters),
        "stage_iters": int(args.stage_iters),
        "num_curriculum_stages": int(args.num_curriculum_stages),
        "segment_guided_ratios": ratios,
        "segment_max_lens": segment_lens,
        "segment_max_units_per_order": int(args.segment_max_units_per_order),
        "segment_top_k_pairs": int(args.segment_top_k_pairs),
        "disable_order_head": bool(args.disable_order_head),
    }
    (benchmark_root / "runner_meta.json").write_text(
        __import__("json").dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[runner] finished segment curriculum; meta saved to {benchmark_root / 'runner_meta.json'}")


if __name__ == "__main__":
    main()
