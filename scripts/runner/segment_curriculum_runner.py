import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_csv_list(raw_value, cast_fn):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [cast_fn(item) for item in values]


def run_command(cmd, cwd):
    print("[runner] exec:")
    print("  " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_num_blocks_from_results(payload, segments_key="aggregated_segments", top_pairs_key="top_pairs"):
    max_block = -1
    for row in payload.get(segments_key, []):
        values = row.get("segment_original", row.get("segment", []))
        for value in values:
            max_block = max(max_block, int(value))
    for row in payload.get(top_pairs_key, []):
        first = row.get("first_original", row.get("first", -1))
        second = row.get("second_original", row.get("second", -1))
        max_block = max(max_block, int(first), int(second))
    if max_block < 0:
        return 0
    return max_block + 1


def build_stage_block_layout(payload, segments_key="aggregated_segments", top_pairs_key="top_pairs", original=False):
    num_blocks = infer_num_blocks_from_results(payload, segments_key=segments_key, top_pairs_key=top_pairs_key)
    aggregated_segments = payload.get(segments_key, [])
    segment_entries = []
    block_lookup = {}
    used_blocks = set()

    for segment_rank, row in enumerate(aggregated_segments, start=1):
        raw_segment = row.get("segment_original", row.get("segment", [])) if original else row.get("segment", [])
        segment = [int(v) for v in raw_segment]
        if len(segment) < 2:
            continue
        entry = {
            "unit_type": "segment",
            "segment_rank": int(segment_rank),
            "segment_length": int(len(segment)),
            "blocks": segment,
        }
        segment_entries.append(entry)
        for position_in_segment, block_idx in enumerate(segment, start=1):
            used_blocks.add(block_idx)
            block_lookup[int(block_idx)] = {
                "unit_type": "segment",
                "segment_rank": int(segment_rank),
                "position_in_segment": int(position_in_segment),
                "segment_length": int(len(segment)),
                "segment_blocks": segment,
            }

    singleton_blocks = [block_idx for block_idx in range(num_blocks) if block_idx not in used_blocks]
    for singleton_rank, block_idx in enumerate(singleton_blocks, start=1):
        segment_entries.append(
            {
                "unit_type": "singleton",
                "segment_rank": None,
                "segment_length": 1,
                "blocks": [int(block_idx)],
            }
        )
        block_lookup[int(block_idx)] = {
            "unit_type": "singleton",
            "segment_rank": None,
            "position_in_segment": 1,
            "segment_length": 1,
            "segment_blocks": [int(block_idx)],
        }

    linearized_order = []
    for entry in segment_entries:
        linearized_order.extend(entry["blocks"])

    return {
        "num_blocks": int(num_blocks),
        "units": segment_entries,
        "linearized_order": linearized_order,
        "block_lookup": block_lookup,
    }


def build_block_aggregation_trace(benchmark_root: Path, num_stages: int):
    stage_payloads = []
    stage_payloads_original = []
    max_num_blocks = 0

    for stage_idx in range(1, int(num_stages) + 1):
        results_path = benchmark_root / f"stage_{stage_idx:02d}" / "results.json"
        if not results_path.exists():
            continue
        payload = load_json(results_path)
        layout = build_stage_block_layout(payload)
        max_num_blocks = max(max_num_blocks, int(layout["num_blocks"]))
        stage_payloads.append(
            {
                "stage": int(stage_idx),
                "results_path": str(results_path),
                "aggregated_segments": payload.get("aggregated_segments", []),
                "linearized_order": layout["linearized_order"],
                "units": layout["units"],
                "block_lookup": layout["block_lookup"],
            }
        )
        if payload.get("aggregated_segments_original"):
            layout_original = build_stage_block_layout(
                payload,
                segments_key="aggregated_segments_original",
                top_pairs_key="top_pairs_original",
                original=True,
            )
            stage_payloads_original.append(
                {
                    "stage": int(stage_idx),
                    "results_path": str(results_path),
                    "aggregated_segments": payload.get("aggregated_segments_original", []),
                    "linearized_order": layout_original["linearized_order"],
                    "units": layout_original["units"],
                    "block_lookup": layout_original["block_lookup"],
                }
            )

    block_paths = []
    for block_idx in range(max_num_blocks):
        path = []
        for stage_payload in stage_payloads:
            lookup = stage_payload["block_lookup"].get(block_idx)
            if lookup is None:
                continue
            path.append(
                {
                    "stage": int(stage_payload["stage"]),
                    "unit_type": lookup["unit_type"],
                    "segment_rank": lookup["segment_rank"],
                    "position_in_segment": int(lookup["position_in_segment"]),
                    "segment_length": int(lookup["segment_length"]),
                    "segment_blocks": lookup["segment_blocks"],
                }
            )
        block_paths.append(
            {
                "block": int(block_idx),
                "path": path,
                "final_stage_position": path[-1] if path else None,
            }
        )

    block_paths_original = []
    if stage_payloads_original:
        for block_idx in range(max_num_blocks):
            path = []
            for stage_payload in stage_payloads_original:
                lookup = stage_payload["block_lookup"].get(block_idx)
                if lookup is None:
                    continue
                path.append(
                    {
                        "stage": int(stage_payload["stage"]),
                        "unit_type": lookup["unit_type"],
                        "segment_rank": lookup["segment_rank"],
                        "position_in_segment": int(lookup["position_in_segment"]),
                        "segment_length": int(lookup["segment_length"]),
                        "segment_blocks": lookup["segment_blocks"],
                    }
                )
            block_paths_original.append(
                {
                    "block": int(block_idx),
                    "path": path,
                    "final_stage_position": path[-1] if path else None,
                }
            )

    latest_stage = stage_payloads[-1] if stage_payloads else None
    latest_stage_original = stage_payloads_original[-1] if stage_payloads_original else None
    payload = {
        "num_stages_found": int(len(stage_payloads)),
        "final_stage_linearized_order": latest_stage["linearized_order"] if latest_stage else [],
        "final_stage_units": latest_stage["units"] if latest_stage else [],
        "block_paths": block_paths,
    }
    if latest_stage_original is not None:
        payload["final_stage_linearized_order_original"] = latest_stage_original["linearized_order"]
        payload["final_stage_units_original"] = latest_stage_original["units"]
        payload["block_paths_original"] = block_paths_original
    return payload


def load_runner_config_from_argv(argv):
    config_ns = {}
    filtered = []
    config_path = None
    for arg in argv:
        if "=" not in arg and not arg.startswith("--") and config_path is None:
            config_path = Path(arg)
        else:
            filtered.append(arg)
    if config_path is not None:
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        print(f"Overriding runner config with {config_path}:")
        with open(config_path, "r", encoding="utf-8") as handle:
            print(handle.read())
        exec(config_path.read_text(encoding="utf-8"), config_ns)
    return config_ns, filtered


def build_train_cmd(
    repo_dir,
    config_path,
    out_dir,
    init_from,
    max_iters,
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
        str(SCRIPTS_ROOT / "benchmark" / "structured_candidate_benchmark.py"),
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
    config_ns, filtered_argv = load_runner_config_from_argv(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description=(
            "Stage-wise curriculum runner: train random backbone, mine structured segments, "
            "then continue mixed random + segment-guided training across multiple stages."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(config_ns.get("config", "config/WikiText103/block32/standard/random.py")),
    )
    parser.add_argument(
        "--train_out_dir",
        type=str,
        default=config_ns.get("train_out_dir", "out-wikitext103-random-b32-curriculum"),
    )
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path(config_ns.get("benchmark_root", "Report/segment_curriculum_b32")),
    )
    parser.add_argument("--warmup_iters", type=int, default=int(config_ns.get("warmup_iters", 4000)))
    parser.add_argument("--stage_iters", type=int, default=int(config_ns.get("stage_iters", 4000)))
    parser.add_argument("--num_curriculum_stages", type=int, default=int(config_ns.get("num_curriculum_stages", 2)))
    parser.add_argument(
        "--segment_guided_ratios",
        type=str,
        default=str(config_ns.get("segment_guided_ratios", "0.3,0.5")),
        help="Comma-separated ratios, one per curriculum stage.",
    )
    parser.add_argument(
        "--segment_max_lens",
        type=str,
        default=str(config_ns.get("segment_max_lens", "4,6")),
        help="Comma-separated max segment lengths, one per curriculum stage.",
    )
    parser.add_argument("--segment_max_units_per_order", type=int, default=int(config_ns.get("segment_max_units_per_order", 2)))
    parser.add_argument("--segment_top_k_pairs", type=int, default=int(config_ns.get("segment_top_k_pairs", 64)))
    parser.add_argument("--benchmark_batch_size", type=int, default=int(config_ns.get("benchmark_batch_size", 64)))
    parser.add_argument("--benchmark_num_batches", type=int, default=int(config_ns.get("benchmark_num_batches", 200)))
    parser.add_argument("--pair_mining_batches", type=int, default=int(config_ns.get("pair_mining_batches", 24)))
    parser.add_argument("--pair_eval_batch_size", type=int, default=int(config_ns.get("pair_eval_batch_size", 8)))
    parser.add_argument("--candidate_eval_batch_size", type=int, default=int(config_ns.get("candidate_eval_batch_size", 64)))
    parser.add_argument("--random_pool_size", type=int, default=int(config_ns.get("random_pool_size", 64)))
    parser.add_argument("--structured_pool_size", type=int, default=int(config_ns.get("structured_pool_size", 64)))
    parser.add_argument("--top_pair_pool_size", type=int, default=int(config_ns.get("top_pair_pool_size", 128)))
    parser.add_argument("--aggregate_top_k_pairs", type=int, default=int(config_ns.get("aggregate_top_k_pairs", 64)))
    parser.add_argument("--prefix_len", type=int, default=int(config_ns.get("prefix_len", 8)))
    parser.add_argument("--pair_score_k", type=int, default=int(config_ns.get("pair_score_k", 2)))
    parser.add_argument("--tv_weight", type=float, default=float(config_ns.get("tv_weight", 0.3)))
    parser.add_argument("--benchmark_log_every_batches", type=int, default=int(config_ns.get("benchmark_log_every_batches", 10)))
    args = parser.parse_args(filtered_argv)
    return args


def main():
    args = parse_args()
    repo_dir = REPO_ROOT
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config

    ratios = parse_csv_list(args.segment_guided_ratios, float)
    segment_lens = parse_csv_list(args.segment_max_lens, int)
    if len(ratios) != int(args.num_curriculum_stages):
        raise ValueError("segment_guided_ratios length must equal num_curriculum_stages")
    if len(segment_lens) != int(args.num_curriculum_stages):
        raise ValueError("segment_max_lens length must equal num_curriculum_stages")

    train_out_dir = args.train_out_dir
    benchmark_root = args.benchmark_root if args.benchmark_root.is_absolute() else REPO_ROOT / args.benchmark_root
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
                segment_guided_ratio=ratios[stage_idx - 1],
                segment_source_json=str(benchmark_dir / "results.json"),
                segment_top_k_pairs=args.segment_top_k_pairs,
                segment_max_len=segment_lens[stage_idx - 1],
                segment_max_units_per_order=args.segment_max_units_per_order,
            ),
            cwd=repo_dir,
        )

        trace_payload = build_block_aggregation_trace(benchmark_root, stage_idx)
        (benchmark_root / "block_aggregation_trace.json").write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
    }
    (benchmark_root / "runner_meta.json").write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trace_payload = build_block_aggregation_trace(benchmark_root, args.num_curriculum_stages)
    (benchmark_root / "block_aggregation_trace.json").write_text(
        json.dumps(trace_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[runner] finished segment curriculum; meta saved to {benchmark_root / 'runner_meta.json'}")


if __name__ == "__main__":
    main()
