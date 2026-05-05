from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from slimevolley.verify.gym_rollout import run_gym_rollout
from slimevolley.verify.jax_rollout import run_jax_rollout
from slimevolley.verify.state_trace import write_trace_csv


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def _check_import(module_name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(module_name)
        return {
            "module": module_name,
            "ok": True,
            "version": str(getattr(module, "__version__", "unknown")),
            "error": "",
        }
    except Exception as exc:
        return {
            "module": module_name,
            "ok": False,
            "version": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_deps() -> list[dict[str, str | bool]]:
    return [_check_import(name) for name in ("numpy", "jax", "gym", "slimevolleygym")]


def _timestamped_run_dir(output_root: Path) -> Path:
    run_id = datetime.now().strftime("verify_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_rollouts(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["backend", "matchup", "seed", "score", "steps", "right_life", "left_life"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summarize(rows) -> dict:
    if not rows:
        return {"count": 0}
    scores = np.asarray([r.score for r in rows], dtype=np.float32)
    steps = np.asarray([r.steps for r in rows], dtype=np.float32)
    return {
        "count": len(rows),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "steps_mean": float(steps.mean()),
        "steps_std": float(steps.std()),
    }


def _build_mismatch_report(gym_rows, jax_rows, score_mean_tolerance, steps_mean_tolerance) -> tuple[list[str], bool]:
    lines = []
    gym_summary = _summarize(gym_rows)
    jax_summary = _summarize(jax_rows)
    passed = True

    if gym_summary["count"] != jax_summary["count"]:
        passed = False
        lines.append(f"Episode count mismatch: gym={gym_summary['count']} jax={jax_summary['count']}")

    if gym_summary["count"] and jax_summary["count"]:
        score_delta = abs(gym_summary["score_mean"] - jax_summary["score_mean"])
        steps_delta = abs(gym_summary["steps_mean"] - jax_summary["steps_mean"])
        if score_delta > score_mean_tolerance:
            passed = False
            lines.append(
                f"Score mean differs by {score_delta:.4f}; "
                f"gym={gym_summary['score_mean']:.4f}, jax={jax_summary['score_mean']:.4f}"
            )
        if steps_delta > steps_mean_tolerance:
            passed = False
            lines.append(
                f"Step mean differs by {steps_delta:.4f}; "
                f"gym={gym_summary['steps_mean']:.4f}, jax={jax_summary['steps_mean']:.4f}"
            )

    if passed:
        lines.append("No aggregate mismatch above configured tolerances.")
    return lines, passed


def _write_text_summary(path: Path, summary: dict, mismatch_lines: list[str]) -> None:
    lines = [
        f"matchup: {summary.get('matchup', '')}",
        f"episodes: {summary.get('episodes', '')}",
        f"passed: {summary.get('passed', False)}",
        "",
        "gym:",
        json.dumps(summary.get("gym", {}), indent=2, sort_keys=True),
        "",
        "jax:",
        json.dumps(summary.get("jax", {}), indent=2, sort_keys=True),
        "",
        "mismatch report:",
        *mismatch_lines,
        "",
    ]
    path.write_text("\n".join(lines))


def run_comparison(args) -> Path:
    run_dir = _timestamped_run_dir(Path(args.output_root))
    seeds = args.seeds if args.seeds else [args.seed + i for i in range(args.episodes)]

    gym_rows = []
    jax_rows = []
    errors = []

    for idx, seed in enumerate(seeds[: args.episodes]):
        try:
            gym_result, gym_trace = run_gym_rollout(
                args.matchup,
                seed,
                genome_path=args.genome,
                trace=args.trace,
                max_steps=args.max_steps,
            )
            gym_rows.append(gym_result)
            if args.trace:
                write_trace_csv(run_dir / f"state_trace_gym_seed_{seed:06d}.csv", gym_trace)
        except Exception as exc:
            errors.append({"backend": "gym", "seed": seed, "error": f"{type(exc).__name__}: {exc}"})

        try:
            jax_result, jax_trace = run_jax_rollout(
                args.matchup,
                seed,
                genome_path=args.genome,
                trace=args.trace,
                max_steps=args.max_steps,
            )
            jax_rows.append(jax_result)
            if args.trace:
                write_trace_csv(run_dir / f"state_trace_jax_seed_{seed:06d}.csv", jax_trace)
        except Exception as exc:
            errors.append({"backend": "jax", "seed": seed, "error": f"{type(exc).__name__}: {exc}"})

    _write_rollouts(run_dir / "gym_rollouts.csv", gym_rows)
    _write_rollouts(run_dir / "jax_rollouts.csv", jax_rows)

    mismatch_lines, passed = _build_mismatch_report(
        gym_rows,
        jax_rows,
        args.score_mean_tolerance,
        args.steps_mean_tolerance,
    )
    if errors:
        passed = False
        mismatch_lines.append("Errors:")
        mismatch_lines.extend(f"{e['backend']} seed={e['seed']}: {e['error']}" for e in errors)

    summary = {
        "matchup": args.matchup,
        "episodes": min(args.episodes, len(seeds)),
        "seeds": seeds[: args.episodes],
        "passed": passed,
        "gym": _summarize(gym_rows),
        "jax": _summarize(jax_rows),
        "errors": errors,
        "score_mean_tolerance": args.score_mean_tolerance,
        "steps_mean_tolerance": args.steps_mean_tolerance,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "mismatch_report.txt").write_text("\n".join(mismatch_lines) + "\n")
    _write_text_summary(run_dir / "summary.txt", summary, mismatch_lines)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original Gym and JAX SlimeVolley rollouts.")
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--policies",
        nargs=2,
        metavar=("RIGHT", "LEFT"),
        choices=["baseline", "random", "noop", "genome"],
        default=None,
        help="Optional alias for --matchup, for example: --policies noop baseline.",
    )
    parser.add_argument(
        "--matchup",
        choices=["baseline_vs_baseline", "random_vs_baseline", "noop_vs_baseline", "genome_vs_baseline"],
        default="baseline_vs_baseline",
    )
    parser.add_argument("--genome", type=str, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--score-mean-tolerance", type=float, default=0.25)
    parser.add_argument("--steps-mean-tolerance", type=float, default=150.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.policies is not None:
        right, left = args.policies
        matchup = f"{right}_vs_{left}"
        supported = {"baseline_vs_baseline", "random_vs_baseline", "noop_vs_baseline", "genome_vs_baseline"}
        if matchup not in supported:
            raise SystemExit(f"Unsupported --policies combination: {right} {left}")
        args.matchup = matchup

    if args.check_deps:
        results = check_deps()
        print(json.dumps(results, indent=2, sort_keys=True))
        if not all(bool(row["ok"]) for row in results):
            raise SystemExit(1)
        return

    if args.matchup == "genome_vs_baseline" and args.genome is None:
        raise SystemExit("--genome is required for genome_vs_baseline.")

    run_dir = run_comparison(args)
    print(f"Verification artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
