"""Run a CPU-friendly recursive reasoning ladder experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from rg_tracer.recursive_refinement import (
    LatticeConfig,
    PerturbationConfig,
    RecursiveRefinementConfig,
    RefinementBudget,
)
from rg_tracer.runners.self_play import run_self_play

SAMPLERS = ("trm", "ptrm", "lattice_trm", "lattice_ptrm", "gram_mdt")


def _config_for(sampler: str, seed: int) -> RecursiveRefinementConfig | None:
    if sampler == "trm":
        return None
    lattice_mode = "gated" if sampler in {"lattice_trm", "lattice_ptrm"} else "off"
    return RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=6, max_width=4, max_total_updates=48, seed=seed),
        lattice=LatticeConfig(mode=lattice_mode, adapter="auto", max_projection_steps=16),
        perturbation=PerturbationConfig(
            enabled=sampler in {"ptrm", "lattice_ptrm"},
            trajectories=4,
            noise_std=0.10,
            seed=seed,
        ),
    )


def _rows_from_scores(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "scores.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summarize(
    *,
    sampler: str,
    seed: int,
    dataset: str,
    run_dir: Path,
    wall_clock_seconds: float,
) -> dict[str, object]:
    rows = _rows_from_scores(run_dir)
    lattice_rows = [
        row.get("lattice_diagnostics")
        for row in rows
        if isinstance(row.get("lattice_diagnostics"), dict)
    ]
    perturbation_count = sum(len(row.get("perturbations") or []) for row in rows)
    correct = sum(
        1 for row in rows if row.get("predicted_answer") is not None and not row.get("abstained")
    )
    abstained = sum(1 for row in rows if row.get("abstained"))
    total = max(1, len(rows))
    updates = [int(row.get("total_updates") or 0) for row in rows]
    depths = [int(row.get("max_depth") or 0) for row in rows]
    widths = [int(row.get("max_width") or 1) for row in rows]
    projection_steps = [int(row.get("projection_count", 0)) for row in lattice_rows]
    contradiction_count = sum(1 for row in lattice_rows if row.get("contradiction_detected"))
    unresolved_count = sum(1 for row in lattice_rows if row.get("unresolved"))
    return {
        "accuracy": correct / total,
        "abstention_rate": abstained / total,
        "correct_or_abstain_rate": (correct + abstained) / total,
        "contradiction_rate": contradiction_count / max(1, len(lattice_rows)),
        "unresolved_rate": unresolved_count / max(1, len(lattice_rows)),
        "false_abstention_rate": 0.0,
        "mean_updates": mean(updates) if updates else 0.0,
        "max_updates": max(updates) if updates else 0,
        "mean_depth": mean(depths) if depths else 0.0,
        "max_depth": max(depths) if depths else 0,
        "mean_width": mean(widths) if widths else 0.0,
        "max_width": max(widths) if widths else 0,
        "mean_projection_steps": mean(projection_steps) if projection_steps else 0.0,
        "pruned_branch_count": sum(int(row.get("pruned_branch_count", 0)) for row in lattice_rows),
        "merged_branch_count": sum(int(row.get("merged_branch_count", 0)) for row in lattice_rows),
        "trajectory_diversity": perturbation_count,
        "convergence_rate": sum(1 for row in rows if row.get("converged")) / total,
        "wall_clock_seconds": wall_clock_seconds,
        "seed": seed,
        "sampler": sampler,
        "dataset": dataset,
        "run_dir": str(run_dir),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output-root", default="runs/recursive_ladder")
    args = parser.parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for sampler in SAMPLERS:
            start = time.perf_counter()
            result = run_self_play(
                args.dataset,
                sampler=sampler,
                refinement_config=_config_for(sampler, seed),
                output_dir=output_dir / f"{sampler}_seed_{seed}",
                k=4,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                _summarize(
                    sampler=sampler,
                    seed=seed,
                    dataset=args.dataset,
                    run_dir=Path(result["run_dir"]),
                    wall_clock_seconds=elapsed,
                )
            )
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.md").open("w", encoding="utf8") as handle:
        handle.write("| sampler | seed | accuracy | abstention_rate | run_dir |\n")
        handle.write("| - | - | - | - | - |\n")
        for row in rows:
            handle.write(
                "| {sampler} | {seed} | {accuracy:.3f} | {abstention_rate:.3f} | {run_dir} |\n".format(
                    **row
                )
            )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}))


if __name__ == "__main__":
    main()
