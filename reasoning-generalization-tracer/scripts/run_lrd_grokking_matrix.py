"""Run optional real toy-transformer LRD grokking ablations."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from rg_tracer.grokking.artifacts import (
    append_jsonl,
    write_summary_csv,
    write_summary_markdown,
)
from rg_tracer.grokking.config import SpectralRegularizationConfig, ToyGrokkingConfig
from rg_tracer.grokking.trainer import train_toy_grokking


def _floats(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=17)
    parser.add_argument("--train-fractions", nargs="+", type=float, default=[0.40])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--l2-weights", nargs="+", default=["0.0", "0.01"])
    parser.add_argument("--lrd-weights", nargs="+", default=["0.0", "0.01"])
    parser.add_argument("--target-matrices", nargs="+", default=["q_proj", "k_proj"])
    parser.add_argument("--newton-schulz-steps", type=int, default=5)
    parser.add_argument("--output-root", default="runs/lrd_grokking")
    parser.add_argument("--limit", type=int)
    return parser


def _regularization_configs(args: argparse.Namespace) -> list[SpectralRegularizationConfig]:
    l2_weights = _floats(args.l2_weights)
    lrd_weights = _floats(args.lrd_weights)
    targets = tuple(args.target_matrices)
    configs = [SpectralRegularizationConfig(mode="none")]
    configs.extend(
        SpectralRegularizationConfig(mode="l2", l2_weight=weight, target_matrices=targets)
        for weight in l2_weights
        if weight > 0.0
    )
    configs.extend(
        SpectralRegularizationConfig(
            mode="lrd",
            lrd_weight=weight,
            target_matrices=targets,
            newton_schulz_steps=args.newton_schulz_steps,
            use_newton_schulz=True,
        )
        for weight in lrd_weights
        if weight > 0.0
    )
    configs.extend(
        SpectralRegularizationConfig(
            mode="l2_plus_lrd",
            l2_weight=l2_weight,
            lrd_weight=lrd_weight,
            target_matrices=targets,
            newton_schulz_steps=args.newton_schulz_steps,
            use_newton_schulz=True,
        )
        for l2_weight in l2_weights
        for lrd_weight in lrd_weights
        if l2_weight > 0.0 and lrd_weight > 0.0
    )
    return configs


def main(argv: list[str] | None = None) -> None:
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = itertools.product(args.train_fractions, args.seeds, _regularization_configs(args))
    for index, (train_fraction, seed, regularization) in enumerate(cells):
        if args.limit is not None and index >= args.limit:
            break
        cell = f"cell_{index:04d}_{regularization.mode}_seed{seed}_tf{train_fraction:g}"
        cell_dir = output_root / cell
        toy_config = ToyGrokkingConfig(
            modulus=args.modulus,
            train_fraction=train_fraction,
            epochs=args.epochs,
            eval_interval=args.eval_interval,
            seed=seed,
        )
        result = train_toy_grokking(toy_config, regularization, output_dir=cell_dir)
        summary = dict(result.summary)
        summary["cell"] = cell
        rows.append(summary)
        append_jsonl(output_root / "results.jsonl", summary)
    write_summary_csv(output_root / "summary.csv", rows)
    write_summary_markdown(output_root / "summary.md", rows)
    print(json.dumps({"output_root": str(output_root), "runs": len(rows)}))


if __name__ == "__main__":  # pragma: no cover
    main()
