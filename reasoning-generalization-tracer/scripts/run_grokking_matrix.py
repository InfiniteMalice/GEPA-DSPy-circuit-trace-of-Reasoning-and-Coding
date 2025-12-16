"""Run attribution-aware grokking experiments across a configuration matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import numbers
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from rg_tracer.attribution import graphs as attr_graphs
from rg_tracer.attribution import metrics as attr_metrics

REGULARISATION = ["no_wd", "wd"]
STABILITY = ["softmax", "stablemax"]
GRADIENT = ["off", "perpend_grad_on"]
REASONING = ["sft_only", "srl_pretrain_then_sft"]
PHASES = ["pre_overfit", "overfit", "pre_grok", "post_grok"]


def _combo_name(combo: Mapping[str, str]) -> str:
    return "_".join(f"{key}={value}" for key, value in sorted(combo.items()))


def _build_combo_grid() -> Iterable[Mapping[str, str]]:
    keys = ["regularisation", "stability", "gradient", "reasoning"]
    values = [REGULARISATION, STABILITY, GRADIENT, REASONING]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=True))


def _extract_phase_graphs(combo_digest: int, combo_name: str) -> list[Mapping[str, object]]:
    backend = attr_graphs.get_backend("null")
    graphs: list[Mapping[str, object]] = []
    for index, phase in enumerate(PHASES):
        payload = {
            "task_id": f"phase_{phase}_{combo_digest}",
            "tokens": [index, combo_digest],
            "phase": phase,
            "probe_index": index,
        }
        phase_key = f"{combo_digest}_{combo_name}_{phase}_{index}"
        phase_seed = int(hashlib.sha256(phase_key.encode("utf8")).hexdigest(), 16) % 10_000
        graph = attr_graphs.extract_graph(
            model=None,
            inputs=payload,
            backend=backend,
            seed=phase_seed,
        )
        graphs.append(graph)
    return graphs


def _summarise_metrics(
    graphs: list[Mapping[str, object]],
) -> dict[str, object]:
    metrics = {
        "sparsity": attr_metrics.path_sparsity(graphs),
        "avg_path_length": attr_metrics.average_path_length(graphs),
        "branching_factor": attr_metrics.average_branching_factor(graphs),
        "repeatability": attr_metrics.repeatability(graphs),
        "delta_sparsity": attr_metrics.delta_sparsity(graphs),
        "delta_repeatability": attr_metrics.delta_repeatability(graphs),
    }
    # Concept annotations are not part of the matrix sweep, so mark alignment
    # deltas as unavailable rather than returning a wall of zeros.
    metrics["delta_alignment"] = None
    return {
        key: (
            float(value)
            if (
                isinstance(value, numbers.Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
            else value
        )
        for key, value in metrics.items()
    }


def _format_metric(value: object, decimals: int = 3) -> str:
    if (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return f"{value:.{decimals}f}"
    return "n/a"


def _write_cell_artifacts(
    cell_dir: Path,
    graphs: list[Mapping[str, object]],
    metrics: Mapping[str, object],
) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    for graph, phase in zip(graphs, PHASES, strict=True):
        with (cell_dir / f"{phase}_graph.json").open("w", encoding="utf8") as handle:
            json.dump(graph, handle, indent=2)
    with (cell_dir / "metrics.json").open("w", encoding="utf8") as handle:
        json.dump(dict(metrics), handle, indent=2)
        handle.write("\n")


def _write_summary(
    summary_path: Path,
    rows: list[tuple[str, Mapping[str, object]]],
) -> None:
    with summary_path.open("w", encoding="utf8") as handle:
        header = (
            "| Setting | ΔAlign (if available) | ΔRepeat | ΔSparsity | "
            "Path | Branch | Repeat |\n"
        )
        separator = (
            "| ------- | --------------------- | -------- | ---------- | "
            "---- | ------ | ------ |\n"
        )
        handle.write(header)
        handle.write(separator)
        for name, metrics in rows:
            align_display = _format_metric(metrics.get("delta_alignment"))
            repeat_display = _format_metric(metrics.get("delta_repeatability"))
            sparsity_display = _format_metric(metrics.get("delta_sparsity"))
            path_display = _format_metric(metrics.get("avg_path_length"))
            branch_display = _format_metric(metrics.get("branching_factor"))
            rep_display = _format_metric(metrics.get("repeatability"))
            row_text = (
                f"| {name} | {align_display} | {repeat_display} | "
                f"{sparsity_display} | {path_display} | "
                f"{branch_display} | {rep_display} |\n"
            )
            handle.write(row_text)


def run_matrix(
    *,
    output_dir: Path,
    limit: int | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = output_dir / f"grokking_matrix_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, Mapping[str, object]]] = []
    total_cells = len(REGULARISATION) * len(STABILITY) * len(GRADIENT) * len(REASONING)
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        warnings.warn("--limit=0 produces an empty matrix run", RuntimeWarning, stacklevel=2)
    max_cells = min(total_cells, limit) if limit is not None else total_cells
    for idx, combo in enumerate(_build_combo_grid()):
        if idx >= max_cells:
            break
        name = _combo_name(combo)
        digest_text = hashlib.sha256(name.encode("utf8")).hexdigest()
        combo_digest = int(digest_text, 16) % (2**32)
        seed = (combo_digest ^ idx) % 10_000
        print(f"Processing cell {idx + 1}/{max_cells}: {name}", flush=True)
        graphs = _extract_phase_graphs(combo_digest, name)
        metrics = _summarise_metrics(graphs)
        # ``seed`` is recorded for bookkeeping; phase seeds incorporate the
        # combo digest directly inside ``_extract_phase_graphs``.
        metrics_with_seed = {**metrics, "seed": seed}
        cell_dir = root / name
        _write_cell_artifacts(cell_dir, graphs, metrics_with_seed)
        rows.append((name, metrics_with_seed))
    _write_summary(root / "summary.md", rows)
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="runs/matrix",
        help="Directory for experiment outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of matrix cells",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on number of matrix cells")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output).resolve()
    run_dir = run_matrix(output_dir=output_dir, limit=args.limit)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":  # pragma: no cover
    main()
