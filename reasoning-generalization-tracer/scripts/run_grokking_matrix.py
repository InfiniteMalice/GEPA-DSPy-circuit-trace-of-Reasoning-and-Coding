"""Run attribution-aware grokking experiments across a configuration matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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
            if isinstance(value, numbers.Real) and not isinstance(value, bool)
            else value
        )
        for key, value in metrics.items()
    }


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
            align_value = metrics.get("delta_alignment")
            if isinstance(align_value, numbers.Real) and not isinstance(align_value, bool):
                align_display = f"{align_value:.3f}"
            else:
                align_display = "n/a"
            repeat = metrics.get("delta_repeatability")
            repeat_display = (
                f"{repeat:.3f}"
                if isinstance(repeat, numbers.Real) and not isinstance(repeat, bool)
                else "n/a"
            )
            sparsity_val = metrics.get("delta_sparsity")
            sparsity_display = (
                f"{sparsity_val:.3f}"
                if isinstance(sparsity_val, numbers.Real) and not isinstance(sparsity_val, bool)
                else "n/a"
            )
            path_val = metrics.get("avg_path_length")
            path_display = (
                f"{path_val:.3f}"
                if isinstance(path_val, numbers.Real) and not isinstance(path_val, bool)
                else "n/a"
            )
            branch_val = metrics.get("branching_factor")
            branch_display = (
                f"{branch_val:.3f}"
                if isinstance(branch_val, numbers.Real) and not isinstance(branch_val, bool)
                else "n/a"
            )
            rep_val = metrics.get("repeatability")
            rep_display = (
                f"{rep_val:.3f}"
                if isinstance(rep_val, numbers.Real) and not isinstance(rep_val, bool)
                else "n/a"
            )
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
    parser.add_argument("--limit", type=int, help="Optional limit on number of matrix cells")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output).resolve()
    run_dir = run_matrix(output_dir=output_dir, limit=args.limit)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":  # pragma: no cover
    main()
