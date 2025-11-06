"""Run attribution-aware grokking experiments across a configuration matrix."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from rg_tracer.attribution import graphs as attr_graphs
from rg_tracer.attribution import metrics as attr_metrics

REGULARISATION = ["no_wd", "wd"]
STABILITY = ["softmax", "stablemax"]
GRADIENT = ["off", "perpend_grad_on"]
REASONING = ["sft_only", "srl_pretrain_then_sft"]
PHASES = ["pre_overfit", "overfit", "pre_grok", "post_grok"]
BACKEND_CHOICES = ["null", "hooked", "external"]


def _combo_name(combo: Mapping[str, str]) -> str:
    return "_".join(f"{key}={value}" for key, value in combo.items())


def _build_combo_grid() -> Iterable[Mapping[str, str]]:
    keys = ["regularisation", "stability", "gradient", "reasoning"]
    values = [REGULARISATION, STABILITY, GRADIENT, REASONING]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=True))


def _extract_phase_graphs(
    backend_name: str,
    seed_offset: int,
) -> List[Mapping[str, object]]:
    backend = attr_graphs.get_backend(backend_name)
    graphs: List[Mapping[str, object]] = []
    for index, phase in enumerate(PHASES):
        payload = {
            "task_id": f"phase_{phase}_{seed_offset}",
            "tokens": [index, seed_offset],
            "phase": phase,
            "probe_index": index,
        }
        graph = attr_graphs.extract_graph(
            model=None,
            inputs=payload,
            backend=backend,
            seed=seed_offset + index,
        )
        graphs.append(graph)
    return graphs


def _summarise_metrics(
    graphs: List[Mapping[str, object]],
) -> Dict[str, float]:
    metrics = {
        "sparsity": attr_metrics.path_sparsity(graphs),
        "avg_path_length": attr_metrics.average_path_length(graphs),
        "branching_factor": attr_metrics.average_branching_factor(graphs),
        "repeatability": attr_metrics.repeatability(graphs),
        "delta_sparsity": attr_metrics.delta_sparsity(graphs),
        "delta_alignment": attr_metrics.delta_alignment(graphs, concept_features=None),
        "delta_repeatability": attr_metrics.delta_repeatability(graphs),
    }
    return {key: float(value) for key, value in metrics.items()}


def _write_cell_artifacts(
    cell_dir: Path,
    graphs: List[Mapping[str, object]],
    metrics: Mapping[str, float],
) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    for graph, phase in zip(graphs, PHASES, strict=True):
        with (cell_dir / f"{phase}_graph.json").open("w", encoding="utf8") as handle:
            json.dump(graph, handle, indent=2)
    with (cell_dir / "metrics.json").open("w", encoding="utf8") as handle:
        json.dump(dict(metrics), handle, indent=2)


def _write_summary(
    summary_path: Path,
    rows: List[Tuple[str, Mapping[str, float]]],
) -> None:
    with summary_path.open("w", encoding="utf8") as handle:
        handle.write("| Setting | ΔAlign | ΔRepeat | ΔSparsity | Path | Branch | Repeat |\n")
        handle.write("| ------- | ------ | -------- | ---------- | ---- | ------ | ------ |\n")
        for name, metrics in rows:
            handle.write(
                "| {name} | {align:.3f} | {repeat:.3f} | {sparsity:.3f} | {path:.3f} | {branch:.3f} | {rep:.3f} |\n".format(
                    name=name,
                    align=metrics.get("delta_alignment", 0.0),
                    repeat=metrics.get("delta_repeatability", 0.0),
                    sparsity=metrics.get("delta_sparsity", 0.0),
                    path=metrics.get("avg_path_length", 0.0),
                    branch=metrics.get("branching_factor", 0.0),
                    rep=metrics.get("repeatability", 0.0),
                )
            )


def run_matrix(
    *,
    output_dir: Path,
    backend: str,
    limit: int | None = None,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    root = output_dir / f"grokking_matrix_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, Mapping[str, float]]] = []
    for idx, combo in enumerate(_build_combo_grid()):
        if limit is not None and idx >= limit:
            break
        name = _combo_name(combo)
        digest = hashlib.sha256(name.encode("utf8")).hexdigest()
        seed = int(digest, 16) % 10_000
        graphs = _extract_phase_graphs(backend, seed)
        metrics = _summarise_metrics(graphs)
        cell_dir = root / name
        _write_cell_artifacts(cell_dir, graphs, metrics)
        rows.append((name, metrics))
    _write_summary(root / "summary.md", rows)
    return root


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/matrix", help="Directory for experiment outputs")
    parser.add_argument(
        "--backend",
        default="null",
        choices=BACKEND_CHOICES,
        help="Attribution backend to use",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on number of matrix cells")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output).resolve()
    run_dir = run_matrix(output_dir=output_dir, backend=args.backend, limit=args.limit)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":  # pragma: no cover
    main()
