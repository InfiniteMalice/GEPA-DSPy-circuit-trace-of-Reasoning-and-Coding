"""Command line interface for rg-tracer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .concepts import ConceptSpec, compute_concept_reward, trace_model
from .fallback import run_academic_pipeline
from .recursive_refinement import (
    LatticeConfig,
    PerturbationConfig,
    RecursiveRefinementConfig,
    RefinementBudget,
    ViewRoutingConfig,
)
from .runners.eval_suite import evaluate_dataset
from .runners.self_play import run_self_play


def _make_concept(name: str | None) -> ConceptSpec | None:
    if name is None:
        return None
    return ConceptSpec(
        name=name,
        definition=f"Auto-generated concept for {name}",
        expected_substructures=[name],
        feature_catalog=[{"id": name, "tags": [name]}],
    )


def _cmd_self_play(args: argparse.Namespace) -> None:
    concept = _make_concept(args.concept)
    refinement_config = None
    if args.lattice_mode != "off" and args.sampler not in {
        "lattice_trm",
        "lattice_ptrm",
        "gram_mdt",
    }:
        raise SystemExit("--lattice-mode requires lattice_trm, lattice_ptrm, or gram_mdt")
    if args.sampler == "gram_mdt" and args.lattice_mode == "gated":
        raise SystemExit("gram_mdt may only use lattice diagnostics in off/shadow/advisory mode")
    if args.sampler in {"ptrm", "lattice_trm", "lattice_ptrm", "gram_mdt"}:
        refinement_config = RecursiveRefinementConfig(
            budget=RefinementBudget(
                max_depth=args.max_depth,
                max_width=args.max_width,
                max_total_updates=args.max_total_updates,
                branch_factor=args.branch_factor,
                seed=args.seed,
            ),
            routing=ViewRoutingConfig(enabled=not args.disable_view_routing),
            lattice=LatticeConfig(
                mode=args.lattice_mode,
                adapter=args.lattice_adapter,
                max_projection_steps=args.max_projection_steps,
            ),
            perturbation=PerturbationConfig(
                enabled=args.sampler in {"ptrm", "lattice_ptrm"},
                noise_std=args.noise_std,
                perturb_after_depth=args.perturb_after_depth,
                trajectories=args.trajectory_count,
                seed=args.seed,
            ),
            adaptive_halting=not args.disable_adaptive_halting,
            progressive_widening=not args.disable_progressive_widening,
            process_reward_weight=args.process_reward_weight,
        )
    result = run_self_play(
        args.problem,
        profile=args.profile,
        k=args.k,
        sampler=args.sampler,
        concept=concept,
        refinement_config=refinement_config,
        process_reward_weight=args.process_reward_weight,
    )
    print(
        json.dumps(
            {
                "run_dir": result["run_dir"],
                "best_composite": result["best"].composite,
            }
        )
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    summary = evaluate_dataset(args.dataset, args.profile, output_csv=args.output)
    print(json.dumps(summary, indent=2))


def _cmd_trace(args: argparse.Namespace) -> None:
    run_path = Path(args.run)
    with run_path.open("r", encoding="utf8") as handle:
        best = json.load(handle)
    concept = _make_concept(args.concept)
    trace = trace_model(args.model_ref, best.get("problem_id", "task"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.json"
    with trace_path.open("w", encoding="utf8") as handle:
        json.dump(trace.to_json(), handle, indent=2)
    if concept is not None:
        reward = compute_concept_reward(
            trace.to_json(),
            concept,
            task_metrics={"concept_reuse": 1.0},
        )
        print(json.dumps({"trace": str(trace_path), "concept_reward": reward}))
    else:
        print(json.dumps({"trace": str(trace_path)}))


def _cmd_humanities(args: argparse.Namespace) -> None:
    output = run_academic_pipeline(args.dataset, profile=args.profile)
    trimmed = {
        "summary": output["summary"],
        "records": [
            {
                "id": record.get("id"),
                "abstained": record.get("abstained"),
                "confidence": record.get("confidence"),
                "humanities": record.get("humanities"),
            }
            for record in output["records"]
        ],
    }
    print(json.dumps(trimmed, indent=2))


def _cmd_fallback(args: argparse.Namespace) -> None:
    output = run_academic_pipeline(args.problem, profile=args.profile)
    print(json.dumps(output, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rg-tracer")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("self-play", help="Run self-play on a problem")
    sp.add_argument("--profile", default="proof_math")
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--problem", required=True)
    sp.add_argument("--concept")
    sp.add_argument(
        "--sampler",
        choices=["trm", "ptrm", "lattice_trm", "lattice_ptrm", "gram_mdt"],
        default="trm",
    )
    sp.add_argument("--lattice-mode", choices=["off", "shadow", "advisory", "gated"], default="off")
    sp.add_argument(
        "--lattice-adapter",
        choices=["auto", "addition", "parity", "finite_domain_constraint"],
        default="auto",
    )
    sp.add_argument("--max-projection-steps", type=int, default=16)
    sp.add_argument("--noise-std", type=float, default=0.10)
    sp.add_argument("--perturb-after-depth", type=int, default=1)
    sp.add_argument("--trajectory-count", type=int, default=4)
    sp.add_argument("--max-depth", type=int, default=8)
    sp.add_argument("--max-width", type=int, default=4)
    sp.add_argument("--max-total-updates", type=int, default=64)
    sp.add_argument("--branch-factor", type=int, default=2)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--process-reward-weight", type=float, default=0.0)
    sp.add_argument("--disable-adaptive-halting", action="store_true")
    sp.add_argument("--disable-progressive-widening", action="store_true")
    sp.add_argument("--disable-view-routing", action="store_true")
    sp.set_defaults(func=_cmd_self_play)

    ev = sub.add_parser("eval", help="Evaluate datasets")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--profile", default="proof_math")
    ev.add_argument("--output")
    ev.set_defaults(func=_cmd_eval)

    tr = sub.add_parser("trace", help="Generate circuit traces")
    tr.add_argument("--run", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--concept")
    tr.add_argument("--model-ref", default="trm")
    tr.set_defaults(func=_cmd_trace)

    hum = sub.add_parser("humanities", help="Score humanities datasets")
    hum.add_argument("--dataset", required=True)
    hum.add_argument("--profile", default="humanities")
    hum.set_defaults(func=_cmd_humanities)

    fb = sub.add_parser("fallback", help="Run academic fallback pipeline")
    fb.add_argument("--problem", required=True)
    fb.add_argument("--profile", default="humanities")
    fb.set_defaults(func=_cmd_fallback)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
