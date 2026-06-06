"""CPU-friendly modular-addition grokking trainer."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import append_jsonl, write_json
from .config import SpectralRegularizationConfig, ToyGrokkingConfig
from .regularizers import apply_low_rank_decay_
from .spectral_metrics import rank_collapse_ratio, require_torch, singular_value_summary
from .toy_transformer import ToyGrokkingTransformer, make_modular_addition_dataset


@dataclass
class GrokkingRunResult:
    """Summary for one toy grokking run."""

    metrics: list[dict[str, object]]
    spectra: list[dict[str, object]]
    summary: dict[str, object]


def _batched_indices(total: int, batch_size: int, seed: int, epoch: int) -> Any:
    torch_mod = require_torch()
    generator = torch_mod.Generator().manual_seed(seed + epoch)
    order = torch_mod.randperm(total, generator=generator)
    for start in range(0, total, batch_size):
        yield order[start : start + batch_size]


def _accuracy(logits: Any, labels: Any) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().item())


def _evaluate(model: Any, inputs: Any, labels: Any, loss_fn: Any) -> tuple[float, float]:
    torch_mod = require_torch()
    model.eval()
    with torch_mod.no_grad():
        logits = model(inputs)
        return float(loss_fn(logits, labels).item()), _accuracy(logits, labels)


def _selected_weight_summaries(model: Any, initial: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or not name.endswith(".weight"):
            continue
        if "q_proj" not in name and "k_proj" not in name:
            continue
        summary = singular_value_summary(parameter.detach())
        summary["matrix_name"] = name
        summary["rank_collapse_ratio"] = rank_collapse_ratio(initial[name], parameter.detach())
        rows.append(summary)
    return rows


def _phase_label(epoch: int, memorization: int | None, generalization: int | None) -> str:
    if memorization is None or epoch < memorization:
        return "pre_overfit"
    if generalization is None:
        return "overfit"
    if epoch < generalization:
        return "pre_grok"
    return "post_grok"


def train_toy_grokking(
    toy_config: ToyGrokkingConfig,
    regularization: SpectralRegularizationConfig | None = None,
    *,
    output_dir: Path | None = None,
) -> GrokkingRunResult:
    """Train a real toy transformer and log grokking/spectral metrics."""

    torch_mod = require_torch()
    regularization = regularization or SpectralRegularizationConfig()
    torch_mod.manual_seed(toy_config.seed)
    dataset = make_modular_addition_dataset(toy_config)
    device = torch_mod.device(toy_config.device)
    train_x = dataset.train_inputs.to(device)
    train_y = dataset.train_labels.to(device)
    test_x = dataset.test_inputs.to(device)
    test_y = dataset.test_labels.to(device)
    model = ToyGrokkingTransformer(toy_config).to(device)
    optimizer = torch_mod.optim.AdamW(
        model.parameters(),
        lr=toy_config.learning_rate,
        weight_decay=0.0,
    )
    loss_fn = torch_mod.nn.CrossEntropyLoss()
    initial_weights = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2 and name.endswith(".weight")
    }
    metrics: list[dict[str, object]] = []
    spectra_rows: list[dict[str, object]] = []
    memorization_epoch: int | None = None
    generalization_epoch: int | None = None
    start_time = time.perf_counter()

    for epoch in range(toy_config.epochs + 1):
        if epoch > 0:
            model.train()
            for batch_idx in _batched_indices(
                len(train_x), toy_config.batch_size, toy_config.seed, epoch
            ):
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x[batch_idx])
                loss = loss_fn(logits, train_y[batch_idx])
                loss.backward()
                optimizer.step()
                apply_low_rank_decay_(model.named_parameters(), regularization)
        if epoch % toy_config.eval_interval != 0 and epoch != toy_config.epochs:
            continue
        train_loss, train_acc = _evaluate(model, train_x, train_y, loss_fn)
        test_loss, test_acc = _evaluate(model, test_x, test_y, loss_fn)
        if memorization_epoch is None and train_acc >= 0.99:
            memorization_epoch = epoch
        if (
            memorization_epoch is not None
            and generalization_epoch is None
            and epoch >= memorization_epoch
            and test_acc >= 0.95
        ):
            generalization_epoch = epoch
        wall_clock = time.perf_counter() - start_time
        spectra = _selected_weight_summaries(model, initial_weights)
        for row in spectra:
            row = dict(row)
            row["epoch"] = epoch
            spectra_rows.append(row)
            if output_dir is not None:
                append_jsonl(output_dir / "spectra.jsonl", row)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "regularization_mode": regularization.mode,
            "l2_weight": regularization.l2_weight,
            "lrd_weight": regularization.lrd_weight,
            "spectral_summaries": spectra,
            "wall_clock_seconds": wall_clock,
            "seed": toy_config.seed,
            "memorization_epoch": memorization_epoch,
            "generalization_epoch": generalization_epoch,
            "grokking_delay": (
                generalization_epoch - memorization_epoch
                if memorization_epoch is not None and generalization_epoch is not None
                else None
            ),
            "phase_label": _phase_label(epoch, memorization_epoch, generalization_epoch),
        }
        metrics.append(record)
        if output_dir is not None:
            append_jsonl(output_dir / "metrics.jsonl", record)
    summary = _summary_from_metrics(toy_config, regularization, metrics)
    if output_dir is not None:
        write_json(
            output_dir / "config.json",
            {"toy": asdict(toy_config), "regularization": asdict(regularization)},
        )
    return GrokkingRunResult(metrics=metrics, spectra=spectra_rows, summary=summary)


def _summary_from_metrics(
    toy_config: ToyGrokkingConfig,
    regularization: SpectralRegularizationConfig,
    metrics: list[dict[str, object]],
) -> dict[str, object]:
    final = metrics[-1] if metrics else {}
    spectra = final.get("spectral_summaries", [])
    q_rank = None
    k_rank = None
    if isinstance(spectra, list):
        for row in spectra:
            if not isinstance(row, dict):
                continue
            name = str(row.get("matrix_name", ""))
            if "q_proj" in name and q_rank is None:
                q_rank = row.get("effective_rank")
            if "k_proj" in name and k_rank is None:
                k_rank = row.get("effective_rank")
    return {
        "mode": regularization.mode,
        "seed": toy_config.seed,
        "train_fraction": toy_config.train_fraction,
        "memorization_epoch": final.get("memorization_epoch"),
        "generalization_epoch": final.get("generalization_epoch"),
        "grokking_delay": final.get("grokking_delay"),
        "final_train_accuracy": final.get("train_accuracy"),
        "final_test_accuracy": final.get("test_accuracy"),
        "q_effective_rank": q_rank,
        "k_effective_rank": k_rank,
        "wall_clock_seconds": final.get("wall_clock_seconds"),
    }


__all__ = ["GrokkingRunResult", "train_toy_grokking"]
