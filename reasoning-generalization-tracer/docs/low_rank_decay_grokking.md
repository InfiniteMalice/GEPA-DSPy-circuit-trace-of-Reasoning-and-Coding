# Low-Rank Decay Grokking Experiments

RG-Tracer keeps two grokking-related paths separate:

- `scripts/run_grokking_matrix.py` is the existing lightweight null-backend
  attribution matrix. It emits deterministic graph-shape artifacts for smoke
  tests and does not claim real spectral collapse or grokking onset.
- `scripts/run_lrd_grokking_matrix.py` is an optional real CPU toy-transformer
  training experiment. It requires `pip install -e .[grokking]` and logs
  singular-value metrics from actual Q/K projection matrices.

The implementation is LRD-inspired: it applies a bounded decoupled update using
the polar factor of selected matrices, with exact SVD for correctness tests and
an optional Newton-Schulz approximation for experimental runs. Default targets
are `q_proj` and `k_proj`; LRD is never silently applied to every matrix.

The trainer uses a small modular-addition transformer with deterministic seeds,
RMSNorm-style scale stabilization, explicit Q/K/V projections, and CPU-friendly
defaults for tests. It logs train/test loss and accuracy, effective rank,
stable rank, spectral entropy, rank-collapse ratios, phase labels, and
memorization/generalization transition fields.

These runs are toy ablations. They do not prove frontier-model performance, do
not replace real attribution backends, and should not be described as a faithful
reproduction unless future work matches the reference setup closely.

Reference: [Low-Rank Decay for Grokking in Scale-Invariant Transformers](https://arxiv.org/abs/2606.04405).
