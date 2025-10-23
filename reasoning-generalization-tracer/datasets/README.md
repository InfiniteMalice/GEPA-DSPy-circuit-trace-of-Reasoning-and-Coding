# Datasets

The JSONL files in this directory provide lightweight reasoning tasks used for
self-play, calibration, and transfer evaluation. Each line is a JSON object with
at least the following keys:

* `id`: Unique identifier for the instance.
* `prompt`: Natural language or symbolic problem statement.
* `answer`: Expected numeric or boolean answer.
* `task`: Task type (e.g. `addition`, `parity`, `carry`).
* `numbers` / `sequence`: Optional structured fields used by the Tiny Recursion
  Model sampler.

Additional fields may be included to describe concept targets or semantic
checks. The toy math set mirrors the TRM training distribution, while
`transfer_tests` introduces distribution shift such as modular arithmetic or
alternate bases to stress abstraction and generalization.
