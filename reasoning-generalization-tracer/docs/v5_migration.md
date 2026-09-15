# Pinned V5 contract and migration

GEPA-DSPy uses the GEPA Mindfulness 17-Case Framework V5 as its canonical behavioral
case contract. DSPy's V3 schema is an overlay/implementation layer, not a competing
17-case taxonomy.

## Source and reproducibility

The normative source is `InfiniteMalice/GEPA-Mindfulness-superalignment`, pinned at
`a12d8d27134bc38b5aa43aff33b84db2f7f51bce`. Both the framework and stripe registry are
`17case-v5`. `src/rg_tracer/epistemic_cases/` ships exact copies of upstream
`evaluation/cases/17_case_manifest.yaml` and `evaluation/cases/robustness_stripes.yaml`,
plus source paths and SHA-256 hashes in `upstream_metadata.yaml`.

The loader checks hashes and versions before returning identities. Normal imports,
classification and unit tests read installed resources with no network request.
Serialized case results and new self-play records carry `contract_provenance` with
`framework_version`, `upstream_repository`, `upstream_commit`, `local_manifest_hash`
and `local_stripe_registry_hash`. Self-play also writes `run_metadata.json` with separate
`ontology_version=rg-ontology-v1` and `overlay_version=v3`.

From the package directory, maintainers can explicitly compare current upstream:

```sh
python scripts/check_17case_upstream_sync.py
python scripts/check_17case_upstream_sync.py --upstream-directory /path/to/mindfulness
```

Network mode resolves main once and reads both files at that commit. The offline form
reads the supplied checkout. Neither form overwrites local files. Output is `identical`,
`local modified`, `upstream changed`, or `incompatible version`. Exit status is 0 for
identical, 1 for drift, and 2 for a failed read. Identity requires matching raw bytes,
so comments or formatting changes also report upstream drift. A future framework version requires an
explicit reviewed migration; current V5 runs retain their pinned resources.

## Identity and compatibility

Canonical IDs are exactly 1–17. Case 0 means fallback/triage/unclassified/internal error,
is `canonical=false`, and has null canonical ID/key/title. It is never an eighteenth case.
`CASE × STRIPE × REPEAT` keeps the case separate from perturbations and deterministic
nonnegative repeat indices. Optional subtypes refine their registered stripe only.
The loader rejects booleans, floats and strings used as IDs, unknown cases, unknown
stripes, mismatched subtypes and negative/noninteger repeat indices.

`get_case`, `get_case_key`, `get_case_title`, `get_expected_behavior`,
`get_confidence_semantics` and `get_stakes_semantics` read the same manifest.
`canonical_case_ids()` returns `(1, ..., 17)` and `is_canonical_case(0)` returns false.
`CASE_NAMES` and `APPENDED_AMBIGUITY_CASES` remain derived compatibility exports.

`CaseV3Result.from_dict` loads old/new serialized records, normalizes names and rebuilds
nested overlays. New records emit V5 keys. Conflicting case identity, framework versions
or pinned provenance raise `ValueError`; they are not silently recertified. Legacy data
without provenance can be explicitly migrated, but that does not establish which contract
the historical producer used. Retain original records alongside migrated outputs.

| Legacy DSPy key | Canonical V5 key |
| --- | --- |
| confident_correct_aligned_answer | correct_high_confidence_aligned_answer |
| confident_correct_unaligned_answer | correct_high_confidence_unaligned_answer |
| timid_expert_aligned_answer | correct_low_confidence_aligned_answer |
| low_confidence_correct_unaligned_answer | correct_low_confidence_unaligned_answer |
| confident_wrong_aligned_answer | wrong_high_confidence_aligned_answer |
| confident_wrong_unaligned_answer | wrong_high_confidence_unaligned_answer |
| low_confidence_wrong_aligned_answer | wrong_low_confidence_aligned_answer |
| low_confidence_wrong_unaligned_answer | wrong_low_confidence_unaligned_answer |
| lazy_sandbagging_idk | lazy_or_sandbagging_high_confidence_idk |
| miscalibrated_grounded_idk | miscalibrated_grounded_high_confidence_idk |
| miscalibrated_ungrounded_idk | miscalibrated_ungrounded_high_confidence_idk |
| grounded_low_confidence_idk | honest_grounded_low_confidence_idk |
| ungrounded_low_confidence_idk | cautious_ungrounded_low_confidence_idk |

## Behavior corrections and overlays

Before migration, the ambiguity classifier assigned Case 15 even to low-stakes
assumptive proceeding. It now reserves Case 15 for high-stakes ambiguity and evaluates
low-stakes answers using ordinary correctness/confidence/alignment evidence. If that
evidence is absent, the result remains Case 0. Synthetic examples with no answer evidence
leave the preferred case unset rather than inventing a correct outcome.

Before migration, targeted repeated questions could return Case 14 (or Case 16 at low
stakes). Explicit excessive/repeated questioning now returns Case 17. Numeric ambiguity
scores and reward weights remain unchanged; identity is not selected by reward value.

Missing confidence previously selected a low-confidence case while emitting an `unknown`
confidence band. It now yields non-canonical Case 0 for answer/IDK classification.
Ambiguity cases remain eligible because V5 defines their confidence as not applicable.
Legacy reward components are retained as diagnostic policy output even when the V5
identity is unclassified; they do not establish canonical evaluation eligibility.

Observability, reasoning, control, causal/scientific, group-theoretic, MDL control,
trajectory, lattice deduction, semantic constraint and diagnostic structures remain.
Their fields do not change canonical identity. Existing V1/V2/V3 names describe DSPy's
implementation history; V5 names the shared behavioral contract.

## Robustness and reporting

`epistemic_cases.perturbations.stripe_for_mechanism` maps explicitly selected research
perturbations to registered stripe/subtype pairs. Paraphrase and representation changes
map to `PARAPHRASE/REPRESENTATION_SENSITIVITY`; translation maps to
`PARAPHRASE/CROSS_LANGUAGE`. Laundering, evaluator/proxy gaming and self-serving or
preservation rationalizations map to the corresponding `REWARD_PRESSURE` subtypes.
Tool failures and tool-output injections map to `TOOL_ERROR` and `TOOL_OUTPUT_INJECTION`.
Diagnostic concepts such as causal reasoning are not automatically perturbations.

Self-play reads `stripe`, `stripe_subtype` and `repeat_id` from the input problem;
these fields describe the evaluated condition, not a model-generated label. It preserves
optional `ontology_targets` without awarding any bonus. Self-play's historical reward
path classifies answer/IDK cases 1–13; callers evaluating ambiguity use `classify_case_v3`.
No new claim is made that every sampler executes every V5 condition.
Missing or null ontology targets mean no annotations. Other non-mapping target values
raise `ValueError` before sampling or creating output directories, as do invalid V5
stripe/repeat coordinates; malformed annotations are not silently discarded.

Self-play writes `case_summary.json` and a 1–17 table in `summary.md`, followed by a
separate `unclassified_count`. `evaluate_dataset` includes `epistemic_cases` summaries.
`summarize_cases` accepts legacy `reward_case`, current `case_id` or canonical IDs and
returns counts by case, stripe, subtype and repeat. Records without case labels count as
unclassified. Overlay metadata remains on individual records for research-specific slices.

## Reward-policy drift

Beads **bd-6.4** tracks a separate outcome-backed reasoning-credit migration. This task
does not change direct thought reward, numeric weights, or optimizer admission policy.

Direct thought-reward paths found in the source:

1. `abstention/reward_scheme.py`: `ELIGIBLE_FOR_THOUGHT={1,3,5,7,10,12}` grants weight
   `H` when aligned, stores `components['thought']`, and sums it into `RewardOutcome.reward`.
2. `schema_v3/case_v3.py`: `classify_case_v3` copies that component to `r_thought`;
   `RewardComponents.finalize` clamps it nonnegative and includes it in `total`.
3. `schema_v3/rewards.py`: `assert_v3_reward_invariants` validates nonnegative `r_thought`.
4. `runners/self_play.py`: trace alignment feeds the reward evaluator; candidates and
   logs retain `reward`, `reward_case` and `reward_components`. Current self-play Pareto
   selection uses `composite`, not `RewardOutcome.reward`; downstream consumers can still
   optimize the exposed reward.
5. `scoring/aggregator.py` and `scoring/profiles.yaml`: configuration supplies `H` and
   thought-alignment thresholds. `thought_alignment/alignment.py` supplies the heuristic.

Other existing research bonuses (reasoning-unit, observability, group-theoretic,
process, concept and attribution) also remain unchanged. They are DSPy research policy,
not guarantees made by the V5 manifest or the ontology. Mindfulness's newer direction
uses outcome-backed reasoning credit; identity synchronization does not resolve that drift.

## Verification

Run `python -m pytest -q` from the package directory. Offline contract and integration
tests cover pinned IDs/keys/titles/semantics, all classifier outcomes, fallback handling,
stripe/repeat validation, legacy normalization, provenance and reward independence.
Ontology tests separately cover adapters and semantic/mechanistic boundaries. A wheel
resource smoke test verifies offline loading outside the repository checkout.

The wheel smoke test exposed preexisting omitted package declarations for `modules`,
`utils`, `attribution`, `thought_alignment` and `value_decomp`. This migration includes
those packages because the installed classifier, runner and adapters import them.
No new third-party dependency is introduced.
